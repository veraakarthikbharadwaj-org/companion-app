-- Reference: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase#create-a-table-and-search-function-in-your-database
-- Visit Supabase blogpost for more: https://supabase.com/blog/openai-embeddings-postgres-vector
-- Enable the pgvector extension to work with embedding vectors
create extension vector;

-- Create a table to store your documents
create table documents (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1536) -- 1536 works for OpenAI embeddings, change if needed
);

-- Audit log table for AI-driven vector search decisions
-- Retention policy: records must be kept for a minimum of 90 days.
-- This table is append-only; DELETE and UPDATE are blocked by rules below.
create table if not exists match_documents_audit (
  audit_id      bigserial primary key,
  logged_at     timestamptz not null default now(),
  principal     text        not null,
  model_id      text        not null,
  model_version text        not null default 'unknown',  -- e.g. '002', '3.5-turbo-0125'
  input_hash    text        not null,  -- SHA-256 hex of the serialised query_embedding
  match_count   int,
  filter        jsonb,
  output        jsonb        -- serialised result set returned to the caller
);

-- Retention enforcement: purge audit rows older than 90 days.
-- Schedule this function via pg_cron:  SELECT cron.schedule('0 2 * * *', $$SELECT purge_match_documents_audit()$$);
create or replace function purge_match_documents_audit(
  retention_days int default 90
) returns void
language plpgsql
security definer
as $$
begin
  -- Temporarily disable the no_delete_audit rule so the purge can proceed.
  set local session_replication_role = replica;
  delete from match_documents_audit
  where logged_at < now() - (retention_days || ' days')::interval;
end;
$$;

-- Append-only enforcement: block DELETE and UPDATE on the audit table.
do $rules$
begin
  -- Block DELETE
  if not exists (
    select 1 from pg_rules
    where tablename = 'match_documents_audit'
      and rulename   = 'no_delete_audit'
  ) then
    execute $r$
      create rule no_delete_audit as
        on delete to match_documents_audit
        do instead nothing
    $r$;
  end if;

  -- Block UPDATE
  if not exists (
    select 1 from pg_rules
    where tablename = 'match_documents_audit'
      and rulename   = 'no_update_audit'
  ) then
    execute $r$
      create rule no_update_audit as
        on update to match_documents_audit
        do instead nothing
    $r$;
  end if;
end;
$rules$;

-- Create a function to search for documents
-- Returns provenance metadata, a synthetic-origin label, and a per-row
-- HMAC-SHA256 signature so callers can verify content integrity.
create function match_documents (
  query_embedding vector(1536),
  match_count     int     DEFAULT null,
  filter          jsonb   DEFAULT '{}',
  -- Caller-supplied model identifier (e.g. 'text-embedding-ada-002').
  -- Must be a non-empty string matching an entry in the approved model registry.
  model_id        text,
  -- Caller-supplied principal / user identifier for audit purposes.
  principal       text    DEFAULT 'anonymous',
  -- HMAC secret used to sign each result row.  Override via GUC
  -- app.match_documents_hmac_secret before calling this function.
  hmac_secret     text    DEFAULT current_setting('app.match_documents_hmac_secret', true)
) returns table (
  content_summary  text,
  similarity       float,
  -- Provenance columns -------------------------------------------------
  provenance_model text,        -- model that produced the embedding
  retrieved_at     timestamptz, -- wall-clock time of this retrieval
  content_origin   text,        -- label: always 'ai-generated-retrieval'
  row_signature    text         -- HMAC-SHA256(content_summary||similarity||model||ts)
)
language plpgsql
as $$
#variable_conflict use_column
declare
  v_now        timestamptz := clock_timestamp();
  v_input_hash text;
  v_secret     text;
begin
  -- Validate query_embedding
  if query_embedding is null then
    raise exception 'query_embedding must not be null';
  end if;

  -- Validate match_count
  if match_count is not null and (match_count < 1 or match_count > 1000) then
    raise exception 'match_count must be between 1 and 1000, got %', match_count;
  end if;

  -- Validate filter: must be a JSON object (not array, null, or scalar)
  if filter is null then
    filter := '{}'::jsonb;
  end if;
  if jsonb_typeof(filter) <> 'object' then
    raise exception 'filter must be a JSON object, got %', jsonb_typeof(filter);
  end if;

  -- Validate model_id: must be non-empty and present in the approved model registry.
  if model_id is null or model_id = '' then
    raise exception 'model_id must not be null or empty; supply an approved model identifier';
  end if;
  if model_id not in (
    'text-embedding-ada-002',
    'text-embedding-3-small',
    'text-embedding-3-large'
  ) then
    raise exception 'model_id ''%'' is not in the approved model registry', model_id;
  end if;

  -- Normalise optional parameters
  if principal is null or principal = '' then principal := 'anonymous'; end if;

  -- Resolve HMAC secret; raise an error if it has not been configured so
  -- the function never operates with an absent or empty secret.
  v_secret := nullif(hmac_secret, '');
  if v_secret is null then
    raise exception 'HMAC secret is not configured. Set app.match_documents_hmac_secret before calling this function.';
  end if;

  -- Hash the serialised query embedding for the audit record.
  v_input_hash := encode(
    digest(query_embedding::text, 'sha256'),
    'hex'
  );

  -- Populate the audit table before returning results.
  insert into match_documents_audit
    (logged_at, principal, model_id, input_hash, match_count, filter)
  values
    (v_now, principal, model_id, v_input_hash, match_count, filter);

  -- Return results with provenance metadata, origin label, and signature.
  return query
  select
    left(d.content, 500)                              as content_summary,
    1 - (d.embedding <=> query_embedding)             as similarity,
    -- Provenance metadata
    model_id                                          as provenance_model,
    v_now                                             as retrieved_at,
    'ai-generated-retrieval'::text                    as content_origin,
    -- Per-row HMAC-SHA256 signature over key fields
    encode(
      hmac(
        left(d.content, 500)
          || '|' || (1 - (d.embedding <=> query_embedding))::text
          || '|' || model_id
          || '|' || v_now::text,
        v_secret,
        'sha256'
      ),
      'hex'
    )                                                 as row_signature
  from documents d
  where d.metadata @> filter
  order by d.embedding <=> query_embedding
  limit match_count;
end;
$$;
