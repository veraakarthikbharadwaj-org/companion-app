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
  audit_id    bigserial primary key,
  logged_at   timestamptz not null default now(),
  principal   text        not null,
  model_id    text        not null,
  input_hash  text        not null,  -- SHA-256 hex of the serialised query_embedding
  match_count int,
  filter      jsonb
);

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
  -- Defaults to 'unknown' when not provided.
  model_id        text    DEFAULT 'unknown',
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

  -- Normalise optional parameters
  if model_id  is null or model_id  = '' then model_id  := 'unknown';   end if;
  if principal is null or principal = '' then principal := 'anonymous'; end if;

  -- Resolve HMAC secret (fall back to a fixed sentinel so the column is
  -- never empty; callers SHOULD set app.match_documents_hmac_secret).
  v_secret := coalesce(nullif(hmac_secret, ''), 'UNSET-CONFIGURE-app.match_documents_hmac_secret');

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
