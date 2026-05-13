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

-- Audit log table for AI-driven vector similarity searches
-- Retention policy: records older than 90 days must be purged.
-- Configure a scheduled job (e.g. pg_cron) to run:
--   DELETE FROM match_documents_audit_log WHERE logged_at < now() - interval '90 days';
-- For high-volume deployments consider range-partitioning by logged_at and
-- dropping old partitions instead of row-level deletes.
create table if not exists match_documents_audit_log (
  id bigserial primary key,
  logged_at timestamptz not null default now(),
  principal text not null,
  model_id text not null default 'unknown',   -- model identifier / version
  input_hash text not null default '',        -- md5 of serialised query_embedding
  match_count int,
  filter jsonb,
  result_ids bigint[],
  result_count int,
  log_error text                              -- non-null when the primary INSERT failed
);
comment on table match_documents_audit_log is
  'Forensic audit trail for AI vector-search actions. Retention: 90 days. '
  'Rotate via scheduled DELETE or partition drop.';

-- Create a function to search for documents
create function match_documents (
  query_embedding vector(1536),
  match_count int DEFAULT null,
  filter jsonb DEFAULT '{}',
  model_id text DEFAULT 'text-embedding-ada-002'  -- caller supplies model identifier/version
) returns table (
  content_snippet text,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
declare
  v_result_ids bigint[];
  v_result_count int;
  v_input_hash text;
begin
  -- Compute a deterministic hash of the query embedding for forensic traceability
  v_input_hash := md5(query_embedding::text);

  -- Collect results into a temporary array for audit logging
  select
    array_agg(d.id order by d.embedding <=> query_embedding),
    count(*)
  into v_result_ids, v_result_count
  from documents d
  where d.metadata @> filter
  limit match_count;

  -- Write audit log entry; catch errors so a logging failure never silently
  -- aborts the function and is itself recorded for later investigation.
  begin
    insert into match_documents_audit_log (
      logged_at,
      principal,
      model_id,
      input_hash,
      match_count,
      filter,
      result_ids,
      result_count
    ) values (
      now(),
      current_user,
      model_id,
      v_input_hash,
      match_count,
      filter,
      v_result_ids,
      v_result_count
    );
  exception when others then
    -- Fallback: record the failure itself so the audit trail is never silently lost
    insert into match_documents_audit_log (
      logged_at,
      principal,
      model_id,
      input_hash,
      match_count,
      filter,
      result_ids,
      result_count,
      log_error
    ) values (
      now(),
      current_user,
      model_id,
      v_input_hash,
      match_count,
      filter,
      v_result_ids,
      v_result_count,
      sqlerrm
    );
  end;

  -- Return the actual query results
  return query
  select
    id,
    content,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
