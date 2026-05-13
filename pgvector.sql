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
create table if not exists match_documents_audit_log (
  id bigserial primary key,
  logged_at timestamptz not null default now(),
  principal text not null,
  match_count int,
  filter jsonb,
  result_ids bigint[],
  result_count int
);

-- Create a function to search for documents
create function match_documents (
  query_embedding vector(1536),
  match_count int DEFAULT null,
  filter jsonb DEFAULT '{}'
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
begin
  -- Collect results into a temporary array for audit logging
  select
    array_agg(d.id order by d.embedding <=> query_embedding),
    count(*)
  into v_result_ids, v_result_count
  from documents d
  where d.metadata @> filter
  limit match_count;

  -- Write audit log entry before returning results
  insert into match_documents_audit_log (
    logged_at,
    principal,
    match_count,
    filter,
    result_ids,
    result_count
  ) values (
    now(),
    current_user,
    match_count,
    filter,
    v_result_ids,
    v_result_count
  );

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
