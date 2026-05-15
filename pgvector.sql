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
create table if not exists match_documents_audit (
  audit_id    bigserial primary key,
  logged_at   timestamptz not null default now(),
  principal   text        not null,
  model_id    text        not null,
  input_hash  text        not null,  -- SHA-256 hex of the serialised query_embedding
  match_count int,
  filter      jsonb
);

-- Create a function to search for documents
create function match_documents (
  query_embedding vector(1536),
  match_count int DEFAULT null,
  filter jsonb DEFAULT '{}'
) returns table (
  content_summary text,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
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

  return query
  select
    left(content, 500) as content_summary,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
