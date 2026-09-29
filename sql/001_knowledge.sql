CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kb_documents (
  corpus_id text NOT NULL,
  source_path text NOT NULL,
  title text NOT NULL,
  visibility text NOT NULL CHECK (visibility = 'public'),
  PRIMARY KEY (corpus_id, source_path)
);
CREATE TABLE IF NOT EXISTS kb_chunks (
  corpus_id text NOT NULL,
  id text NOT NULL,
  source_path text NOT NULL,
  title text NOT NULL,
  content text NOT NULL,
  content_hash text NOT NULL,
  search_vector tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('portuguese', title), 'A') ||
    setweight(to_tsvector('portuguese', content), 'B')
  ) STORED,
  PRIMARY KEY (corpus_id, id),
  FOREIGN KEY (corpus_id, source_path) REFERENCES kb_documents(corpus_id, source_path)
);
CREATE INDEX IF NOT EXISTS kb_chunks_fts ON kb_chunks USING gin(search_vector);
CREATE TABLE IF NOT EXISTS kb_embeddings (
  corpus_id text NOT NULL,
  chunk_id text NOT NULL,
  model text NOT NULL,
  execution_kind text NOT NULL CHECK (execution_kind IN ('live', 'fixture')),
  embedding vector(1536) NOT NULL,
  PRIMARY KEY(corpus_id, chunk_id, model, execution_kind),
  FOREIGN KEY(corpus_id, chunk_id) REFERENCES kb_chunks(corpus_id, id)
);
-- Exact cosine search is used for the bounded experimental corpus.
-- No approximate index: all eligible vectors participate in ranking.
