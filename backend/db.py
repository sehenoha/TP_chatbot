"""Accès à PostgreSQL — fourni, vous n'avez normalement pas à le modifier.

Une seule table, `documents`, qui contient les fichiers chargés depuis l'interface.
Vous pouvez l'explorer avec :  docker compose exec db psql -U chatbot -d chatbot
"""
import os
import time

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://chatbot@db:5432/chatbot")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    name         TEXT        NOT NULL,
    content_type TEXT,
    size         INTEGER     NOT NULL,
    data         BYTEA       NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Contextes générés par le LLM (contextual retrieval), un par chunk et par réglage de découpage
CREATE TABLE IF NOT EXISTS chunk_contexts (
    document_id INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_size  INTEGER     NOT NULL,
    overlap     INTEGER     NOT NULL,
    chunk_index INTEGER     NOT NULL,
    model       TEXT        NOT NULL,
    context     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, chunk_size, overlap, chunk_index)
);

-- Suivi de la contextualisation automatique d'un document (une ligne par document)
CREATE TABLE IF NOT EXISTS contextualization (
    document_id INTEGER     PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    chunk_size  INTEGER     NOT NULL,
    overlap     INTEGER     NOT NULL,
    status      TEXT        NOT NULL,      -- running | done | partial | error | skipped
    total       INTEGER     NOT NULL DEFAULT 0,
    failed      INTEGER     NOT NULL DEFAULT 0,
    message     TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Recettes de cuisine : une ligne par recette, sous forme structurée (voir recipes.py)
CREATE TABLE IF NOT EXISTS recipes (
    document_id      INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      INTEGER     NOT NULL,       -- le chunk qui contient la recette (1 chunk = 1 recette)
    title            TEXT        NOT NULL,
    category         TEXT,
    duration_minutes INTEGER,
    duration_text    TEXT,
    servings         INTEGER,
    source_url       TEXT,
    description      TEXT        NOT NULL,       -- le bloc **Description** du .md (ou générée par le LLM)
    model            TEXT        NOT NULL,
    ingredients      JSONB       NOT NULL,       -- [{"raw", "quantity", "unit", "name"}, ...]
    utensils         JSONB       NOT NULL,       -- ustensiles bruts du site
    utensils_norm    TEXT[]      NOT NULL DEFAULT '{}',   -- ustensiles normalisés (recipes.UTENSILS), pour filtrer en SQL
    steps            JSONB       NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, chunk_index)
);
ALTER TABLE recipes ADD COLUMN IF NOT EXISTS utensils_norm TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE recipes ADD COLUMN IF NOT EXISTS image_url TEXT;          -- photo (lue sur Marmiton à la demande) ; '' = pas de photo
CREATE INDEX IF NOT EXISTS recipes_utensils_idx ON recipes USING GIN (utensils_norm);
"""

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))      # mistral-embed produit des vecteurs de dimension 1024
VECTORS_ENABLED = False                                           # passe à True si l'extension pgvector est disponible

# Stockage vectoriel : un chunk par ligne, avec son texte et son vecteur (des 1 tant qu'on n'a pas calculé l'embedding)
VECTOR_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    document_id INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_size  INTEGER     NOT NULL,
    overlap     INTEGER     NOT NULL,
    chunk_index INTEGER     NOT NULL,
    content     TEXT        NOT NULL,                  -- le texte du chunk
    embedding   vector({EMBEDDING_DIM}) NOT NULL,      -- le vecteur : des 1 (np.ones) au départ !
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, chunk_size, overlap, chunk_index)
);
"""

META_COLUMNS = "id, name, content_type, size, created_at"


def connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db(retries: int = 30):
    """Crée les tables ; réessaie quelques secondes si PostgreSQL démarre encore."""
    for attempt in range(retries):
        try:
            with connect() as conn:
                conn.execute(SCHEMA)
            _init_vectors()
            return
        except psycopg.OperationalError:
            if attempt == retries - 1:
                raise
            time.sleep(1)


def _init_vectors() -> None:
    """Active le stockage vectoriel. Si pgvector manque (ancienne image PostgreSQL), le reste de l'appli fonctionne."""
    global VECTORS_ENABLED
    try:
        with connect() as conn:
            conn.execute(VECTOR_SCHEMA)
        VECTORS_ENABLED = True
    except psycopg.Error as e:
        VECTORS_ENABLED = False
        print(f"[db] pgvector indisponible, stockage vectoriel désactivé : {e}\n"
              "     -> utilisez l'image pgvector/pgvector:pg16 (docker-compose.yml) puis `docker compose down -v && docker compose up`.",
              flush=True)


def vectors_enabled() -> bool:
    """Vrai si la table des vecteurs existe (donc si pgvector est disponible). Vérifié à la demande : fonctionne
    aussi dans un script ou un autre processus que le serveur."""
    global VECTORS_ENABLED
    if VECTORS_ENABLED:
        return True
    try:
        with connect() as conn:
            VECTORS_ENABLED = conn.execute("SELECT to_regclass('chunk_embeddings') AS t").fetchone()["t"] is not None
    except psycopg.Error:
        VECTORS_ENABLED = False
    return VECTORS_ENABLED


def list_documents(with_data: bool = False) -> list[dict]:
    columns = META_COLUMNS + (", data" if with_data else "")
    with connect() as conn:
        return conn.execute(f"SELECT {columns} FROM documents ORDER BY created_at, id").fetchall()


def get_document(doc_id: int) -> dict | None:
    with connect() as conn:
        return conn.execute(f"SELECT {META_COLUMNS}, data FROM documents WHERE id = %s", (doc_id,)).fetchone()


def insert_document(name: str, content_type: str | None, data: bytes) -> int:
    with connect() as conn:
        row = conn.execute(
            "INSERT INTO documents (name, content_type, size, data) VALUES (%s, %s, %s, %s) RETURNING id",
            (name, content_type, len(data), data),
        ).fetchone()
        return row["id"]


def delete_document(doc_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,)).rowcount > 0


def get_contexts(doc_id: int, chunk_size: int, overlap: int) -> dict[int, str]:
    """Contextes déjà générés pour ce document et ce découpage : {index du chunk: contexte}."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT chunk_index, context FROM chunk_contexts WHERE document_id = %s AND chunk_size = %s AND overlap = %s",
            (doc_id, chunk_size, overlap),
        ).fetchall()
    return {r["chunk_index"]: r["context"] for r in rows}


def save_context(doc_id: int, chunk_size: int, overlap: int, index: int, model: str, context: str) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO chunk_contexts (document_id, chunk_size, overlap, chunk_index, model, context)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (document_id, chunk_size, overlap, chunk_index)
               DO UPDATE SET model = EXCLUDED.model, context = EXCLUDED.context, created_at = now()""",
            (doc_id, chunk_size, overlap, index, model, context),
        )


def set_job(doc_id: int, chunk_size: int, overlap: int, status: str, total: int = 0, failed: int = 0,
            message: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO contextualization (document_id, chunk_size, overlap, status, total, failed, message)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (document_id) DO UPDATE SET chunk_size = EXCLUDED.chunk_size, overlap = EXCLUDED.overlap,
                   status = EXCLUDED.status, total = EXCLUDED.total, failed = EXCLUDED.failed,
                   message = EXCLUDED.message, updated_at = now()""",
            (doc_id, chunk_size, overlap, status, total, failed, message),
        )


def job_exists(doc_id: int) -> bool:
    """Faux si le document a été supprimé entre-temps (la ligne disparaît avec lui)."""
    with connect() as conn:
        return conn.execute("SELECT 1 FROM contextualization WHERE document_id = %s", (doc_id,)).fetchone() is not None


def list_jobs(only_running: bool = False) -> list[dict]:
    """Toutes les tâches, avec `done` = nombre de chunks déjà contextualisés."""
    where = "WHERE j.status = 'running'" if only_running else ""
    with connect() as conn:
        return conn.execute(
            f"""SELECT j.document_id, j.chunk_size, j.overlap, j.status, j.total, j.failed, j.message,
                       (SELECT count(*) FROM chunk_contexts c
                         WHERE c.document_id = j.document_id AND c.chunk_size = j.chunk_size
                           AND c.overlap = j.overlap)::int AS done
                  FROM contextualization j {where} ORDER BY j.document_id"""
        ).fetchall()
