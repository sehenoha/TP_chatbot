"""Stockage vectoriel (PostgreSQL + pgvector) — fourni.

Chaque chunk d'un document a une ligne dans la table `chunk_embeddings` :

    document_id | chunk_size | overlap | chunk_index | content (texte du chunk) | embedding (vecteur)

À l'ajout d'un document, toutes ses lignes sont créées avec un **vecteur factice** (que des 1, c'est-à-dire `np.ones(1024)`) : c'est à
VOUS de calculer les vrais embeddings et de les enregistrer avec `set_embeddings`.

Fonctions fournies :
    get_chunks(...)          -> les chunks stockés (texte, contexte généré par le LLM, et vecteur si demandé)
    set_embeddings(items)    -> enregistre des vecteurs : items = [(document_id, chunk_index, vecteur), ...]
    set_embedding(...)       -> pareil pour un seul chunk
    embedded_indices(doc_id) -> indices des chunks dont le vecteur n'est plus le vecteur factice

Pour la recherche, vous écrivez vous-même le SQL avec `db.connect()` : pgvector fournit les opérateurs de
distance `<->` (euclidienne), `<#>` (produit scalaire négatif) et `<=>` (cosinus). Exemple de forme :

    with db.connect() as conn:
        rows = conn.execute("SELECT chunk_index, content FROM chunk_embeddings "
                            "ORDER BY embedding <-> %s::vector LIMIT 5", ("[0.1, 0.2, ...]",)).fetchall()

⚠️ Tant que les vecteurs sont factices, ils sont tous identiques : la recherche donne des égalités parfaites (distance 0 pour tous
les chunks) et l'ordre des résultats est arbitraire.
"""
import math

import db
import tools

DIM = db.EMBEDDING_DIM
CHUNK_SIZE, OVERLAP = 1000, 200      # réglages du découpage indexé (les mêmes que la contextualisation automatique)


def to_literal(vector) -> str:
    """Liste de nombres -> littéral pgvector '[0.1,0.2,...]' (vérifie la dimension et les valeurs)."""
    values = [float(x) for x in vector]
    if len(values) != DIM:
        raise ValueError(f"vecteur de dimension {len(values)} : la table attend des vecteurs de dimension {DIM}")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("le vecteur contient NaN ou une valeur infinie")
    return "[" + ",".join(repr(x) for x in values) + "]"


def index_document(doc_id: int, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> int:
    """Crée une ligne par chunk du document, avec un vecteur factice (des 1). Renvoie le nombre de chunks.
    Sans effet sur les chunks déjà présents (leur vecteur n'est pas écrasé)."""
    if not db.vectors_enabled():
        return 0
    doc = db.get_document(doc_id)
    if doc is None:
        return 0
    try:
        text = tools.extract_text({**doc, "data": bytes(doc["data"])})
    except ValueError as e:
        print(f"[vectorstore] {doc['name']} : pas de chunks indexés ({e})", flush=True)
        return 0
    chunks = tools.chunk_text(text, chunk_size, overlap)
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO chunk_embeddings (document_id, chunk_size, overlap, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, array_fill(1::real, ARRAY[{DIM}])::vector)
                    ON CONFLICT DO NOTHING""",
                [(doc_id, chunk_size, overlap, i, c) for i, c in enumerate(chunks)],
            )
    return len(chunks)


def get_chunks(doc_id: int | None = None, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP,
               with_embedding: bool = False) -> list[dict]:
    """Les chunks stockés : [{"document_id", "source", "chunk_index", "content", "context"} (+ "embedding")].

    `context` est le contexte généré par le LLM (contextual retrieval), ou None s'il n'existe pas encore.
    `embedding` (si demandé) est une liste de floats. `doc_id=None` : tous les documents.
    """
    if not db.vectors_enabled():
        return []
    emb = ", e.embedding::real[] AS embedding" if with_embedding else ""
    where, params = "e.chunk_size = %s AND e.overlap = %s", [chunk_size, overlap]
    if doc_id is not None:
        where += " AND e.document_id = %s"
        params.append(doc_id)
    with db.connect() as conn:
        return conn.execute(
            f"""SELECT e.document_id, d.name AS source, e.chunk_index, e.content, c.context{emb}
                  FROM chunk_embeddings e
                  JOIN documents d ON d.id = e.document_id
                  LEFT JOIN chunk_contexts c
                         ON c.document_id = e.document_id AND c.chunk_size = e.chunk_size
                        AND c.overlap = e.overlap AND c.chunk_index = e.chunk_index
                 WHERE {where}
                 ORDER BY e.document_id, e.chunk_index""",
            params,
        ).fetchall()


def set_embeddings(items, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> int:
    """Enregistre des vecteurs. `items` : [(document_id, chunk_index, vecteur), ...]. Renvoie le nombre de lignes mises à jour."""
    rows = [(to_literal(vec), doc_id, chunk_size, overlap, index) for doc_id, index, vec in items]
    updated = 0
    with db.connect() as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """UPDATE chunk_embeddings SET embedding = %s::vector
                        WHERE document_id = %s AND chunk_size = %s AND overlap = %s AND chunk_index = %s""",
                    row,
                )
                updated += cur.rowcount
    return updated


def set_embedding(doc_id: int, chunk_index: int, vector, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> bool:
    return set_embeddings([(doc_id, chunk_index, vector)], chunk_size, overlap) == 1


def embedded_indices(doc_id: int, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[int]:
    """Indices des chunks du document dont le vecteur n'est plus le vecteur factice (des 1)."""
    if not db.vectors_enabled():
        return []
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT chunk_index FROM chunk_embeddings
                WHERE document_id = %s AND chunk_size = %s AND overlap = %s AND embedding <> array_fill(1::real, ARRAY[%s])::vector
                ORDER BY chunk_index""",
            (doc_id, chunk_size, overlap, DIM),
        ).fetchall()
    return [r["chunk_index"] for r in rows]
