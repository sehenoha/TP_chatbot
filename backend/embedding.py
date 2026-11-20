"""Calcul des embeddings des chunks — fourni. Appelle VOTRE fonction `ai.embed`.

Quand ? Automatiquement :
    1. à l'ajout d'un document : une fois les contextes générés (contextual retrieval), le texte de chaque chunk
       (contexte + chunk) est transformé en vecteur par `ai.embed`, puis enregistré dans PostgreSQL ;
    2. au (re)démarrage du serveur, pour les chunks dont le vecteur est encore factice : en enregistrant
       votre fichier `ai.py`, le serveur redémarre tout seul et vos embeddings sont calculés pour les
       documents déjà chargés ;
    3. à la demande : bouton « Calculer les embeddings » (fenêtre d'un document) ou
       `POST /api/documents/{id}/embed`.

Tant que `ai.embed` renvoie le vecteur factice (np.ones), il n'y a rien à enregistrer : on le signale dans
les logs (`docker compose logs backend`) et dans la fenêtre du document.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import ai
import contextual
import db
import vectorstore

WORKERS = max(1, int(os.environ.get("AUTO_CONTEXT_WORKERS", "64")))       # mêmes appels parallèles que la contextualisation

_lock = threading.Lock()
_state: dict[int, dict] = {}     # id du document -> {running, total, done, failed, stub, error}


def status() -> dict[int, dict]:
    with _lock:
        return {k: dict(v) for k, v in _state.items()}


def _is_placeholder(vector: np.ndarray) -> bool:
    return vector.shape == (vectorstore.DIM,) and bool(np.allclose(vector, 1.0))


def _embed_chunk(chunk: dict) -> np.ndarray:
    """Appelle ai.embed sur le texte contextualisé du chunk et vérifie le résultat."""
    text = contextual.contextualize_text(chunk["context"] or "", chunk["content"])
    vector = np.asarray(ai.embed(text), dtype=float)
    if vector.shape != (vectorstore.DIM,):
        raise ValueError(f"embed() doit renvoyer un vecteur de dimension {vectorstore.DIM}, pas de forme {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("embed() a renvoyé un vecteur contenant NaN ou une valeur infinie")
    return vector


def start(doc_id: int, force: bool = False) -> bool:
    """Lance le calcul des embeddings d'un document en tâche de fond (sans effet s'il est déjà en cours).
    `force=True` recalcule aussi les chunks dont le vecteur a déjà été calculé."""
    with _lock:
        if _state.get(doc_id, {}).get("running"):
            return False
        _state[doc_id] = {"running": True, "total": 0, "done": 0, "failed": 0, "stub": False, "error": None}
    threading.Thread(target=_run, args=(doc_id, force), daemon=True, name=f"embed-{doc_id}").start()
    return True


def _update(doc_id: int, **fields) -> None:
    with _lock:
        _state.setdefault(doc_id, {}).update(fields)


def _run(doc_id: int, force: bool) -> None:
    try:
        _process(doc_id, force)
    except Exception as e:
        _update(doc_id, error=f"{type(e).__name__}: {e}"[:300])
        print(f"[embedding] document {doc_id} : {type(e).__name__}: {e}", flush=True)
    finally:
        _update(doc_id, running=False)


def _process(doc_id: int, force: bool) -> None:
    if not db.vectors_enabled():
        _update(doc_id, error="stockage vectoriel indisponible (pgvector)")
        return
    chunks = vectorstore.get_chunks(doc_id)
    if not chunks:
        return
    already = set() if force else set(vectorstore.embedded_indices(doc_id))
    todo = [c for c in chunks if c["chunk_index"] not in already]
    _update(doc_id, total=len(todo), done=0, failed=0)
    if not todo:
        return

    # Le premier chunk part seul : si embed() renvoie encore le vecteur factice, inutile de continuer.
    first = _embed_chunk(todo[0])
    if _is_placeholder(first):
        _update(doc_id, stub=True)
        print("[embedding] ai.embed() renvoie encore np.ones : rien à enregistrer (TODO dans backend/ai.py).", flush=True)
        return
    vectorstore.set_embedding(doc_id, todo[0]["chunk_index"], first)
    _update(doc_id, done=1)

    def work(chunk: dict) -> None:
        try:
            vectorstore.set_embedding(doc_id, chunk["chunk_index"], _embed_chunk(chunk))
            with _lock:
                _state[doc_id]["done"] += 1
        except Exception as e:
            with _lock:
                st = _state[doc_id]
                st["failed"] += 1
                if st["error"] is None:
                    st["error"] = f"{type(e).__name__}: {e}"[:300]
                    print(f"[embedding] chunk {chunk['chunk_index']} du document {doc_id} : {type(e).__name__}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(work, todo[1:]))


def resume_pending() -> None:
    """Au démarrage du serveur : calcule les embeddings manquants des documents déjà chargés (en tâche de fond).
    C'est ce qui fait que vos embeddings sont calculés dès que vous enregistrez `ai.py` et que le serveur redémarre."""
    def scan():
        try:
            for doc in db.list_documents():
                n_chunks = len(vectorstore.get_chunks(doc["id"]))
                if n_chunks and len(vectorstore.embedded_indices(doc["id"])) < n_chunks:
                    start(doc["id"])
        except Exception as e:
            print(f"[embedding] reprise impossible : {e}", flush=True)
    threading.Thread(target=scan, daemon=True, name="embed-resume").start()
