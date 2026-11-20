"""Contextualisation automatique des documents — fournie.

Quand un document est ajouté, ce module génère en tâche de fond le contexte de TOUS ses chunks
(voir `contextual.py`), sans bloquer l'envoi du fichier. La progression est suivie en base
(table `contextualization`) et affichée dans la page « Documents ».

Réglages (variables d'environnement, voir `.env.example`) :
    AUTO_CONTEXTUALIZE=true        mettre `false` pour désactiver (génération manuelle uniquement)
    AUTO_CONTEXT_MAX_CHUNKS=5000   au-delà, le document est ignoré (garde-fou contre une facture surprise)
    AUTO_CONTEXT_WORKERS=64        nombre d'appels au LLM en parallèle (le débit reste borné par llm.RateLimiter)
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import contextual
import db
import embedding
import llm
import recipes
import tools

ENABLED = os.environ.get("AUTO_CONTEXTUALIZE", "true").strip().lower() not in ("0", "false", "no", "non", "off")
MAX_CHUNKS = int(os.environ.get("AUTO_CONTEXT_MAX_CHUNKS", "5000"))
WORKERS = max(1, int(os.environ.get("AUTO_CONTEXT_WORKERS", "64")))
CHUNK_SIZE, OVERLAP = 1000, 200      # mêmes valeurs par défaut que la fenêtre de visualisation des chunks

_lock = threading.Lock()
_active: set[int] = set()            # documents en cours de traitement dans ce processus


def start(doc_id: int, force: bool = False) -> None:
    """Lance (ou reprend) la contextualisation d'un document. Sans effet si elle est déjà en cours."""
    if not ENABLED and not force:
        embedding.start(doc_id)          # pas de contexte : on calcule directement les embeddings des chunks bruts
        return
    with _lock:
        if doc_id in _active:
            return
        _active.add(doc_id)
    db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "running")
    threading.Thread(target=_run, args=(doc_id,), daemon=True, name=f"contextualize-{doc_id}").start()


def resume_unfinished() -> None:
    """Au démarrage du serveur : reprend les tâches interrompues (redémarrage, rechargement du code...)."""
    if not ENABLED:
        return
    for job in db.list_jobs(only_running=True):
        start(job["document_id"], force=True)


def _run(doc_id: int) -> None:
    try:
        _process(doc_id)
        if db.get_document(doc_id) is not None:      # contextes prêts (ou impossibles) : on calcule les embeddings
            embedding.start(doc_id)
    except Exception as e:                                   # ne jamais laisser une tâche « running » à l'infini
        try:
            db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "error", message=f"{type(e).__name__}: {e}"[:300])
        except Exception:
            pass
    finally:
        with _lock:
            _active.discard(doc_id)


def _process(doc_id: int) -> None:
    doc = db.get_document(doc_id)
    if doc is None:
        return
    if not llm.api_key_set():
        db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "error",
                   message="MISTRAL_API_KEY est vide : renseignez-la dans .env puis relancez `docker compose up`.")
        return
    try:
        text = tools.extract_text({**doc, "data": bytes(doc["data"])})
    except ValueError as e:
        db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "error", message=str(e)[:300])
        return

    spans = tools.chunk_spans(text, CHUNK_SIZE, OVERLAP)
    total = len(spans)
    if total > MAX_CHUNKS:
        db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "skipped", total=total,
                   message=f"{total} chunks : au-delà de la limite automatique ({MAX_CHUNKS}). "
                           "Contextualisez à la main, ou augmentez AUTO_CONTEXT_MAX_CHUNKS dans .env.")
        return

    already = db.get_contexts(doc_id, CHUNK_SIZE, OVERLAP)
    todo = [i for i in range(total) if i not in already]
    db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "running", total=total)

    state = {"failed": 0, "ok": 0, "fatal": None, "last_error": None}

    def work(i: int) -> None:
        if state["fatal"] is not None:
            return
        try:
            if not db.job_exists(doc_id):                    # document supprimé : on arrête
                state["fatal"] = "supprimé"
                return
            a, b = spans[i]
            recipe = recipes.parse_recipe(text[a:b])
            if recipe:        # une recette : rangée sous forme structurée (table `recipes`), pas de contexte à générer
                d = ({"description": recipe["description"], "model": "fichier"} if recipe["description"]
                     else recipes.describe(recipe))        # description déjà dans le .md, sinon le LLM la rédige
                if not d["description"]:
                    raise llm.LLMError("description vide")
                recipes.save(doc_id, i, recipe, d["description"], d["model"])
                db.save_context(doc_id, CHUNK_SIZE, OVERLAP, i, d["model"],
                                f"Recette « {recipe['title']} » ({recipe['category'] or 'cuisine'}).")
                state["ok"] += 1
                return
            result = contextual.situate_chunk(text, spans, i, name=doc["name"])
            if not result["context"]:
                raise llm.LLMError("contexte vide")
            db.save_context(doc_id, CHUNK_SIZE, OVERLAP, i, result["model"], result["context"])
            state["ok"] += 1
        except llm.AuthError as e:
            state["fatal"] = str(e)
        except Exception as e:
            state["failed"] += 1
            state["last_error"] = f"{type(e).__name__}: {e}"[:200]
            if state["failed"] >= 10 and state["ok"] == 0:   # rien ne marche (quota épuisé ?) : on s'arrête
                state["fatal"] = f"Trop d'échecs d'affilée, dernier message : {state['last_error']}"

    if todo:
        work(todo[0])      # le premier chunk part seul : le client y apprend la limite de débit de la clé (llm.py)
        todo = todo[1:]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(work, todo))

    if state["fatal"] == "supprimé" or not db.job_exists(doc_id):
        return
    if state["fatal"]:
        db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "error", total=total, failed=state["failed"], message=state["fatal"][:300])
    elif state["failed"]:
        db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "partial", total=total, failed=state["failed"],
                   message=f"{state['failed']} chunk(s) en échec (dernier : {state['last_error']}). Relancez pour réessayer.")
    else:
        db.set_job(doc_id, CHUNK_SIZE, OVERLAP, "done", total=total)
