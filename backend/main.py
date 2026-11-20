"""API du chatbot — fournie, vous n'avez normalement pas à la modifier.

Le seul fichier à compléter est `ai.py`.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import autocontext
import contextual
import db
import embedding
import llm
import models
import recipes
import tools
import vectorstore
from ai import generate_reply

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 Mo


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    autocontext.resume_unfinished()     # reprend les contextualisations interrompues
    embedding.resume_pending()          # calcule les embeddings encore manquants (appelle ai.embed)
    yield


app = FastAPI(title="Chatbot TP2", lifespan=lifespan)


class Message(BaseModel):
    role: str       # "user" ou "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]   # historique complet de la conversation
    model: str | None = None  # modèle choisi dans l'interface (sinon LLM_CHATBOT)
    reasoning: str | None = None  # niveau de raisonnement du curseur : low, medium, high, xhigh


class ChatResponse(BaseModel):
    reply: str
    recipes: list[dict] = []    # recettes citées (titre exact) dans la réponse : une bulle à côté de chaque titre
    servings: int | None = None # nombre de personnes de la conversation : les fiches s'ouvrent avec ces quantités
    thinking: str | None = None # réflexion du modèle (raisonnement), affichée dans un bloc dépliable


def decode_text(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def get_documents():
    return db.list_documents()


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile]):
    saved = []
    for f in files:
        name = Path(f.filename or "document").name
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(413, f"{name} dépasse {MAX_FILE_SIZE // (1024 * 1024)} Mo")
        doc_id = db.insert_document(name, f.content_type, data)
        saved.append(doc_id)
        try:
            vectorstore.index_document(doc_id)    # une ligne par chunk, avec un vecteur factice (des 1), à remplacer par vos embeddings
        except Exception as e:                    # ne jamais faire échouer l'envoi du fichier
            print(f"[vectorstore] indexation impossible pour {name} : {e}", flush=True)
        autocontext.start(doc_id)         # contextualisation automatique, en tâche de fond
    return {"saved": saved, "documents": db.list_documents()}


@app.get("/api/documents/{doc_id}/download")
def download_document(doc_id: int):
    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document introuvable")
    return Response(
        content=bytes(doc["data"]),
        media_type=doc["content_type"] or "application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(doc['name'])}"},
    )


def _load_text(doc_id: int) -> tuple[dict, str]:
    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document introuvable")
    doc = {**doc, "data": bytes(doc["data"])}
    try:
        return doc, tools.extract_text(doc)
    except ValueError as e:
        raise HTTPException(422, str(e))


def _spans(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    try:
        return tools.chunk_spans(text, chunk_size, overlap)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/documents/{doc_id}/chunks")
def document_chunks(doc_id: int, chunk_size: int = 1000, overlap: int = 200):
    """Aperçu du texte extrait et de son découpage (utile pour régler chunk_size et overlap),
    avec les contextes déjà générés (contextual retrieval)."""
    doc, text = _load_text(doc_id)
    chunks = [text[a:b] for a, b in _spans(text, chunk_size, overlap)]
    return {
        "name": doc["name"],
        "n_chars": len(text),
        "n_chunks": len(chunks),
        "chunks": chunks,
        "contexts": db.get_contexts(doc_id, chunk_size, overlap),
        "embedded": vectorstore.embedded_indices(doc_id, chunk_size, overlap),    # chunks dont le vecteur n'est plus le vecteur factice
        "vector_store": db.vectors_enabled(),
        "context_model": contextual.MODEL,
        "api_key_set": llm.api_key_set(),
    }


@app.get("/api/contextualization")
def contextualization_status():
    """État de la contextualisation automatique de chaque document : {id du document: {status, total, done, ...}}."""
    return {j["document_id"]: j for j in db.list_jobs()}


@app.post("/api/documents/{doc_id}/contextualize")
def contextualize_document(doc_id: int):
    """(Re)lance la contextualisation de tous les chunks d'un document ; reprend là où elle s'était arrêtée."""
    if db.get_document(doc_id) is None:
        raise HTTPException(404, "Document introuvable")
    autocontext.start(doc_id, force=True)
    return {"started": True}


@app.get("/api/embedding")
def embedding_status():
    """État du calcul des embeddings de chaque document : {id: {running, total, done, failed, stub, error}}."""
    return embedding.status()


@app.post("/api/documents/{doc_id}/embed")
def embed_document(doc_id: int, force: bool = False):
    """(Re)calcule les embeddings des chunks avec `ai.embed`. `force=true` recalcule aussi ceux déjà calculés."""
    if db.get_document(doc_id) is None:
        raise HTTPException(404, "Document introuvable")
    return {"started": embedding.start(doc_id, force=force)}


@app.get("/api/documents/{doc_id}/embedded")
def embedded_chunks(doc_id: int, chunk_size: int = 1000, overlap: int = 200):
    """Indices des chunks dont le vecteur n'est plus le vecteur factice."""
    return {"embedded": vectorstore.embedded_indices(doc_id, chunk_size, overlap)}


@app.get("/api/documents/{doc_id}/contexts")
def document_contexts(doc_id: int, chunk_size: int = 1000, overlap: int = 200):
    """Contextes déjà générés (léger : sans recalculer le découpage)."""
    return {"contexts": db.get_contexts(doc_id, chunk_size, overlap)}


@app.post("/api/documents/{doc_id}/chunks/{index}/context")
def generate_chunk_context(doc_id: int, index: int, chunk_size: int = 1000, overlap: int = 200):
    """Génère (ou régénère) avec le LLM le contexte d'un chunk, et le mémorise."""
    doc, text = _load_text(doc_id)
    spans = _spans(text, chunk_size, overlap)
    if not 0 <= index < len(spans):
        raise HTTPException(404, f"Chunk {index} inexistant (le document en compte {len(spans)})")
    start, end = spans[index]
    recipe = recipes.parse_recipe(text[start:end])
    try:
        if recipe:      # une recette : le « contexte » est sa description, rangée aussi dans la table `recipes`
            d = recipes.describe(recipe)      # (re)génération à la demande par le LLM
            result = {"context": d["description"], "system_prompt": recipes.SYSTEM_PROMPT,
                      "prompt": recipes.recipe_text(recipe), "window": None, "usage": None, "model": d["model"]}
        else:
            result = contextual.situate_chunk(text, spans, index, name=doc["name"])
    except llm.LLMError as e:
        raise HTTPException(502, str(e))
    if not result["context"]:
        raise HTTPException(502, "Le modèle a renvoyé un contexte vide, réessayez.")
    db.save_context(doc_id, chunk_size, overlap, index, result["model"], result["context"])
    if recipe and (chunk_size, overlap) == (autocontext.CHUNK_SIZE, autocontext.OVERLAP):
        recipes.save(doc_id, index, recipe, result["context"], result["model"])
    return {"index": index, **result, "contextualized": contextual.contextualize_text(result["context"], text[start:end])}


@app.get("/api/recipes/{doc_id}/{chunk_index}/image")
def recipe_image(doc_id: int, chunk_index: int):
    """Redirige vers la photo de la recette (lue sur la page Marmiton la première fois) ; 404 s'il n'y en a pas."""
    url = recipes.image_url(doc_id, chunk_index)
    if not url:
        raise HTTPException(404, "Pas de photo pour cette recette")
    return RedirectResponse(url)


@app.get("/api/recipes/{doc_id}/{chunk_index}")
def get_recipe(doc_id: int, chunk_index: int, servings: int | None = None):
    """Une recette structurée ; `?servings=4` adapte les quantités à 4 personnes."""
    recipe = recipes.get(doc_id, chunk_index, servings)
    if recipe is None:
        raise HTTPException(404, "Recette introuvable (document pas encore contextualisé, ou pas une recette)")
    return recipe


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    if not db.delete_document(doc_id):
        raise HTTPException(404, "Document introuvable")
    return {"documents": db.list_documents()}


@app.get("/api/models")
def list_models():
    """Modèles proposés dans le menu du chat (models.py) et modèle par défaut (LLM_CHATBOT)."""
    return models.available()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.model and not models.is_allowed(req.model):
        raise HTTPException(400, f"Modèle non proposé : {req.model} (voir backend/models.py)")
    try:
        level, api_effort = models.resolve_reasoning(req.model or llm.CHATBOT_MODEL, req.reasoning)
    except ValueError as e:
        raise HTTPException(400, str(e))
    tokens = llm.select_model(req.model, level, api_effort)
    try:
        response = _chat(req)
        response.thinking = "\n\n".join(llm.thoughts()) or None
        return response
    finally:
        llm.unselect(tokens)


def _chat(req: ChatRequest) -> ChatResponse:
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    documents = []
    for d in db.list_documents(with_data=True):
        data = bytes(d["data"])
        documents.append({
            "id": d["id"],
            "name": d["name"],
            "content_type": d["content_type"],
            "size": d["size"],
            "data": data,                 # contenu brut (bytes)
            "text": decode_text(data),    # contenu texte (str), ou None si le fichier n'est pas du texte
        })
    reply = generate_reply(messages, documents)
    try:
        cited = recipes.find_in_text(reply)
    except Exception as e:                  # jamais bloquant pour la réponse
        print(f"[chat] recherche des recettes citées impossible : {e}", flush=True)
        cited = []
    servings = recipes.guess_servings(reply, [m["content"] for m in messages if m["role"] == "user"]) if cited else None
    return ChatResponse(reply=reply, recipes=cited, servings=servings)
