"""⭐ C'est ICI que vous travaillez : la partie IA du chatbot. ⭐

`generate_reply` reçoit :

1. `messages` : tout l'historique de la conversation, sous la forme

       [
           {"role": "user",      "content": "Bonjour !"},
           {"role": "assistant", "content": "Bonjour, comment puis-je vous aider ?"},
           {"role": "user",      "content": "Que dit l'article 2 du Code civil ?"},
       ]

2. `documents` : les fichiers chargés par l'utilisateur dans la page « Documents » (ou avec le
   bouton « + » du chat), lus dans la base PostgreSQL. Liste vide s'il n'y en a pas :

       [
           {"id": 1, "name": "civil.txt", "content_type": "text/plain", "size": 12345,
            "data": b"...",          # contenu brut (bytes)
            "text": "Article 1 ..."}, # contenu décodé en texte, ou None si ce n'est pas du texte
           ...
       ]

   Le fichier `tools.py` vous fournit de quoi exploiter ces documents :

       from tools import extract_text, chunk_text, load_chunks

       extract_text(doc)                 # texte d'un document (PDF, .docx, .txt, .md, ...)
       chunk_text(text, 1000, 200)       # découpe un texte en morceaux de ~1000 caractères
       load_chunks(documents)            # les deux, pour tous les documents :
                                         # [{"doc_id", "source", "chunk_id", "text"}, ...]

   Le contexte de chaque chunk (contextual retrieval) est généré automatiquement à l'ajout d'un document :

       import db
       from contextual import contextualize_text
       contexts = db.get_contexts(doc["id"], 1000, 200)   # {index du chunk: contexte}
       contextualize_text(contexts.get(i, ""), chunk)     # texte à indexer : contexte + chunk

   `llm.py` fournit aussi `chat(messages)` pour appeler le modèle Mistral.

   Fichier d'outils pour l'agent (tool calling, recherche SQL) : `agent_tools.py`. Recettes structurées : `recipes.py`.

   Les chunks, leur contexte et une table de vecteurs (des `np.ones` pour l'instant) sont dans PostgreSQL :

       import vectorstore
       vectorstore.get_chunks()                          # [{"document_id", "source", "chunk_index", "content", "context"}, ...]
       vectorstore.set_embeddings([(doc_id, chunk_index, vecteur), ...])   # vecteurs de dimension 1024
       vectorstore.embedded_indices(doc_id)              # chunks dont le vecteur n'est plus le vecteur factice

La fonction doit renvoyer le texte de la réponse du chatbot (une `str`).

Le travail, dans l'ordre des TODO :
    1. un chatbot « classique » : generate_reply appelle le LLM avec l'historique (sans les documents) ;
    2-7. le RAG : embed -> vector_search -> keyword_search -> fuse -> rerank, puis les chunks retrouvés dans le prompt ;
    8. l'agent : tool calling (agent_tools.py).
Tant que vous n'avez rien codé, le chatbot répond "TEST".
"""
import json
import os
import urllib.request

import numpy as np

import db
import llm
import recipes
import vectorstore
from db import EMBEDDING_DIM        # dimension des vecteurs stockés : 1024 (mistral-embed)

SYSTEM_PROMPT = ("Tu es un assistant culinaire bienveillant. Réponds toujours en français, de façon claire et concise. "
                 "Si tu ne sais pas, dis-le plutôt que d'inventer.")


# ---------------------------------------------------------------------------------------------- Génération
def generate_reply(messages: list[dict], documents: list[dict]) -> str:
    question = messages[-1]["content"]

    # Récupération (RAG) : déjà câblée ; elle renvoie [] tant que les fonctions des TODO 2 à 6 ne sont pas codées.
    chunks = rerank(question, fuse(vector_search(question), keyword_search(question)))

    # Étape 1 - Chatbot classique : consignes (message « system ») + tout l'historique.
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    try:
        return llm.chat([{"role": "system", "content": SYSTEM_PROMPT}] + history,
                        max_tokens=700, temperature=0.3)["content"]
    except llm.LLMError as e:
        return f"Erreur : {e}"

    # TODO 7 - RAG : mettre les chunks retenus dans le prompt, demander au modèle de répondre en français à partir de ces
    #          passages uniquement et de citer ses sources (nom de la recette, article...).
    #          Astuce : si la réponse contient le TITRE EXACT d'une recette, l'interface affiche sa fiche (photo, étapes...)
    #          sous la réponse : inutile de recopier les étapes ou d'ajouter des liens.
    #          Que doit répondre le chatbot si `chunks` est vide, ou si les chunks ne répondent pas à la question ?
    #
    # TODO 8 - Agent : tool calling (voir agent_tools.py). Donner les outils (TOOLS) au modèle, exécuter ceux qu'il demande
    #          avec call_tool(...), lui renvoyer les résultats, recommencer jusqu'à la réponse finale (au plus
    #          llm.agent_max_steps() tours : 3 à 16 selon le curseur « Raisonnement » de l'interface). Le modèle peut aussi
    #          répondre par une question à l'utilisateur quand il lui manque une information (« Vous avez quoi comme ustensiles ? »).
    #          Avant une recherche dans les documents (RAG), faites reformuler la requête par le petit modèle
    #          (llm.chat(..., model=llm.CONTEXT_MODEL), variable LLM_CONTEXT) : des mots-clés plutôt qu'une phrase de conversation.
    #          Une demande peut contenir plusieurs besoins (« un repas complet » = entrée + plat + dessert) : laissez le
    #          modèle lancer PLUSIEURS requêtes (multi-query), une par besoin, et chercher chacune séparément.
    return "TEST"


# ---------------------------------------------------------------------------------------------- Embeddings
def embed(text: str) -> np.ndarray:
    """Transforme un texte en vecteur (embedding) : un tableau numpy de dimension EMBEDDING_DIM (1024).

    Sert à vectoriser les chunks (texte contextualisé) ET les questions des utilisateurs : les deux doivent
    être calculés avec le même modèle, sinon les distances n'ont aucun sens.
    """
    # Pour une recette, on vectorise titre + description (courts, proches des mots d'un utilisateur) plutôt que
    # la recette entière (ingrédients, quantités, étapes : beaucoup de bruit).
    recipe = recipes.parse_recipe(text)
    if recipe and recipe["description"]:
        text = f"{recipe['title']}. {recipe['description']}"

    request = urllib.request.Request(
        "https://api.mistral.ai/v1/embeddings",
        data=json.dumps({"model": "mistral-embed", "input": [text[:8000]]}).encode(),
        headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}", "Content-Type": "application/json"})
    llm.limiter.acquire()                          # respecte la limite de requêtes par minute (appels en parallèle)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    return np.asarray(data["data"][0]["embedding"], dtype=float)


# ---------------------------------------------------------------------------------------------- Recherche
def vector_search(query: str, k: int = 20) -> list[dict]:
    """Les k chunks les plus proches de la question, par similarité de vecteurs.

    Renvoie une liste triée du plus au moins proche de dicts
    {"document_id", "source", "chunk_index", "content"} (même forme que vectorstore.get_chunks).
    """
    literal = vectorstore.to_literal(embed(query))
    with db.connect() as conn:
        return conn.execute(
            f"""SELECT e.document_id, d.name AS source, e.chunk_index, e.content
                  FROM chunk_embeddings e JOIN documents d ON d.id = e.document_id
                 WHERE e.chunk_size = %s AND e.overlap = %s
                   AND e.embedding <> array_fill(1::real, ARRAY[{EMBEDDING_DIM}])::vector
                 ORDER BY e.embedding <=> %s::vector
                 LIMIT %s""",
            (vectorstore.CHUNK_SIZE, vectorstore.OVERLAP, literal, k)).fetchall()


def keyword_search(query: str, k: int = 20) -> list[dict]:
    """Les k chunks qui correspondent le mieux aux mots de la question (recherche lexicale). Même format que ci-dessus."""
    # TODO 4 : recherche plein texte PostgreSQL (to_tsvector('french', content) @@ to_tsquery(...), ts_rank) ou autre méthode.
    #          Les mots de la question sont-ils tous nécessaires (AND) ou un seul suffit-il (OR) ?
    return []


# ---------------------------------------------------------------------------------------------- Fusion, re-classement
def fuse(*rankings: list[dict]) -> list[dict]:
    """Combine plusieurs classements de chunks en un seul (sans doublon), du meilleur au moins bon."""
    # TODO 5 : par exemple Reciprocal Rank Fusion : chaque chunk reçoit 1/(60 + rang) dans chaque liste, on additionne et on trie.
    #          Un chunk est identifié par (document_id, chunk_index).
    return []


def rerank(query: str, chunks: list[dict], keep: int = 4) -> list[dict]:
    """Ne garde que les `keep` chunks les plus pertinents pour la question."""
    # TODO 6 : par exemple demander au LLM (llm.chat) de classer les candidats, ou simplement garder les premiers.
    return chunks[:keep]
