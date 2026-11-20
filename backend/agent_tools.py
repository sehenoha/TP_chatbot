"""⭐ Les OUTILS de l'agent (TODO 8) : le second fichier sur lequel vous travaillez, avec ai.py. ⭐

(Ne pas confondre avec `tools.py`, qui lit et découpe les documents et que vous n'avez pas à modifier.)

Principe du *tool calling* : on décrit au modèle des fonctions Python (nom, rôle, paramètres, en JSON) ; au lieu
de répondre directement, il peut demander « appelle `search_recipes` avec ces arguments » ; votre code exécute la
fonction, renvoie le résultat au modèle, et celui-ci rédige la réponse (ou rappelle un autre outil : c'est la
boucle **agentique**). Il peut aussi répondre par une **question** à l'utilisateur quand il lui manque une
information : « Vous avez quoi comme ustensiles ? » — l'historique est de toute façon renvoyé à `generate_reply`
à chaque message, la réponse de l'utilisateur arrivera au tour suivant.

Exemple de scénario : « Je veux faire un gâteau » → le modèle appelle `list_utensils`, constate qu'il lui faut
savoir ce que vous avez, vous demande « Vous avez un four ? un mixeur ? », puis appelle `search_recipes(...)`.

Les recettes sont dans la table PostgreSQL `recipes` (voir `recipes.py`) ; leurs ustensiles sont **normalisés** dans
la colonne `utensils_norm` (tableau de textes : 'four', 'saladier', 'fouet'...) pour pouvoir filtrer en SQL :

    SELECT title, duration_minutes FROM recipes
    WHERE utensils_norm <@ ARRAY['four', 'saladier', 'fouet', 'balance', 'couteau']   -- <@ : « inclus dans »
    ORDER BY duration_minutes LIMIT 5;
"""
import db
import recipes


def list_utensils() -> list[str]:
    """Le vocabulaire normalisé des ustensiles (fourni)."""
    return recipes.UTENSILS


def search_recipes(utensils_available: list[str] | None = None, max_minutes: int | None = None,
                   category: str | None = None, limit: int = 5) -> list[dict]:
    """Recherche de recettes par filtres SQL : à écrire.

    Renvoie des recettes (dicts avec au moins `title`, `category`, `duration_minutes`, `servings`,
    `document_id`, `chunk_index`) dont **tous les ustensiles requis** sont dans `utensils_available`.
    """
    # TODO : écrire la requête SQL avec db.connect() :
    #   - ustensiles : `utensils_norm <@ %s::text[]` (tous les ustensiles de la recette sont disponibles) ;
    #   - durée : `duration_minutes <= %s` ; catégorie : `category = %s` ;
    #   - ne mettre une condition que si le paramètre est renseigné ; paramètres passés à psycopg (jamais de f-string !).
    # Idées d'amélioration : combiner avec la recherche vectorielle (description) ; filtrer sur les ingrédients (JSONB).
    return []


# Les outils décrits au modèle (format « function calling » de l'API Mistral).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_utensils",
            "description": "Liste les ustensiles de cuisine connus (noms normalisés), pour savoir quoi demander à l'utilisateur.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # TODO : décrire `search_recipes` (nom, description claire, paramètres et leur type), et d'autres outils si besoin
    #        (`get_recipe` pour une recette complète adaptée à un nombre de personnes : voir recipes.get).
]

# nom de l'outil -> fonction Python à exécuter
FUNCTIONS = {
    "list_utensils": list_utensils,
    "search_recipes": search_recipes,
    "get_recipe": recipes.get,
}


def call_tool(name: str, arguments: dict):
    """Exécute l'outil demandé par le modèle (ses `arguments` arrivent déjà décodés en dict)."""
    if name not in FUNCTIONS:
        return {"error": f"outil inconnu : {name}"}
    try:
        return FUNCTIONS[name](**arguments)
    except Exception as e:                 # l'erreur est renvoyée au modèle, qui peut corriger son appel
        return {"error": f"{type(e).__name__}: {e}"}
