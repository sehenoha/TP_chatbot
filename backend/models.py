"""Modèles proposés dans le menu à côté de la zone de saisie du chat, et leur niveau de raisonnement — fourni.

{identifiant du modèle chez Mistral: nom affiché}. Le modèle par défaut est LLM_CHATBOT (fichier .env) ; il est
ajouté à la liste s'il n'y est pas. Le modèle choisi est celui qu'utilise `llm.chat(...)` (sans `model=`) pendant le
traitement du message : votre code n'a rien à changer.
"""
import llm

MODELS = {
    "mistral-small-latest": "Mistral Small",
    "mistral-large-latest" : "Mistral Large",
    "ministral-14b-latest" : "Ministral Extra",
    "ministral-8b-latest" : "Ministral Medium",
    "ministral-3b-latest" : "Ministral Mini"
}


# Raisonnement « maison » : 4 niveaux identiques pour tous les modèles (curseur à côté de la zone de saisie).
#   - pour les modèles qui ont un raisonnement natif, le niveau est traduit en `reasoning_effort` de l'API Mistral
#     (valeurs acceptées testées sur l'API : Small = none/high, Large = low/medium/high ; les Ministral n'en ont pas) ;
#   - pour TOUS les modèles, le niveau fixe le nombre maximal d'étapes de l'agent (llm.agent_max_steps()).
LEVELS = ["low", "medium", "high", "xhigh"]
LEVEL_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra High"}
DEFAULT_LEVEL = "medium"
AGENT_STEPS = llm.AGENT_STEPS        # étapes max de l'agent : {"low": 3, "medium": 6, "high": 10, "xhigh": 16}
API_EFFORT = {   # niveau maison -> reasoning_effort envoyé à l'API
    "mistral-small-latest": {"low": "none", "medium": "none", "high": "high", "xhigh": "high"},
    "mistral-large-latest": {"low": "low", "medium": "medium", "high": "high", "xhigh": "high"},
}


def available() -> dict:
    models = dict(MODELS)
    models.setdefault(llm.CHATBOT_MODEL, llm.CHATBOT_MODEL)
    return {
        "default": llm.CHATBOT_MODEL,
        "reasoning": {"levels": LEVELS, "labels": LEVEL_LABELS, "default": DEFAULT_LEVEL, "agent_steps": AGENT_STEPS,
                      "budget": llm.REASONING_BUDGET},
        "models": [{"id": k, "label": v, "native_reasoning": API_EFFORT.get(k)} for k, v in models.items()],
    }


def is_allowed(model_id: str) -> bool:
    return any(m["id"] == model_id for m in available()["models"])


def resolve_reasoning(model_id: str, level: str | None) -> tuple[str, str | None]:
    """Niveau maison -> (niveau retenu, reasoning_effort à envoyer à l'API ou None). Lève ValueError si inconnu."""
    level = level or DEFAULT_LEVEL
    if level not in LEVELS:
        raise ValueError(f"niveau de raisonnement inconnu : {level!r} (attendu : {LEVELS})")
    return level, API_EFFORT.get(model_id, {}).get(level)
