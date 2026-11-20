"""Petit client pour l'API Mistral — fourni, utilisable aussi dans `ai.py`.

    from llm import chat

    r = chat([{"role": "user", "content": "Bonjour !"}])
    r["content"]     # le texte de la réponse
    r["usage"]       # {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}

Modèles (fichier `.env`) : LLM_CHATBOT (défaut de chat(), mistral-small-latest) et LLM_CONTEXT
(contextualisation et descriptions, ministral-14b-latest).

La clé d'API est lue dans la variable d'environnement MISTRAL_API_KEY (fichier `.env`).
Aucune bibliothèque supplémentaire n'est nécessaire (c'est un simple appel HTTP).
"""
import contextvars
import http.client
import json
import os
import threading
import time
from collections import deque
import urllib.error
import urllib.request

API_URL = "https://api.mistral.ai/v1/chat/completions"
CONTEXT_MODEL = os.environ.get("LLM_CONTEXT", "") or "ministral-14b-latest"   # contextualisation des chunks, descriptions de recettes
CHATBOT_MODEL = os.environ.get("LLM_CHATBOT", "") or "mistral-small-latest"   # le chatbot (ai.py)
DEFAULT_MODEL = CHATBOT_MODEL                                                # modèle par défaut du chatbot

# Modèle choisi dans l'interface pour le message en cours (menu à côté de la zone de saisie) ; sinon LLM_CHATBOT.
_selected = contextvars.ContextVar("selected_model", default=None)
_level = contextvars.ContextVar("reasoning_level", default="medium")   # niveau maison choisi (curseur) : low...xhigh
_effort = contextvars.ContextVar("reasoning_effort", default=None)     # reasoning_effort envoyé à l'API (modèles natifs)
_thoughts = contextvars.ContextVar("thoughts", default=None)          # réflexions du modèle pendant le message
REASONING_BUDGET = {"low": 1000, "medium": 2000, "high": 4000, "xhigh": 8000}   # jetons de réflexion max selon le niveau
AGENT_STEPS = {"low": 3, "medium": 6, "high": 10, "xhigh": 16}   # étapes max de l'agent selon le niveau du curseur


def current_model() -> str:
    """Le modèle du chatbot pour le message en cours : celui choisi dans l'interface, sinon LLM_CHATBOT."""
    return _selected.get() or CHATBOT_MODEL


def select_model(model: str | None, level: str = "medium", api_effort: str | None = None):
    """Fixe le modèle et le niveau de raisonnement du message en cours (utilisé par main.py), et commence à
    collecter les réflexions du modèle. Renvoie de quoi revenir en arrière avec `unselect`."""
    return _selected.set(model or None), _level.set(level), _effort.set(api_effort), _thoughts.set([])


def unselect(tokens) -> None:
    for var, token in zip((_selected, _level, _effort, _thoughts), tokens):
        var.reset(token)


def current_level() -> str:
    """Niveau de raisonnement choisi avec le curseur : "low", "medium", "high" ou "xhigh"."""
    return _level.get()


def current_effort() -> str | None:
    """`reasoning_effort` envoyé à l'API pour ce niveau et ce modèle (None si le modèle n'a pas de raisonnement natif)."""
    return _effort.get()


def reasoning_budget() -> int:
    """Jetons de réflexion autorisés pour le niveau choisi (ajoutés à max_tokens quand le modèle raisonne)."""
    return REASONING_BUDGET.get(current_level(), 2000)


def agent_max_steps() -> int:
    """Nombre maximal d'étapes (appels d'outils) de l'agent pour le niveau choisi : à utiliser dans votre boucle."""
    return AGENT_STEPS.get(current_level(), 6)


def record_thinking(text: str) -> None:
    """Garde la réflexion du modèle pour l'afficher dans l'interface (bloc « Réflexion »)."""
    if text and (thoughts := _thoughts.get()) is not None:
        thoughts.append(text)


def thoughts() -> list[str]:
    return list(_thoughts.get() or [])


def split_content(content) -> tuple[str, str]:
    """Contenu d'une réponse Mistral -> (texte de la réponse, réflexion). Avec le raisonnement, `content` est une
    liste de morceaux : {"type": "thinking", "thinking": [{"type": "text", "text": ...}]} puis {"type": "text", ...}."""
    if not isinstance(content, list):
        return (content or ""), ""
    text, thinking = [], []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "thinking":
            inner = part.get("thinking") or []
            thinking.append("".join(t.get("text", "") for t in inner if isinstance(t, dict)) if isinstance(inner, list) else str(inner))
        else:
            text.append(part.get("text", ""))
    return "".join(text), "\n".join(thinking)


# ---- Limiteur de débit : jamais plus de N requêtes par minute (fenêtre glissante) -------------------------
# Mistral limite chaque clé (ex. 500 requêtes/minute). Au-delà il répond 429. Ce limiteur, partagé par tous
# les threads du backend, fait attendre les appels qui dépasseraient le quota ; il compte aussi les réessais.
SAFETY = 0.95    # on vise 95 % de la limite, pour garder une marge


class RateLimiter:
    def __init__(self, per_minute: int):
        self.limit = max(1, int(per_minute))
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Bloque jusqu'à ce qu'une requête soit autorisée, puis l'enregistre."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= 60:
                    self._stamps.popleft()
                if len(self._stamps) < self.limit:
                    self._stamps.append(now)
                    return
                wait = 60 - (now - self._stamps[0])
            time.sleep(max(wait, 0.05))

    def set_limit(self, per_minute: int) -> None:
        with self._lock:
            self.limit = max(1, int(per_minute))


_CONFIGURED = int(os.environ.get("LLM_REQUESTS_PER_MINUTE", "500"))     # limite connue de la clé (modifiable dans .env)
limiter = RateLimiter(int(_CONFIGURED * SAFETY))


def _learn_limit(headers) -> None:
    """Mistral annonce la limite de la clé dans `x-ratelimit-limit-req-minute` : on s'y adapte automatiquement."""
    try:
        announced = int(headers.get("x-ratelimit-limit-req-minute"))
    except (TypeError, ValueError):
        return
    limiter.set_limit(min(int(_CONFIGURED * SAFETY), int(announced * SAFETY)))


class LLMError(RuntimeError):
    """Erreur lors de l'appel au modèle (quota dépassé, réseau...)."""


class AuthError(LLMError):
    """Clé d'API absente ou refusée : inutile de réessayer."""


def api_key_set() -> bool:
    return bool(os.environ.get("MISTRAL_API_KEY", "").strip())


def _send(payload: dict, timeout: float, retries: int) -> dict:
    """POST vers l'API Mistral avec limite de débit et réessais ; renvoie la réponse JSON."""
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        raise AuthError("MISTRAL_API_KEY est vide : renseignez-la dans le fichier .env, "
                       "puis relancez `docker compose up`.")
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    for attempt in range(retries + 1):
        limiter.acquire()                    # attend si on a déjà atteint la limite de requêtes par minute
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                _learn_limit(response.headers)
                return json.load(response)
        except urllib.error.HTTPError as e:
            _learn_limit(e.headers)
            detail = e.read().decode("utf-8", errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = float(e.headers.get("Retry-After") or 2 ** attempt)   # quota ou panne passagère : on réessaie
                time.sleep(min(wait, 20))
                continue
            if e.code in (401, 403):
                raise AuthError("Clé d'API Mistral refusée (401/403) : vérifiez MISTRAL_API_KEY dans le fichier .env.") from None
            raise LLMError(f"Mistral a répondu {e.code} : {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:   # coupure réseau
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"Impossible de joindre l'API Mistral : {e}") from None
    raise LLMError("trop de tentatives")


def chat(messages: list[dict], model: str | None = None, max_tokens: int = 300,
         temperature: float = 0.0, timeout: float = 90, retries: int = 6) -> dict:
    """Envoie une conversation au modèle et renvoie {"content", "thinking", "usage", "model"}.

    Sans `model`, c'est le modèle choisi dans l'interface (sinon LLM_CHATBOT), avec le niveau de raisonnement du
    curseur : voir current_model() et current_effort(). `thinking` est la réflexion du modèle ("" s'il n'a pas raisonné).

    Budget de réflexion : l'API n'a pas de « max_reasoning_tokens » ; la réflexion compte dans `max_tokens`. On ajoute
    donc reasoning_budget() jetons (1000 à 8000 selon le niveau). Si la réflexion épuise tout le budget (réponse vide),
    on redemande une fois sans réflexion."""
    effort = None
    if model is None:
        model, effort = current_model(), current_effort()
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if effort:
        payload["reasoning_effort"] = effort
        if effort != "none":
            payload["max_tokens"] = max_tokens + reasoning_budget()

    data = _send(payload, timeout, retries)
    content, thinking, finish = _parse(data)
    if not content.strip() and finish == "length" and effort not in (None, "none"):
        print(f"[llm] budget de réflexion dépassé ({reasoning_budget()} jetons) : nouvelle demande sans réflexion", flush=True)
        try:
            data = _send({**payload, "reasoning_effort": "none", "max_tokens": max_tokens}, timeout, retries)
        except LLMError:                    # modèle qui raisonne toujours (ex. Large) : pas de « none »
            raise LLMError("La réflexion du modèle a dépassé son budget sans répondre : "
                           "choisissez un niveau de raisonnement plus élevé.") from None
        content, thinking2, finish = _parse(data)
        thinking = (thinking + "\n\n[budget de réflexion dépassé : réponse sans réflexion]").strip()
    record_thinking(thinking)
    return {"content": content.strip(), "thinking": thinking, "usage": data.get("usage", {}), "model": data.get("model", model)}


def _parse(data: dict) -> tuple[str, str, str | None]:
    """Réponse JSON -> (texte, réflexion, finish_reason)."""
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"Réponse inattendue de Mistral : {str(data)[:200]}") from None
    text, thinking = split_content(content)      # avec le raisonnement : liste de morceaux « thinking » + « text »
    return text, thinking, choice.get("finish_reason")
