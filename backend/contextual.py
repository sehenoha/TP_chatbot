"""Contextual retrieval — fourni.

Problème : une fois découpé, un chunk perd son contexte.
    « Le droit de rétention se perd par le dessaisissement volontaire. »
→ de quel droit parle-t-on ? dans quel article ? Une recherche risque de ne pas le retrouver.

Solution (Anthropic, 2024) : demander à un LLM de rédiger, pour CHAQUE chunk, une ou deux phrases qui
le replacent dans le document, et de les coller devant le chunk **avant** de l'indexer
(embeddings, recherche par mots-clés...) :

    contexte généré par le LLM  +  chunk d'origine   →   c'est ce texte qu'on indexe

Ce qu'on donne au LLM pour situer un chunk (pas le document entier, ce serait trop long et trop cher) :
    - le chemin de titres du chunk (ex. « Code civil > Livre IV > Article 2286 ») ;
    - le chunk, avec les NEIGHBORS chunks qui le précèdent et les NEIGHBORS chunks qui le suivent ;
    - sans jamais sortir de la **section** qui contient le chunk (le « chapitre », titre de niveau
      SECTION_LEVEL ou supérieur) : au plus, on donne donc la section entière.

Fonctions :
    situate_chunk(text, spans, index, name) -> dict     # appelle le LLM pour UN chunk
    contextualize_text(context, chunk)      -> str      # le texte « enrichi » à indexer
"""
import bisect
import os
import re
from functools import lru_cache

import llm

MODEL = llm.CONTEXT_MODEL                                                      # LLM_CONTEXT dans .env
NEIGHBORS = max(0, int(os.environ.get("CONTEXT_NEIGHBORS", "2")))              # chunks voisins de chaque côté
SECTION_LEVEL = max(1, int(os.environ.get("CONTEXT_SECTION_LEVEL", "3")))      # niveau de titre = « chapitre » (### par défaut)
MAX_WINDOW_CHARS = 24_000                                                      # garde-fou : extrait jamais plus long

# Message « system » : les consignes, identiques pour tous les chunks
SYSTEM_PROMPT = """Tu aides un moteur de recherche documentaire. On te donne un extrait d'un document (un passage et \
les passages qui l'entourent) puis le passage à situer. Rédige UNE SEULE phrase courte (40 mots au maximum) qui \
situe ce passage dans le document, afin d'améliorer la recherche de ce passage : indique où il se trouve (partie, \
article...) et de quoi il parle.

Règles :
- Décris uniquement ce que contient le passage ; ne parle pas de ce qui le précède ou le suit.
- La position indiquée (chemin de titres) fait foi : si tu cites un livre, un titre ou un article, ce doit \
être exactement celui de cette position, jamais un autre titre vu dans l'extrait.
- N'invente aucun titre, numéro ni information absents du document.
- Réponds uniquement avec cette phrase, sans mise en forme (pas de gras, pas de liste)."""

# Message « user » : le contenu à traiter, différent pour chaque chunk
USER_PROMPT = """<document>
{document}
</document>
Voici le passage à situer dans le document :
<chunk>
{chunk}
</chunk>"""

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)


@lru_cache(maxsize=4)
def _headings(text: str) -> tuple[tuple[int, int, int, str], ...]:
    """Tous les titres Markdown du texte : (début, fin, niveau, intitulé). Calculé une fois par texte."""
    return tuple((m.start(), m.end(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(text))


@lru_cache(maxsize=8)
def _sections(text: str, level: int) -> tuple[list[int], list[int], list[str]]:
    """Les titres de niveau <= `level` (les « chapitres »), sous forme de listes (débuts, fins, intitulés)."""
    secs = [h for h in _headings(text) if h[2] <= level]
    return [h[0] for h in secs], [h[1] for h in secs], [h[3] for h in secs]


def heading_path(text: str, pos: int) -> str:
    """Chemin de titres Markdown qui précède la position `pos` (ex. « Code civil > Livre Ier > Article 7 »)."""
    stack: dict[int, str] = {}
    for _, end, level, title in _headings(text):
        if end > pos:
            break
        stack = {k: v for k, v in stack.items() if k < level}
        stack[level] = title
    return " > ".join(stack[k] for k in sorted(stack))


def content_position(text: str, start: int, end: int) -> int:
    """Position à laquelle on mesure « où est le chunk ». Si le chunk commence par des titres
    (« ## Livre Ier / ### Titre Ier / #### Article 7 »), on les saute : ils font partie du chemin du chunk."""
    pos = start
    while pos < end and text.startswith("#", pos):
        nl = text.find("\n", pos)
        pos = end if nl == -1 else min(nl + 1, end)
    return pos


def section_bounds(text: str, pos: int, level: int = SECTION_LEVEL) -> tuple[int, int, str | None]:
    """Début, fin et intitulé de la section (« chapitre ») qui contient la position `pos`.
    Une section va d'un titre de niveau <= `level` jusqu'au titre de même rang suivant."""
    starts, ends, titles = _sections(text, level)
    k = bisect.bisect_right(ends, pos)                   # dernier titre de section terminé avant `pos`
    j = bisect.bisect_left(starts, pos)                  # premier titre de section qui commence à partir de `pos`
    return (starts[k - 1] if k else 0), (starts[j] if j < len(starts) else len(text)), (titles[k - 1] if k else None)


def neighborhood(text: str, spans: list[tuple[int, int]], index: int, neighbors: int = NEIGHBORS) -> tuple[str, dict]:
    """L'extrait donné au LLM : le chunk `index` et ses `neighbors` voisins de chaque côté, limité à la section
    qui contient le chunk. Renvoie (extrait, infos sur l'extrait)."""
    start, end = spans[index]
    first, last = max(0, index - neighbors), min(len(spans) - 1, index + neighbors)
    sec_start, sec_end, section = section_bounds(text, content_position(text, start, end))

    a = min(max(spans[first][0], sec_start), start)      # voisins, mais sans sortir de la section
    b = max(min(spans[last][1], sec_end), end)           # (le chunk lui-même est toujours inclus en entier)
    if b - a > MAX_WINDOW_CHARS:                         # garde-fou pour de très gros chunks / réglages
        margin = max(0, (MAX_WINDOW_CHARS - (end - start)) // 2)
        a, b = max(a, start - margin), min(b, end + margin)
        a = text.rfind("\n", 0, a) + 1 if a > 0 else 0
        nl = text.find("\n", b)
        b = len(text) if nl == -1 else nl

    before = sum(1 for k in range(first, index) if spans[k][0] >= a)
    after = sum(1 for k in range(index + 1, last + 1) if spans[k][1] <= b)
    info = {"chars": b - a, "before": before, "after": after, "section": section,
            "whole_section": a <= sec_start and b >= sec_end and bool(section)}
    return text[a:b], info


def build_prompt(text: str, spans: list[tuple[int, int]], index: int, name: str = "") -> tuple[str, str, dict]:
    """Renvoie (message utilisateur complet, version « lisible » pour l'affichage, infos sur l'extrait)."""
    start, end = spans[index]
    excerpt, info = neighborhood(text, spans, index)
    path = heading_path(text, content_position(text, start, end))        # ex. « Code civil > Livre IV > Article 2286 »

    intro = f"Titre du document : {name}\n" if name else ""
    if path:
        intro += f"Position du passage dans le document (titres qui le précèdent) : {path}\n"
    scope = f"{info['before']} passage(s) avant et {info['after']} après le passage à situer"
    scope += f", sans sortir de la section « {info['section']} »" if info["section"] else ""
    intro += f"(Seul un extrait est fourni : {scope}.)\n"

    chunk = text[start:end]
    full = USER_PROMPT.format(document=intro + "\n" + excerpt, chunk=chunk)
    shown = USER_PROMPT.format(
        document=intro + "\n" + f"[… extrait de {info['chars']:,} caractères …]".replace(",", " "),
        chunk=chunk,
    )
    return full, shown, info


def situate_chunk(text: str, spans: list[tuple[int, int]], index: int, name: str = "", model: str = MODEL) -> dict:
    """Demande au LLM le contexte du chunk n° `index` (`text[spans[index][0]:spans[index][1]]`).

    `spans` : positions des chunks, telles que renvoyées par `tools.chunk_spans(text, ...)`.
    Renvoie {"context", "system_prompt", "prompt" (message utilisateur, version lisible), "window" (infos sur
    l'extrait envoyé), "usage", "model"}. Lève `llm.LLMError` en cas de problème (clé absente, quota...).
    """
    full, shown, info = build_prompt(text, spans, index, name)
    result = llm.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full}],
        model=model, max_tokens=150,
    )
    context = re.sub(r"[*`]+", "", result["content"]).strip().strip('"«»').strip()   # pas de mise en forme
    return {"context": context, "system_prompt": SYSTEM_PROMPT, "prompt": shown, "window": info,
            "usage": result["usage"], "model": result["model"]}


def contextualize_text(context: str, chunk: str) -> str:
    """Texte à indexer : le contexte, puis le chunk."""
    return f"{context}\n\n{chunk}" if context else chunk
