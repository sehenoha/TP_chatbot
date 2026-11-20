"""Outils fournis pour exploiter les documents — vous n'avez normalement pas à les modifier.

Trois fonctions, à utiliser dans `ai.py` :

    extract_text(doc)                       -> str          # le texte d'un document
    chunk_text(text, chunk_size, overlap)   -> list[str]    # découpage d'un texte en morceaux (sans couper les phrases)
    load_chunks(documents, ...)             -> list[dict]    # les deux à la fois, pour tous les documents

Exemple :

    from tools import load_chunks

    chunks = load_chunks(documents)          # documents = le paramètre de generate_reply
    chunks[0]  # {"doc_id": 1, "source": "civil.txt", "chunk_id": 0, "text": "Article 1 ..."}
"""
import io
import re

TEXT_EXTENSIONS = (".txt", ".md", ".csv", ".json", ".html", ".xml", ".log", ".py", ".yaml", ".yml")


def _clean(text: str) -> str:
    """Normalise les retours à la ligne et supprime les caractères parasites."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)       # espaces en fin de ligne
    text = re.sub(r"\n{3,}", "\n\n", text)       # au plus une ligne vide d'affilée
    return text.strip()


def extract_text(doc: dict) -> str:
    """Renvoie le texte d'un document (dict reçu dans `documents`).

    Formats gérés : PDF, Word (.docx) et fichiers texte (.txt, .md, .csv, .json, ...).
    Lève `ValueError` si le format n'est pas pris en charge ou si le fichier est illisible.
    """
    name = doc["name"].lower()
    data: bytes = doc["data"]

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as e:
            raise ValueError(f"PDF illisible ({doc['name']}) : {e}") from e
        text = "\n\n".join(pages)
        if not text.strip():
            raise ValueError(f"Aucun texte dans {doc['name']} (PDF scanné ? il faudrait de l'OCR)")
        return _clean(text)

    if name.endswith(".docx"):
        from docx import Document
        try:
            paragraphs = [p.text for p in Document(io.BytesIO(data)).paragraphs]
        except Exception as e:
            raise ValueError(f"Fichier Word illisible ({doc['name']}) : {e}") from e
        return _clean("\n".join(paragraphs))

    # Texte brut : UTF-8 en priorité, sinon latin-1 pour les extensions texte connues
    try:
        return _clean(data.decode("utf-8"))
    except UnicodeDecodeError:
        if name.endswith(TEXT_EXTENSIONS):
            return _clean(data.decode("latin-1"))
        raise ValueError(f"Format non pris en charge : {doc['name']}")


# Abréviations après lesquelles un point ne termine pas une phrase
_ABBREVIATIONS = {"m", "mm", "mme", "mmes", "mlle", "dr", "pr", "me", "art", "arts", "al", "cf", "ex",
                  "no", "p", "pp", "vol", "éd", "ed", "env", "av", "apr", "s", "ss", "t", "v", "c", "n"}

# Frontière de phrase : ponctuation finale suivie d'une majuscule / d'un guillemet / d'un chiffre,
# ou retour à la ligne devant un titre Markdown, une puce, ou « Article ».
_BOUNDARY = re.compile(
    r"""(?P<s>[.!?…]["»”)\]]*)\s+(?=["«“(\[A-ZÀ-ÖØ-Þ0-9])"""
    r"""|\n(?=[ \t]*(?:\#{1,6}[ \t]|[-*•][ \t]|\d+[.)][ \t]|Article[ \t]))"""
    r"""|(?P<h>(?m:^)\#{1,6}[ \t][^\n]*)(?:\n|$)"""      # une ligne de titre Markdown forme une unité à elle seule
)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
MIN_FILL = 0.25   # un chunk n'est jamais raccourci pour respecter un paragraphe s'il serait rempli à moins de 25 %


def _is_abbreviation(text: str, dot_pos: int) -> bool:
    m = re.search(r"([^\W\d_]+)$", text[max(0, dot_pos - 12):dot_pos])
    return bool(m) and (len(m.group(1)) == 1 or m.group(1).lower() in _ABBREVIATIONS)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _hard_split(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Dernier recours pour une « phrase » plus longue que `size` : coupure à un retour à la ligne,
    à une virgule ou à un espace."""
    pieces = []
    while end - start > size:
        cut = -1
        for sep in ("\n", "; ", ", ", " "):
            cut = text.rfind(sep, start + size // 2, start + size)
            if cut != -1:
                cut += len(sep)
                break
        if cut == -1:
            cut = start + size
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


def _split_units(text: str, chunk_size: int) -> list[tuple[int, int, int]]:
    """Découpe le texte en « unités » (phrases) : liste de (début, fin, numéro de paragraphe)."""
    units = []
    paragraphs, pos = [], 0
    for m in _PARAGRAPH_BREAK.finditer(text):
        paragraphs.append((pos, m.start()))
        pos = m.end()
    paragraphs.append((pos, len(text)))

    def add(start, end, para):
        start, end = _trim(text, start, end)
        if start >= end:
            return
        for s, e in _hard_split(text, start, end, chunk_size) if end - start > chunk_size else [(start, end)]:
            s, e = _trim(text, s, e)
            if s < e:
                units.append((s, e, para))

    for para, (ps, pe) in enumerate(paragraphs):
        last = ps
        for m in _BOUNDARY.finditer(text, ps, pe):
            if m.group("h") is not None:
                add(last, m.end("h"), para)
            elif m.group("s") is not None:
                if text[m.start("s")] == "." and _is_abbreviation(text, m.start("s")):
                    continue
                add(last, m.end("s"), para)
            else:
                add(last, m.start(), para)
            last = m.end()
        add(last, pe, para)
    return units


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Découpe `text` en chunks d'au plus `chunk_size` caractères, **sans couper les phrases**.

    Voir `chunk_spans` pour le détail ; ici on renvoie directement le texte de chaque chunk.
    """
    return [text[a:b] for a, b in chunk_spans(text, chunk_size, overlap)]


_SEPARATOR = re.compile(r"(?m)^[ \t]*---[ \t]*$")
MAX_BLOCK = 6000   # un bloc délimité par des « --- » reste un seul chunk jusqu'à cette taille


def chunk_spans(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[tuple[int, int]]:
    """Comme `chunk_text`, mais renvoie les positions `(début, fin)` de chaque chunk dans `text`
    (utile pour retrouver ce qui entoure un chunk dans le document).

    Règles du découpage, **sans couper les phrases** :

    - le texte est d'abord découpé en phrases ; un chunk est fait de phrases **entières** ;
    - un **paragraphe** qui tient dans un chunk n'est jamais coupé en deux : si la place manque,
      il commence le chunk suivant ;
    - le chevauchement (`overlap`) est fait de phrases entières reprises de la fin du chunk
      précédent (au plus `overlap` caractères, éventuellement 0 si la dernière phrase est plus longue) ;
    - seule une phrase plus longue que `chunk_size` est coupée, à une virgule ou à un espace ;
    - si le texte contient des lignes `---`, chaque bloc entre deux séparateurs est **un seul chunk** (sans
      tenir compte de `chunk_size`), sauf s'il dépasse `MAX_BLOCK` caractères : il est alors découpé comme ci-dessus.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap doit être compris entre 0 et chunk_size - 1")

    if _SEPARATOR.search(text):
        spans, pos = [], 0
        for m in list(_SEPARATOR.finditer(text)) + [None]:
            end = m.start() if m else len(text)
            a, b = _trim(text, pos, end)
            if a < b:
                if b - a <= MAX_BLOCK:
                    spans.append((a, b))
                else:
                    spans += [(a + s, a + e) for s, e in chunk_spans(text[a:b], chunk_size, overlap)]
            pos = m.end() if m else end
        return spans

    offset = len(text) - len(text.lstrip())     # les positions renvoyées sont relatives au texte d'origine
    text = text.strip()
    units = _split_units(text, chunk_size)
    n = len(units)

    para_span = {}   # numéro de paragraphe -> (début, fin) du paragraphe
    for s, e, p in units:
        para_span[p] = (para_span[p][0] if p in para_span else s, e)

    def span(a: int, b: int) -> int:        # longueur du texte des unités a..b (inclus)
        return units[b][1] - units[a][0]

    chunks = []
    prev = None            # (premier, dernier+1) des unités du chunk précédent
    i = 0                  # première unité pas encore couverte
    while i < n:
        # 1. chevauchement : phrases entières de la fin du chunk précédent
        j = i
        if prev and overlap > 0:
            first_prev, end_prev = prev
            while j - 1 > first_prev and span(j - 1, end_prev - 1) <= overlap:
                j -= 1
        while j < i and span(j, i) > chunk_size:     # si ça ne laisse pas la place, on réduit le chevauchement
            j += 1
        # 2. on ajoute des phrases tant que ça tient
        k = i + 1
        while k < n and span(j, k) <= chunk_size:
            k += 1
        # 3. si on coupe au milieu d'un paragraphe qui tiendrait entier ailleurs, on s'arrête avant lui
        if k < n and units[k][2] == units[k - 1][2]:
            ps, pe = para_span[units[k][2]]
            if pe - ps <= chunk_size:
                b = next((x for x in range(k - 1, i, -1) if units[x][2] != units[x - 1][2]), None)
                if b is not None and span(j, b - 1) >= MIN_FILL * chunk_size:
                    k = b
        chunks.append((offset + units[j][0], offset + units[k - 1][1]))
        prev = (j, k)
        i = k
    return chunks


def load_chunks(documents: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Extrait et découpe tous les documents.

    Renvoie une liste de dicts : {"doc_id", "source" (nom du fichier), "chunk_id", "text"}.
    Un document illisible est ignoré (un message est affiché dans les logs du backend).
    """
    result = []
    for doc in documents:
        try:
            text = extract_text(doc)
        except ValueError as e:
            print(f"[tools] document ignoré : {e}", flush=True)
            continue
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            result.append({"doc_id": doc["id"], "source": doc["name"], "chunk_id": i, "text": chunk})
    return result
