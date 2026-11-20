"""Recettes de cuisine — fourni.

Quand un chunk est une recette au format de `exemples/recettes-marmiton.md` (titre `###`, puis Catégorie, Durée,
Personnes, Source, **Description** (facultative), **Ingrédients**, **Ustensiles**, **Préparation**), la contextualisation automatique fait deux choses :

    1. la **description** de la recette (type de plat, ingrédients principaux, technique...) est lue dans le
       bloc `**Description**` du fichier Markdown ; s'il n'y en a pas, un LLM la rédige. Que vectoriser pour retrouver une recette : la description,
       la recette entière, un mélange ? C'est à vous d'en décider (et de comparer) ;
    2. la recette est rangée **sous forme structurée** dans la table PostgreSQL `recipes` (une ligne par recette,
       même (document_id, chunk_index) que la ligne de `chunk_embeddings`).

Table `recipes` : document_id, chunk_index, title, category, duration_minutes, duration_text, servings,
source_url, description, ingredients (JSONB), utensils (JSONB), utensils_norm (TEXT[]), steps (JSONB).
`utensils` est la liste brute du site (« formidable plat à four », « pôele », prix...) ; `utensils_norm` est la
même liste **normalisée** (noms canoniques de `UTENSILS` : « four », « poêle », « mixeur »...), faite pour filtrer en SQL :
    SELECT title FROM recipes WHERE utensils_norm <@ ARRAY['four','saladier','fouet']      -- tout ce qu'il faut est dispo
Un ingrédient est un dict {"raw": "400 g de faisselle", "quantity": 400.0, "unit": "g", "name": "faisselle"}
(`quantity` vaut None s'il n'y en a pas : « poivre et sel »).

Fonctions :
    get(document_id, chunk_index, servings=None) -> dict | None
        la recette ; avec `servings`, les quantités sont adaptées à ce nombre de personnes (comme sur Marmiton)
    get_many([(document_id, chunk_index), ...], servings=None) -> list[dict]
    scale(recipe, servings) -> dict          # même chose sur une recette déjà chargée
    normalize_utensils(liste_brute) -> list[str]   # noms canoniques (voir UTENSILS)
    parse_recipe(chunk_text) -> dict | None  # lit une recette au format Markdown ci-dessus
"""
import re
from functools import lru_cache
from fractions import Fraction

import llm

MODEL = llm.CONTEXT_MODEL      # LLM_CONTEXT dans .env

# ------------------------------------------------------------------ lecture d'une recette Markdown

_SECTIONS = {"description": "description", "ingrédients": "ingredients", "ustensiles": "utensils", "préparation": "steps"}
_BULLET = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+)")


def _duration_minutes(text: str) -> int | None:
    h = re.search(r"(\d+)\s*h", text)
    m = re.search(r"(\d+)\s*(?:min|mn)", text) or (re.search(r"h\s*(\d+)", text))
    if not h and not m:
        return None
    return (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)


def parse_recipe(chunk: str) -> dict | None:
    """Lit une recette Markdown. Renvoie None si le texte n'en est pas une."""
    titles = list(re.finditer(r"^###[ \t]+(.+?)[ \t]*$", chunk, re.M))
    if not titles or "**Ingrédients**" not in chunk or "**Préparation**" not in chunk:
        return None
    body = chunk[titles[-1].end():]
    recipe = {"title": titles[-1].group(1), "category": None, "duration_text": None, "duration_minutes": None,
              "servings": None, "source_url": None, "description": None, "ingredients": [], "utensils": [], "steps": []}
    section = None
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        head = re.fullmatch(r"\*\*(.+?)\*\*", line)
        if head:
            section = _SECTIONS.get(head.group(1).strip().lower())
            continue
        if section is None:
            m = re.match(r"-\s*([^:]+?)\s*:\s*(.+)", line)
            if not m:
                continue
            key, value = m.group(1).lower(), m.group(2).strip()
            if key.startswith("catég"):
                recipe["category"] = value
            elif key.startswith("dur"):
                recipe["duration_text"], recipe["duration_minutes"] = value, _duration_minutes(value)
            elif key.startswith("personn"):
                n = re.match(r"\d+", value)
                recipe["servings"] = int(n.group()) if n else None
            elif key.startswith("source"):
                recipe["source_url"] = value
        else:
            item = _BULLET.sub("", line)
            if section == "description":
                recipe["description"] = f"{recipe['description']} {line}" if recipe["description"] else line
                continue
            recipe[section].append(parse_ingredient(item) if section == "ingredients" else item)
    return recipe if recipe["ingredients"] and recipe["steps"] else None


# ------------------------------------------------------------------ ingrédients et adaptation des quantités

_NUM = r"\d+(?:[.,]\d+)?(?:\s+\d+/\d+)?|\d+/\d+"
_QTY = re.compile(rf"^(?P<q>{_NUM})(?:\s*(?:à|-)\s*(?P<q2>{_NUM}))?(?=\s|$)")
_UNITS = ("kg", "mg", "g", "dl", "cl", "ml", "l", "cuillères? à soupe", "cuillères? à café", "cuillères?", "c\\. à (?:s|c)\\.?",
          "pincées?", "verres?", "tasses?", "sachets?", "boîtes?", "pots?", "gousses?", "tranches?", "bottes?",
          "bouquets?", "feuilles?", "brins?", "branches?", "noix", "morceaux?", "poignées?", "carrés?")
_UNIT = re.compile(rf"^(?P<u>{'|'.join(_UNITS)})(?=\s|$)", re.I)
_PAREN = re.compile(rf"\((?P<q>{_NUM})\s*(?P<u>kg|g|cl|ml|dl|l)\b")


def _to_float(s: str) -> float:
    s = s.strip().replace(",", ".")
    if " " in s:
        whole, frac = s.split()
        return float(whole) + float(Fraction(frac))
    return float(Fraction(s)) if "/" in s else float(s)


def _fmt(v: float) -> str:
    if v >= 10:
        return str(round(v))
    quarter = round(v * 4) / 4
    if abs(v - quarter) < 0.04:
        whole, frac = int(quarter), quarter - int(quarter)
        label = {0.25: "1/4", 0.5: "1/2", 0.75: "3/4"}.get(frac)
        return f"{whole} {label}" if whole and label else (label or str(whole))
    return f"{v:.1f}".rstrip("0").rstrip(".").replace(".", ",")


_VULGAR = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅛": "1/8"}


def parse_ingredient(raw: str) -> dict:
    """« 400 g de faisselle » -> {"raw", "quantity": 400.0, "unit": "g", "name": "faisselle"}."""
    raw = re.sub(r"(\d)\s*([½⅓⅔¼¾⅛])", lambda m: f"{m.group(1)} {_VULGAR[m.group(2)]}", raw)   # « 1½ » -> « 1 1/2 »
    raw = re.sub(r"[½⅓⅔¼¾⅛]", lambda m: _VULGAR[m.group()], raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    quantity, unit, rest = None, None, raw
    m = _QTY.match(raw)
    if m:
        quantity, rest = _to_float(m.group("q")), raw[m.end():].strip()
        u = _UNIT.match(rest)
        if u:
            unit, rest = u.group("u").lower(), rest[u.end():].strip()
    name = re.sub(r"^(?:de |d'|du |des )\s*", "", rest, flags=re.I).strip()
    return {"raw": raw, "quantity": quantity, "unit": unit, "name": name}


def scale_ingredient(ing: dict, ratio: float) -> dict:
    """Adapte un ingrédient (quantité, texte et mesures entre parenthèses comme « (200 g) »)."""
    if ing["quantity"] is None:
        return dict(ing)
    raw = ing["raw"]
    m = _QTY.match(raw)
    new_qty = _fmt(_to_float(m.group("q")) * ratio)
    if m.group("q2"):
        new_qty += f" à {_fmt(_to_float(m.group('q2')) * ratio)}"
    raw = new_qty + raw[m.end():]
    raw = _PAREN.sub(lambda p: f"({_fmt(_to_float(p.group('q')) * ratio)} {p.group('u')}", raw)
    return {**ing, "raw": raw, "quantity": round(ing["quantity"] * ratio, 3)}


def scale(recipe: dict, servings: int) -> dict:
    """Recette adaptée à `servings` personnes (règle de trois sur toutes les quantités)."""
    if not recipe.get("servings") or servings <= 0:
        return recipe
    ratio = servings / recipe["servings"]
    return {**recipe, "servings": servings, "scaled_from": recipe["servings"],
            "ingredients": [scale_ingredient(i, ratio) for i in recipe["ingredients"]]}


# ------------------------------------------------------------------ normalisation des ustensiles

# (nom canonique, expression régulière sur le texte brut en minuscules). Le texte brut contient du bruit :
# marques, prix, fautes (« pôele »), plusieurs ustensiles collés... : on cherche donc les mots-clés.
_UTENSIL_RULES = [
    ("four", r"(?<!à )(?<!a )\bfour\b(?! ?(?:à|de) )"), ("plat à four", r"plat à (?:four|gratin|tarte)|plaque à four|plaque de four"),
    ("plaque de cuisson", r"induction|plaque de cuisson|plaque vitro|table de cuisson|cuisinière"),
    ("micro-ondes", r"micro-?ondes"), ("barbecue", r"barbecue|plancha"),
    ("casserole", r"casserole"), ("poêle", r"p[oô]e?le|poêle|sauteuse"), ("sauteuse", r"sauteuse"), ("wok", r"\bwok\b"),
    ("cocotte", r"cocotte(?!-minute)"), ("cocotte-minute", r"cocotte-minute"), ("mijoteuse", r"mijoteuse|slow ?cooker"),
    ("friteuse", r"friteuse"), ("robot cuiseur", r"thermomix|monsieur cuisine|cookeo|multicuiseur|blender chauffant|companion"),
    ("mixeur plongeant", r"mixeur plongeant"), ("mixeur", r"mixeur(?! plongeant)|mixer\b"), ("blender", r"blender"),
    ("batteur électrique", r"batteur"), ("robot pâtissier", r"robot (?:p[aâ]tissier|multifonction)|kitchenaid"),
    ("sorbetière", r"sorbeti[eè]re|turbine à glace"), ("appareil spécial", r"appareil à|machine à"),
    ("balance", r"balance"), ("couteau", r"couteau"), ("planche à découper", r"planche"), ("économe", r"économe|eplucheur|éplucheur"),
    ("dénoyauteur", r"dénoyauteur|denoyauteur"), ("râpe", r"râpe"), ("presse-agrumes", r"presse[- ]agrume"),
    ("tamis", r"tamis"), ("passoire", r"passoire|chinois"), ("écumoire", r"écumoire"), ("louche", r"louche"),
    ("spatule", r"spatu|maryse"), ("ciseaux", r"ciseaux"), ("presse-ail", r"presse-ail"), ("hachoir", r"hachoir"),
    ("petite cuillère", r"petite cuillère"), ("cuillère en bois", r"cuillère en bois|cuillere en bois"), ("fouet", r"fouet"),
    ("fourchette", r"fourchette"), ("pinceau", r"pinceau"), ("rouleau à pâtisserie", r"rouleau"),
    ("saladier", r"saladier"), ("bol", r"\bbol\b"), ("assiette", r"assiette"), ("verre", r"\bverre\b|verrine|\btasse\b"),
    ("ramequin", r"ramequin"), ("moule", r"moule|cercle à p[aâ]tisserie|emporte-pièce"),
    ("grille", r"grille"), ("papier sulfurisé", r"papier sulfuris"), ("papier aluminium", r"aluminium"),
    ("film alimentaire", r"film alimentaire"), ("couvercle", r"couvercle"), ("entonnoir", r"entonnoir"),
    ("poche à douille", r"douille"), ("réfrigérateur", r"réfrigérateur|frigo"), ("congélateur", r"congélateur|freezer"),
]
_UTENSIL_RULES = [(name, re.compile(rx, re.I)) for name, rx in _UTENSIL_RULES]
UTENSILS = sorted({name for name, _ in _UTENSIL_RULES})      # le vocabulaire normalisé


def normalize_utensils(raw_list: list[str]) -> list[str]:
    """Liste brute (« 1 formidable plat à four », « pôele », « pinceau 6,72€ ») -> noms canoniques triés, sans doublon."""
    found = set()
    for raw in raw_list:
        text = raw.lower()
        for name, rx in _UTENSIL_RULES:
            if rx.search(text):
                found.add(name)
    return sorted(found)


# ------------------------------------------------------------------ description (si elle manque dans le .md)

SYSTEM_PROMPT = """Tu aides un moteur de recherche de recettes de cuisine. On te donne une recette ; rédige sa \
DESCRIPTION en français : 2 ou 3 phrases (70 mots au maximum) qui disent quel type de plat c'est, ses ingrédients \
principaux, sa technique de cuisson ou de préparation et, si c'est utile, son temps et son côté pratique (rapide, \
sans cuisson, au four, au Cookeo...). Cette description sera transformée en vecteur : choisis les mots qu'un \
utilisateur emploierait pour chercher cette recette.

Règles :
- N'écris que ce qui ressort de la recette ; n'invente rien (régime, origine, goût) qui n'y figure pas.
- Reste factuel : ingrédients, technique, cuisson, durée. Ne parle ni de goût, ni de texture, ni de température de service \
(chaud, froid, tiède) sauf si c'est écrit dans la recette.
- Pas de quantités, pas d'URL, pas de liste ; ne recopie pas le titre tel quel en début de phrase.
- Réponds uniquement avec la description, sans mise en forme."""


def recipe_text(recipe: dict) -> str:
    """La recette sous forme de texte, pour le prompt du LLM."""
    lines = [f"Titre : {recipe['title']}", f"Catégorie : {recipe['category']}",
             f"Durée : {recipe['duration_text']}", f"Personnes : {recipe['servings']}", "Ingrédients :"]
    lines += [f"- {i['raw']}" for i in recipe["ingredients"]]
    if recipe["utensils"]:
        lines += ["Ustensiles :"] + [f"- {u}" for u in recipe["utensils"]]
    lines += ["Préparation :"] + [f"{n}. {s}" for n, s in enumerate(recipe["steps"], 1)]
    return "\n".join(lines)


def describe(recipe: dict, model: str = MODEL) -> dict:
    """Demande au LLM la description de la recette. Renvoie {"description", "model"} (lève llm.LLMError)."""
    result = llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": recipe_text(recipe)}], model=model, max_tokens=200)
    text = re.sub(r"[*`]+", "", result["content"]).strip().strip('"«»').strip()
    return {"description": text, "model": result["model"]}


# ------------------------------------------------------------------ stockage (table `recipes`)

def save(doc_id: int, chunk_index: int, recipe: dict, description: str, model: str) -> None:
    import db
    from psycopg.types.json import Jsonb
    with db.connect() as conn:
        conn.execute(
            """INSERT INTO recipes (document_id, chunk_index, title, category, duration_minutes, duration_text,
                                    servings, source_url, description, model, ingredients, utensils, utensils_norm, steps)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                 title = EXCLUDED.title, category = EXCLUDED.category, duration_minutes = EXCLUDED.duration_minutes,
                 duration_text = EXCLUDED.duration_text, servings = EXCLUDED.servings, source_url = EXCLUDED.source_url,
                 description = EXCLUDED.description, model = EXCLUDED.model, ingredients = EXCLUDED.ingredients,
                 utensils = EXCLUDED.utensils, utensils_norm = EXCLUDED.utensils_norm, steps = EXCLUDED.steps, created_at = now()""",
            (doc_id, chunk_index, recipe["title"], recipe["category"], recipe["duration_minutes"],
             recipe["duration_text"], recipe["servings"], recipe["source_url"], description, model,
             Jsonb(recipe["ingredients"]), Jsonb(recipe["utensils"]), normalize_utensils(recipe["utensils"]),
             Jsonb(recipe["steps"])),
        )


def descriptions(doc_id: int) -> dict[int, str]:
    """{chunk_index: description} des recettes d'un document."""
    import db
    with db.connect() as conn:
        rows = conn.execute("SELECT chunk_index, description FROM recipes WHERE document_id = %s", (doc_id,)).fetchall()
    return {r["chunk_index"]: r["description"] for r in rows}


def _finish(row: dict | None, servings: int | None) -> dict | None:
    if row is None:
        return None
    row = {k: v for k, v in row.items() if k != "created_at"}
    return scale(row, servings) if servings else row


def get(doc_id: int, chunk_index: int, servings: int | None = None) -> dict | None:
    import db
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM recipes WHERE document_id = %s AND chunk_index = %s",
                           (doc_id, chunk_index)).fetchone()
    return _finish(row, servings)


def _norm(text: str) -> str:
    import unicodedata
    text = text.lower().replace("œ", "oe").replace("æ", "ae")          # « Bœuf » = « Boeuf »
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"[^a-z0-9]+", " ", text).strip() + " "


_VARIANTS = {"a": "aàâä", "e": "eéèêë", "i": "iîï", "o": "oôö", "u": "uùûü", "c": "cç", "y": "yÿ", "n": "nñ"}
_QUOTES = "\"'«»“”„‘’"


@lru_cache(maxsize=20000)
def _title_regex(title: str) -> re.Pattern:
    """Motif qui retrouve un titre sans tenir compte de la casse, des accents, des ligatures (« Bœuf » = « Boeuf »),
    des guillemets ni des espaces."""
    import unicodedata
    t = title.lower().replace("œ", "oe").replace("æ", "ae")
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    out, i = [], 0
    while i < len(t):
        c = t[i]
        if c in "oa" and t[i + 1:i + 2] == "e":
            out.append("(?:o[eéèêë]|œ)" if c == "o" else "(?:a[eéèêë]|æ)")
            i += 2
            continue
        if c in _VARIANTS:
            out.append(f"[{_VARIANTS[c]}]")
        elif c.isspace():
            out.append(r"\s+")
        elif c in _QUOTES:
            out.append(f"\\s*[{re.escape(_QUOTES)}]?\\s*")
        elif c.isalnum():
            out.append(re.escape(c))
        else:
            out.append(re.escape(c) + "?")          # ponctuation facultative
        i += 1
    return re.compile(r"(?<!\w)" + "".join(out) + r"(?!\w)", re.I)


def is_title_mention(text: str, start: int, end: int, title: str) -> bool:
    """La recette est-elle vraiment CITÉE à cet endroit ? Oui si le titre est mis en avant : début de ligne, après une
    ponctuation ou une mise en forme (**, «, :, -...), écrit avec une majuscule, ou s'il est long (5 mots ou plus).
    Évite « 1 c. à soupe de lait » -> recette « Soupe de lait »."""
    before = text[:start].rstrip(" \t")
    if not before or not before[-1].isalnum():
        return True
    return text[start].isupper() or len(title.split()) >= 5


def find_in_text(text: str, limit: int = 6) -> list[dict]:
    """Recettes CITÉES dans `text` (la réponse du chatbot) par leur titre, dans l'ordre d'apparition.
    Sert à afficher une bulle (photo, fiche complète) à côté de chaque recette citée."""
    import db
    if not text:
        return []
    with db.connect() as conn:
        rows = conn.execute("SELECT document_id, chunk_index, title, category, duration_text, servings FROM recipes").fetchall()
    haystack = _norm(text)
    found = []
    for r in rows:
        if len(_norm(r["title"])) < 8 or _norm(r["title"]) not in haystack:     # filtre rapide
            continue
        for m in _title_regex(r["title"]).finditer(text):
            if is_title_mention(text, m.start(), m.end(), r["title"]):
                found.append((m.start(), m.end(), r))
                break
    found.sort(key=lambda f: (f[0], -(f[1] - f[0])))
    kept, seen = [], set()
    for start, end, r in found:          # un titre contenu dans un titre plus long déjà retenu est ignoré
        if any(s <= start and end <= e for s, e, _ in kept) or r["title"].lower() in seen:
            continue
        kept.append((start, end, r))
        seen.add(r["title"].lower())
    return [r for _, _, r in kept][:limit]


_NUMBERS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9,
            "dix": 10, "onze": 11, "douze": 12, "quinze": 15, "vingt": 20}
_N = r"(\d{1,3}|" + "|".join(_NUMBERS) + r")"
_PEOPLE = re.compile(_N + r"\s+(?:personnes?|pers\b|convives?|couverts?|parts?)", re.I)
_GUESTS = re.compile(_N + r"\s+(?:amis?|amies?|invités?|invitées?|copains?|copines?|collègues?)", re.I)


def _number(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBERS[word.lower()]


def guess_servings(reply: str, user_messages: list[str]) -> int | None:
    """Nombre de personnes de la conversation, pour ouvrir les fiches avec les bonnes quantités.

    D'abord dans la réponse (« pour 10 personnes »), sinon dans les messages de l'utilisateur, du plus récent au
    plus ancien (« pour 4 personnes », ou « 9 amis » -> 10 : les amis + l'utilisateur)."""
    if m := _PEOPLE.search(reply or ""):
        return _number(m.group(1))
    for text in reversed(user_messages):
        if m := _PEOPLE.search(text):
            return _number(m.group(1))
        if m := _GUESTS.search(text):
            return _number(m.group(1)) + 1
    return None


def image_url(doc_id: int, chunk_index: int) -> str | None:
    """URL de la photo de la recette, lue (une seule fois, puis mise en cache en base) dans la page Marmiton."""
    import json
    import urllib.request
    import db
    with db.connect() as conn:
        row = conn.execute("SELECT source_url, image_url FROM recipes WHERE document_id = %s AND chunk_index = %s",
                           (doc_id, chunk_index)).fetchone()
    if row is None:
        return None
    if row["image_url"] is not None:
        return row["image_url"] or None              # '' : déjà cherchée, pas de photo
    url = ""
    try:
        req = urllib.request.Request(row["source_url"], headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", "replace")
        for block in re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', html, re.S):
            try:
                data = json.loads(block)
            except ValueError:
                continue
            for item in data if isinstance(data, list) else data.get("@graph", [data]):
                if isinstance(item, dict) and item.get("@type") == "Recipe":
                    images = item.get("image") or []
                    images = [images] if isinstance(images, str) else images
                    images = [i for i in images if isinstance(i, str) and "default-recipe-picture" not in i]
                    url = next((i for i in images if "w1024h768" in i and i.endswith(".jpg")), images[0] if images else "")
    except Exception as e:
        print(f"[recipes] photo introuvable pour {row['source_url']} : {e}", flush=True)
        return None                                  # erreur réseau : on réessaiera plus tard
    with db.connect() as conn:
        conn.execute("UPDATE recipes SET image_url = %s WHERE document_id = %s AND chunk_index = %s", (url, doc_id, chunk_index))
    return url or None


def get_many(keys: list[tuple[int, int]], servings: int | None = None) -> list[dict]:
    """Plusieurs recettes d'un coup (dans l'ordre demandé) : keys = [(document_id, chunk_index), ...]."""
    return [r for r in (get(d, c, servings) for d, c in keys) if r]
