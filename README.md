# TP Chatbot

Une interface (React), une API (FastAPI) et une base de données (PostgreSQL) vous sont **fournies**.
Votre travail : la **partie IA**.

```
Navigateur ──► frontend (React, :5180) ──/api──► backend (FastAPI, :8010) ──► ai.py  ◄── vous
                                                        │
                                          ┌─────────────┴─────────────┐
                                          ▼                           ▼
                              PostgreSQL + pgvector (:5433)    API Mistral (mistral-small-latest,
                          documents, contextes, vecteurs        ministral-14b-latest) : chatbot, contextualisation ; embeddings
                                                                et réponse (à vous)
```

## 0. Installer Docker

Tout le TP tourne dans Docker : le projet se lance de la même façon sur toutes les machines. Suivez le guide
**[INSTALL_DOCKER.md](INSTALL_DOCKER.md)** (aussi en PDF : `INSTALL_DOCKER.pdf`) pour macOS, Windows ou Linux, puis
vérifiez que `docker run --rm hello-world` affiche « Hello from Docker! ».

Si `docker` ne répond pas, c'est presque toujours que **Docker Desktop n'est pas démarré** (attendez qu'il soit stable).

## Configuration : le fichier `.env`

Les réglages sont dans un fichier `.env` à la racine du projet (modèle : `.env.example`) :

```bash
cp .env.example .env      # inutile si le fichier .env existe déjà
```

| Variable | Rôle | Valeur par défaut |
|---|---|---|
| `MISTRAL_API_KEY` | votre clé d'API Mistral, lisible dans `ai.py` avec `os.environ["MISTRAL_API_KEY"]` | *(vide : à renseigner)* |
| `FRONTEND_PORT` | port de l'interface sur votre machine | `5180` |
| `LLM_CHATBOT` | modèle du chatbot (`llm.chat` par défaut) | `mistral-small-latest` |
| `LLM_CONTEXT` | modèle de la contextualisation des chunks et des descriptions de recettes | `ministral-14b-latest` |
| `BACKEND_PORT` | port de l'API sur votre machine | `8010` |
| `DB_PORT` | port de PostgreSQL sur votre machine | `5433` |
| `AUTO_CONTEXTUALIZE` | contextualisation automatique des documents ajoutés (voir plus bas) ; `false` pour la désactiver | `true` |
| `AUTO_CONTEXT_MAX_CHUNKS` | garde-fou : un document de plus de chunks que cette valeur n'est pas contextualisé automatiquement | `5000` |
| `AUTO_CONTEXT_WORKERS` | nombre d'appels au LLM en parallèle | `64` |
| `LLM_REQUESTS_PER_MINUTE` | limite de requêtes par minute de votre clé Mistral. Le backend n'envoie **jamais plus de 95 %** de cette limite (fenêtre glissante de 60 s, réessais compris) et s'adapte à la limite annoncée par Mistral. Si votre clé est plus limitée (ex. 1 requête/s), mettez `60` | `500` |

⚠️ Ne partagez jamais votre clé d'API et ne committez pas le fichier `.env` (il est déjà dans `.gitignore`).
Après modification du `.env`, relancez `docker compose up`.

## Lancer le projet

Docker démarré et vérifié (étape 0), dans ce dossier :

```bash
docker compose up --build
```

Puis ouvrir <http://localhost:5180> :

- onglet **💬 Chat** : envoyez un message, le chatbot répond **TEST** ;
- onglet **📄 Documents** : glissez-déposez des fichiers (ou bouton **+** dans le chat) ; ils sont stockés dans PostgreSQL,
  et le **contexte de chaque chunk est généré automatiquement** par le LLM (colonne « Contexte (LLM) »).

La documentation de l'API est sur <http://localhost:8010/docs>.

Le code est monté en volume : toute modification de `backend/ai.py` est prise en compte automatiquement.
Pour installer une bibliothèque Python, ajoutez-la dans `backend/requirements.txt` puis relancez
`docker compose up --build`.

> **Un problème ?** Les logs : `docker compose logs -f backend`. Repartir de zéro : `docker compose down -v`.
> Erreur de cache Docker (« snapshot ») : `docker builder prune -f`. Erreur « port is already allocated » : changez le
> port concerné dans `.env` (par exemple `FRONTEND_PORT=5190`), relancez, et ouvrez la nouvelle adresse.

## Ce que vous devez faire

Compléter la fonction `generate_reply` dans **[backend/ai.py](backend/ai.py)** :

```python
def generate_reply(messages: list[dict], documents: list[dict]) -> str:
    return "TEST"      # ← à remplacer
```

- `messages` est l'historique complet : `[{"role": "user" | "assistant", "content": "..."}, ...]`
- `documents` liste les fichiers chargés par l'utilisateur (vide s'il n'y en a pas) :
  `[{"id", "name", "content_type", "size", "data": bytes, "text": str | None}, ...]`
  (`text` vaut `None` si le fichier n'est pas du texte, par exemple un PDF).
- la fonction renvoie le texte de la réponse (`str`).

Tout le reste (interface, routes, base de données, Docker) fonctionne déjà et n'a pas besoin d'être modifié.

### Les étapes

On construit le chatbot **en trois temps** : un chatbot classique, puis on lui donne accès aux documents (RAG,
*Retrieval-Augmented Generation*), puis on en fait un agent qui utilise des outils. Vous complétez, **dans l'ordre**,
les `TODO` de [backend/ai.py](backend/ai.py) (puis de [backend/agent_tools.py](backend/agent_tools.py)).
Tant que vous n'avez rien codé, le chatbot répond « TEST ».

| # | Où | Votre travail | Déjà fourni |
|---|---|---|---|
| 0 | — | `docker compose up --build`, charger `exemples/recettes-marmiton.md` et explorer (chunks, descriptions, tables `recipes` / `chunk_embeddings`) | tout |
| **1** | `generate_reply` | **Chatbot classique** : appeler `mistral-small-latest` (`LLM_CHATBOT`) avec des consignes (message « system ») et l'historique de la conversation | `llm.chat` |
| **2** | `embed` | **RAG** — transformer un texte en vecteur avec l'API `mistral-embed` ; elle est **appelée automatiquement** pour chaque chunk dès que vous enregistrez (voir plus bas). *Que vectoriser pour une recette ?* | les chunks, les contextes, la table de vecteurs (**`np.ones`** tant que `embed` n'est pas codée), `llm.limiter` |
| 3 | `vector_search` | vectoriser la question puis écrire le SQL pgvector (distance cosinus `<=>`) | `db.connect()`, `vectorstore.to_literal` |
| 4 | `keyword_search` | recherche par mots-clés (plein texte PostgreSQL) | `db.connect()` |
| 5 | `fuse` | fusionner les deux classements en un seul (ex. Reciprocal Rank Fusion) | — |
| 6 | `rerank` | ne garder que les meilleurs chunks (par le LLM, ou autrement) | `llm.chat` |
| 7 | `generate_reply` | mettre les chunks retrouvés dans le prompt (l'appel de la recherche est déjà câblé), répondre en citant la recette par son **titre exact** : l'interface affiche alors sa fiche sous la réponse | fiches de recettes (photo, ingrédients adaptables au nombre de personnes, ustensiles, étapes) |
| **8** | `agent_tools.py` + `generate_reply` | **Agent** avec tool calling : `search_recipes` en SQL, description des outils, boucle d'appel, questions à l'utilisateur, reformulation de la requête par `LLM_CONTEXT` et multi-requête (« un repas complet » → une recherche pour l'entrée, une pour le plat, une pour le dessert) (voir « Tool calling et mode agentique » plus bas) | `list_utensils`, `call_tool`, `recipes.get`, colonne `utensils_norm` |

Testez après **chaque** étape : après l'étape 1, le chatbot répond (mais ne connaît pas vos recettes) ; après l'étape 2,
la colonne « vecteur » de la fenêtre d'un document passe de « factice » à « calculé » ; après l'étape 3, une question sur
« saumon fumé » doit ramener des recettes au saumon ; etc.

**Affichage des réponses** : l'interface affiche la réponse en **Markdown** (gras, listes, titres, tableaux). Un bloc
```` ```html ... ``` ```` est affiché comme une page, par exemple un graphique Chart.js
(`https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js`), dans un cadre isolé : il ne peut ni lire la page
ni appeler l'API. À vous de dire au modèle, dans son prompt, quand et comment l'utiliser.

**Choix du modèle et raisonnement** : à côté de la zone de saisie, un menu propose les modèles listés dans
[backend/models.py](backend/models.py) (par défaut `LLM_CHATBOT`), et une barre « Effort » règle le **raisonnement** :
Low, Medium, High ou Extra High. Le modèle et le niveau choisis sont ceux qu'utilise `llm.chat(...)` quand vous ne
précisez pas `model=` (voir `llm.current_model()`), sans rien changer à votre code :

- pour les modèles qui savent raisonner (Mistral Small et Large), le niveau est traduit en paramètre `reasoning_effort`
  de l'API : le modèle « réfléchit » avant de répondre, et sa réflexion s'affiche dans un bloc « 💭 Réflexion du modèle » ;
  l'API n'a pas de plafond de réflexion : la réflexion compte dans `max_tokens`, donc `llm.chat` ajoute un budget de
  réflexion selon le niveau (`llm.reasoning_budget()` : 1 000 à 8 000 jetons) et, si la réflexion l'épuise sans
  répondre, redemande une fois sans réflexion ;
- pour tous les modèles, le niveau fixe le nombre maximal d'étapes de l'agent : `llm.agent_max_steps()` (3, 6, 10 ou 16).

## Outils fournis : lire et découper les documents

Le fichier **[backend/tools.py](backend/tools.py)** contient trois fonctions prêtes à l'emploi :

| Fonction | Rôle |
|---|---|
| `extract_text(doc)` | renvoie le texte d'un document : PDF, Word (`.docx`) ou texte (`.txt`, `.md`, `.csv`…). Lève `ValueError` si le fichier est illisible ou si c'est un PDF scanné (sans texte) |
| `chunk_text(text, chunk_size=1000, overlap=200)` | découpe un texte en chunks d'au plus `chunk_size` caractères **sans couper les phrases** : un chunk est fait de phrases entières, un paragraphe qui tient dans un chunk n'est jamais coupé, et le chevauchement (`overlap`) est fait de phrases entières reprises du chunk précédent. Seule une phrase plus longue que `chunk_size` est coupée (à une virgule ou un espace). Les titres Markdown (`#`, `##`…) forment des unités à part. Si le document contient des lignes `---` (séparateurs), chaque bloc entre deux séparateurs devient **un seul chunk**, quelle que soit sa taille (au-delà de 6 000 caractères, il est tout de même découpé) |
| `load_chunks(documents, chunk_size, overlap)` | enchaîne les deux sur tous les documents et renvoie `[{"doc_id", "source", "chunk_id", "text"}, ...]` ; un document illisible est ignoré (message dans les logs) |

```python
from tools import load_chunks

def generate_reply(messages, documents):
    chunks = load_chunks(documents)
    ...   # à vous : quels morceaux sont pertinents pour la question ? comment les donner au LLM ?
```

Pour **voir** ce que donne le découpage d'un document, ouvrez l'onglet **📄 Documents** et **cliquez sur le nom
d'un document** : une fenêtre affiche tous ses chunks. Vous pouvez y modifier `chunk_size` et `overlap` et voir le
résultat se mettre à jour ; le **texte surligné** est le chevauchement avec le chunk précédent.
(La même information est disponible via `GET /api/documents/{id}/chunks?chunk_size=1000&overlap=200`,
dans la documentation de l'API <http://localhost:8010/docs>.)

Les fonctions de `tools.py` ne choisissent pas **quels** chunks utiliser pour répondre à une question : c'est la suite du TP.

## Contextual retrieval : donner un contexte à chaque chunk

**Le problème.** Un chunk lu seul perd ce qui l'entourait. « *Le droit de rétention se perd par le dessaisissement
volontaire.* » : dans quel article ? de quel livre ? Une recherche risque de ne pas le retrouver.

**L'idée** (Anthropic, 2024) : avant d'indexer, un LLM écrit pour **chaque chunk** une phrase qui le replace dans le
document, et on la **colle devant le chunk**. C'est ce texte « enrichi » (`contexte + chunk`) qu'on indexe.

```
chunk seul   : « Le droit de rétention se perd par le dessaisissement volontaire. »
contexte LLM : « Dans l'article 2286 du Code civil (Livre IV), ce passage énumère les cas où le droit de rétention… »
indexé       : contexte + chunk
```

**Comment c'est fait ici** (module **[backend/contextual.py](backend/contextual.py)**, modèle `ministral-14b-latest`, réglable avec `LLM_CONTEXT`) :

1. un **message système** donne les consignes (une seule phrase, ne rien inventer, la position indiquée fait foi) ;
2. un **message utilisateur** contient : le **chemin de titres** du chunk (`Code civil > Livre IV > Article 2286`), un
   **extrait** du document, puis le chunk à situer. L'extrait n'est pas le document entier (trop long, trop cher) :
   c'est le chunk et **les 2 chunks qui le précèdent et les 2 qui le suivent**, **sans jamais sortir de la section**
   (le « chapitre », titre Markdown de niveau 3 ou supérieur) qui contient le chunk. Au maximum, on donne donc la
   section entière ;
3. le modèle répond par le contexte, qui est **mémorisé en base** (table `chunk_contexts`).

Sur le Code civil, l'extrait fait **≈ 3 800 caractères en moyenne** (≈ 1 000 à 1 500 tokens par appel). Réglages
(variables d'environnement, voir `.env.example`) : `CONTEXT_NEIGHBORS` (2), `CONTEXT_SECTION_LEVEL` (3), `LLM_CONTEXT`.

**Automatique à l'ajout d'un document.** Dès qu'un document est chargé, le backend génère en tâche de fond le contexte de
tous ses chunks (`chunk_size=1000`, `overlap=200`, jusqu'à 64 appels en parallèle, dans la limite de requêtes par minute de la clé). La colonne **Contexte (LLM)** de l'onglet
Documents affiche la progression (⏳ 412/1406, puis ✅). Si le serveur redémarre en cours de route, la génération **reprend
là où elle s'était arrêtée**.

**Visualiser.** Cliquez sur un document : un encadré explique la méthode, et chaque chunk affiche son contexte (en vert).
Les boutons permettent de **voir le texte indexé** (contexte + chunk) et **le prompt exact** envoyé au modèle (système +
utilisateur), de régénérer un chunk, ou de contextualiser les premiers chunks pour d'autres réglages de découpage.

> 💸 **Attention au coût.** Chaque chunk = un appel au LLM. D'après nos essais, un appel envoie ≈ 1 000 à 1 500 tokens :
> le **Code civil (1 406 chunks) représente de l'ordre de 2 millions de tokens d'entrée** (estimation d'après quelques
> essais : je ne l'ai pas lancé en entier) et **2 à 3 minutes** de traitement (la limite de 500 requêtes/minute de la clé de test est le facteur limitant : 700 chunks mesurés en 76 s, sans aucune erreur 429). Vérifiez le tarif de votre compte Mistral
> avant de charger de très gros documents. Pour tester sans dépenser : mettez `AUTO_CONTEXTUALIZE=false` dans `.env`
> (la génération reste possible chunk par chunk depuis la fenêtre du document), ou chargez un extrait. Un document de plus
> de `AUTO_CONTEXT_MAX_CHUNKS` chunks (5 000 par défaut) n'est **pas** contextualisé automatiquement.

**Utiliser les contextes dans `ai.py`.** Les contextes sont indexés comme les chunks de `chunk_text(extract_text(doc), 1000, 200)` :

```python
import db, tools
from contextual import contextualize_text

text = tools.extract_text(doc)
chunks = tools.chunk_text(text, 1000, 200)
contexts = db.get_contexts(doc["id"], 1000, 200)         # {index du chunk: contexte}
to_index = [contextualize_text(contexts.get(i, ""), c) for i, c in enumerate(chunks)]
```

Le fichier **[backend/llm.py](backend/llm.py)** contient un petit client Mistral (`chat(messages)`) que vous pouvez aussi
utiliser pour générer la réponse du chatbot.

## Stockage vectoriel (pgvector)

PostgreSQL tourne avec l'extension **pgvector** (image `pgvector/pgvector:pg16`) : elle permet de stocker des vecteurs
et de chercher les plus proches. À l'ajout d'un document, **chaque chunk reçoit une ligne** dans la table
`chunk_embeddings`, avec son texte et un **vecteur factice** (`np.ones(1024)`, 1 024 fois la valeur 1) : à vous de calculer les vrais embeddings.
La fonction **`embed(text)`** de [backend/ai.py](backend/ai.py) existe déjà : elle renvoie pour l'instant `np.ones(1024)`, avec un `TODO` pour appeler `mistral-embed` (dimension 1 024, vecteurs déjà normalisés).

### Quand `embed()` est-elle appelée ? Automatiquement

```
document ajouté ──► chunks (vecteur factice) ──► contexte de chaque chunk (LLM) ──► embed(contexte + chunk) ──► vecteur enregistré
```

Vous n'avez **rien à appeler** : le backend ([backend/embedding.py](backend/embedding.py)) appelle `ai.embed` pour chaque chunk,
en parallèle, avec le texte **contexte + chunk** :

1. **à l'ajout d'un document**, une fois les contextes générés ;
2. **dès que vous enregistrez `ai.py`** : le serveur redémarre tout seul et calcule les embeddings des documents déjà
   chargés dont les vecteurs sont encore factices. Écrivez `embed()`, sauvegardez, regardez les badges passer à « vecteur calculé » ;
3. **à la demande**, avec les boutons *Calculer les embeddings manquants* / *Tout recalculer* de la fenêtre d'un document
   (ou `POST /api/documents/{id}/embed`).

Tant que `embed()` renvoie `np.ones`, il n'y a rien à enregistrer : la fenêtre du document et les logs
(`docker compose logs backend`) vous le disent. Si votre fonction lève une exception ou renvoie un vecteur qui n'a pas
la forme `(1024,)`, l'erreur est affichée de la même façon. `embed()` est appelée **en parallèle** (jusqu'à
`AUTO_CONTEXT_WORKERS` appels simultanés) : elle doit donc respecter la limite de requêtes par minute (`llm.limiter`).

| Colonne | Contenu |
|---|---|
| `document_id`, `chunk_size`, `overlap`, `chunk_index` | identifient le chunk (réglages indexés : `1000` / `200`) |
| `content` | le texte du chunk |
| `embedding` | `vector(1024)` : **des 1 au départ** (`mistral-embed` produit des vecteurs de dimension 1024) |

Fonctions fournies dans **[backend/vectorstore.py](backend/vectorstore.py)** :

```python
import vectorstore

vectorstore.get_chunks(doc_id=None)                      # [{"document_id", "source", "chunk_index", "content", "context"}, ...]
vectorstore.get_chunks(doc_id, with_embedding=True)      # ... + "embedding": liste de 1024 floats
vectorstore.set_embeddings([(doc_id, chunk_index, vecteur), ...])   # enregistre des vecteurs (vérifie la dimension)
vectorstore.embedded_indices(doc_id)                     # chunks dont le vecteur n'est plus le vecteur factice
```

Pour **chercher**, vous écrivez le SQL vous-même avec `db.connect()`. pgvector fournit les opérateurs `<->` (distance
euclidienne), `<#>` (produit scalaire négatif) et `<=>` (distance cosinus) :

```python
with db.connect() as conn:
    rows = conn.execute("SELECT chunk_index, content FROM chunk_embeddings "
                        "ORDER BY embedding <-> %s::vector LIMIT 5", ("[0.1, 0.2, ...]",)).fetchall()
```

⚠️ **Pièges** (testés) :
- tant que les vecteurs sont factices, **ils sont tous identiques** : la distance (euclidienne, cosinus ou produit scalaire)
  vaut la même chose pour tous les chunks, donc l'ordre des résultats est arbitraire. Si vos résultats ne changent pas
  quand vous changez de question, c'est que les vecteurs n'ont pas encore été calculés (voir le badge ci-dessous) ;
- la dimension doit être exactement 1 024 (sinon `set_embeddings` refuse le vecteur) ;
- l'API Mistral est limitée en requêtes par minute : `llm.limiter.acquire()` (voir `llm.py`) attend si besoin.

Dans la fenêtre d'un document (clic sur son nom), chaque chunk affiche un badge **🔢 vecteur factice** puis
**🔢 vecteur calculé** quand vous l'avez rempli. Pour explorer en SQL :

```bash
docker compose exec db psql -U chatbot -d chatbot -c "SELECT chunk_index, vector_norm(embedding) FROM chunk_embeddings LIMIT 5;"
```

Un vecteur factice (que des 1) a pour norme 32 (√1024) ; un vrai embedding aura une autre norme.

> ⚠️ **Mise à jour depuis une version précédente** : l'image PostgreSQL a changé (pgvector). Si vous aviez déjà lancé le
> projet, repartez d'une base neuve : `docker compose down -v && docker compose up --build` (vos documents seront à recharger).

## Document d'exemple : le Code civil

Le fichier **[exemples/code-civil.md](exemples/code-civil.md)** est un bon document pour tester : le Code civil français
en un seul fichier Markdown structuré (2 802 articles, ≈ 1 Mo).

```
## Livre Ier
### Titre II
#### Article 34
Les actes de l'état civil énonceront l'année, le jour et l'heure où ils seront reçus...
```

Chargez-le dans l'onglet **📄 Documents**, puis cliquez dessus pour voir son découpage. Essayez par exemple
`chunk_size` = 500, 1000 et 2000 : combien de chunks obtenez-vous ? Que perd-on, que gagne-t-on ?

- **Source** : dépôt [steeve/france.code-civil](https://github.com/steeve/france.code-civil) (texte de Légifrance),
  sous licence [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/) : citez la source si vous le réutilisez.
- ⚠️ C'est un **instantané de mars 2015** : le texte n'est plus à jour (réforme du droit des contrats de 2016, etc.). Ne vous
  en servez pas pour un vrai conseil juridique ; il sert uniquement d'exemple.
- Les titres n'ont que des numéros (« Titre II »), sans intitulé, car c'est ce que contient la source.

## Tool calling et mode agentique

Après le RAG, on passe à un **agent** : le modèle ne se contente plus de chercher dans des documents, il peut
**appeler des outils** (des fonctions Python que vous écrivez) et **poser des questions à l'utilisateur**.

- Exemple : l'utilisateur dit « je veux faire un gâteau » ; le modèle répond « Vous avez quoi comme ustensiles ? »,
  comprend la réponse, puis lance une recherche **SQL** qui ne garde que les recettes réalisables avec ces ustensiles.
- Pour que ce filtre soit possible, les ustensiles des recettes sont **normalisés** (« formidable plat à four »,
  « pôele » → `plat à four`, `poêle`) dans la colonne `utensils_norm` de la table `recipes`.
- Le fichier **`backend/agent_tools.py`** est celui où vous écrivez ces outils (`search_recipes` en SQL, leur description
  JSON pour le modèle, la boucle d'appel dans `ai.py`). À ne pas confondre avec `tools.py` (lecture et découpage des
  documents), que vous n'avez pas à modifier.

## Explorer la base de données

```bash
docker compose exec db psql -U chatbot -d chatbot
chatbot=# SELECT id, name, size, created_at FROM documents;
chatbot=# SELECT chunk_index, context FROM chunk_contexts WHERE document_id = 1 ORDER BY chunk_index LIMIT 5;
chatbot=# SELECT chunk_index, vector_dims(embedding), vector_norm(embedding) FROM chunk_embeddings LIMIT 5;
```

Ou depuis un client graphique (DBeaver, TablePlus…) : hôte `localhost`, port `5433`, base et utilisateur `chatbot`, **sans mot de passe**.
Les données sont conservées entre deux lancements ; pour repartir de zéro : `docker compose down -v`.

## Arborescence

```
TP_chatbot/
├── INSTALL_DOCKER.md      # installer Docker (macOS, Windows, Linux) ; aussi en INSTALL_DOCKER.pdf
├── .env                   # clé Mistral + ports (voir .env.example)
├── exemples/
│   └── code-civil.md      # document d'exemple à charger dans l'interface
├── docker-compose.yml     # db (PostgreSQL) + backend + frontend
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py            # API (fournie)
│   ├── db.py              # accès PostgreSQL (fourni)
│   ├── tools.py           # lecture et découpage des documents (fourni)
│   ├── models.py          # modèles proposés dans le menu du chat (fourni)
│   ├── agent_tools.py     # outils de l'agent : tool calling, recherche SQL (à compléter)
│   ├── recipes.py         # recettes structurées, ustensiles normalisés (fourni)
│   ├── llm.py             # client Mistral (fourni)
│   ├── contextual.py      # contextual retrieval : prompts + génération du contexte d'un chunk (fourni)
│   ├── autocontext.py     # contextualisation automatique en tâche de fond (fourni)
│   ├── vectorstore.py     # stockage vectoriel pgvector : chunks et vecteurs (fourni)
│   ├── embedding.py       # appelle ai.embed pour chaque chunk, à l'ajout d'un document (fourni)
│   └── ai.py              # ⭐ partie IA : à compléter
└── frontend/              # interface React (fournie) : Chat + Documents
```
