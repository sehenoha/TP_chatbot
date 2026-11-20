// Petit rendu Markdown (titres, gras, italique, listes, liens, code, tableaux, séparateurs) pour les réponses du chatbot.
// Sans bibliothèque et sans HTML brut : le texte est transformé en éléments React (aucun risque d'injection).
import { Fragment } from "react";
import { RecipeCard, RecipeCards } from "./Recipes.jsx";
import HtmlFrame from "./HtmlFrame.jsx";

const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*|__[^_]+__)|(\*[^*\s][^*]*\*|_[^_\s][^_]*_)|(\[[^\]]+\]\([^)\s]+\))/;

// Recettes citées dans la réponse en cours de rendu : la bulle de chaque recette est placée juste après la
// première occurrence de son titre (le rendu est synchrone : une variable de module suffit).
let CITED = null;

// Le titre est cherché sans tenir compte de la casse, des accents, des ligatures (« Bœuf » = « Boeuf »),
// des apostrophes (' ou ’) ni des espaces multiples.
const VARIANTS = { a: "aàâä", e: "eéèêë", i: "iîï", o: "oôö", u: "uùûü", c: "cç", y: "yÿ", n: "nñ" };
const titleRegexes = new Map();
function titleRegex(title) {
  if (!titleRegexes.has(title)) {
    const t = title.toLowerCase().replace(/œ/g, "oe").replace(/æ/g, "ae").normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    let pattern = "";
    for (let i = 0; i < t.length; i++) {
      const c = t[i];
      if ((c === "o" || c === "a") && t[i + 1] === "e") {
        pattern += c === "o" ? "(?:o[eéèêë]|œ)" : "(?:a[eéèêë]|æ)";
        i++;
      } else if (VARIANTS[c]) pattern += `[${VARIANTS[c]}]`;
      else if (/\s/.test(c)) pattern += "\\s+";
      else if (c === "'" || c === "’") pattern += "['’]";
      else if ('"«»“”„'.includes(c)) pattern += '\\s*["«»“”„]?\\s*';
      else pattern += c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
    titleRegexes.set(title, new RegExp(`(?<![\\p{L}\\p{N}])${pattern}(?![\\p{L}\\p{N}])`, "iu"));
  }
  return titleRegexes.get(title);
}

// Le titre est-il vraiment cité ici ? Début du segment (gras, titre...), après une ponctuation, avec une majuscule,
// ou titre long (5 mots ou plus). Évite « 1 c. à soupe de lait » -> recette « Soupe de lait ».
function isMention(text, start, title) {
  const before = text.slice(0, start).trimEnd();
  if (!before || !/[\p{L}\p{N}]$/u.test(before)) return true;
  const first = text[start];
  return first !== first.toLowerCase() || title.trim().split(/\s+/).length >= 5;
}

function withCards(text, key) {
  if (!CITED) return [text];
  const out = [];
  let rest = text;
  for (;;) {
    let best = null;
    CITED.recipes.forEach((r, k) => {
      if (CITED.placed.has(k)) return;
      const re = new RegExp(titleRegex(r.title).source, "giu");
      let m;
      while ((m = re.exec(rest))) {
        if (isMention(rest, m.index, r.title)) break;      // première occurrence vraiment mise en avant
      }
      if (m && (!best || m.index < best.pos)) best = { pos: m.index, len: m[0].length, k, r };
    });
    if (!best) break;
    CITED.placed.add(best.k);
    const end = best.pos + best.len;
    out.push(rest.slice(0, end), <RecipeCard key={`${key}-r${best.k}`} recipe={best.r} servings={CITED.servings} />);
    rest = rest.slice(end);
  }
  if (rest) out.push(rest);
  return out;
}

function inline(text, key = "i") {
  const out = [];
  let rest = text;
  let n = 0;
  while (rest) {
    const m = rest.match(INLINE);
    if (!m) {
      out.push(...withCards(rest, `${key}-${n}`));
      break;
    }
    if (m.index > 0) out.push(...withCards(rest.slice(0, m.index), `${key}-${n}b`));
    const tok = m[0];
    const k = `${key}-${n++}`;
    if (m[1]) out.push(<code key={k}>{tok.slice(1, -1)}</code>);
    else if (m[2]) out.push(<strong key={k}>{inline(tok.slice(2, -2), k)}</strong>);
    else if (m[3]) out.push(<em key={k}>{inline(tok.slice(1, -1), k)}</em>);
    else {
      const [, label, url] = tok.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
      out.push(
        /^https?:\/\//.test(url) ? (
          <a key={k} href={url} target="_blank" rel="noreferrer">
            {inline(label, k)}
          </a>
        ) : (
          <span key={k}>{inline(label, k)}</span>   // lien vers un fichier local : on garde juste le texte
        ),
      );
    }
    rest = rest.slice(m.index + tok.length);
  }
  return out;
}

const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

function list(lines, start) {
  // Liste (éventuellement imbriquée par l'indentation) à partir de lines[start] ; renvoie [élément, ligne suivante]
  const indent = lines[start].match(LIST_ITEM)[1].length;
  const ordered = /\d/.test(lines[start].match(LIST_ITEM)[2]);
  const items = [];
  let i = start;
  while (i < lines.length) {
    const m = lines[i].match(LIST_ITEM);
    if (!m || m[1].length < indent) break;
    if (m[1].length === indent && /\d/.test(m[2]) !== ordered) break;     // liste à puces -> numérotée : nouvelle liste
    if (m[1].length > indent && items.length) {
      const [sub, next] = list(lines, i);
      items[items.length - 1].children.push(sub);
      i = next;
      continue;
    }
    items.push({ text: m[3], children: [] });
    i++;
    while (i < lines.length && lines[i].trim() && !LIST_ITEM.test(lines[i]) && /^\s+/.test(lines[i])) {
      items[items.length - 1].text += " " + lines[i].trim();     // suite d'un élément sur la ligne suivante
      i++;
    }
  }
  const Tag = ordered ? "ol" : "ul";
  return [
    <Tag key={`l${start}`}>
      {items.map((it, j) => (
        <li key={j}>
          {inline(it.text, `l${start}-${j}`)}
          {it.children}
        </li>
      ))}
    </Tag>,
    i,
  ];
}

const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

// « Source : ... » / « - *Source* : [fichier.md, section 466] » : inutile, la bulle de la recette est à côté de son titre
const SOURCE_LINE = /^\s*(?:[-*+]\s+)?[*_]*sources?[*_]*\s*[*_]*\s*:/i;

export default function Markdown({ text, recipes = [], servings = null }) {
  const lines = (text || "").replace(/\r\n/g, "\n").split("\n").filter((l) => !(recipes.length && SOURCE_LINE.test(l)));
  const blocks = [];
  CITED = { recipes, placed: new Set(), servings };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
    } else if (line.trim().startsWith("```")) {
      const lang = line.trim().slice(3).trim().toLowerCase();
      const code = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i++]);
      i++;
      // Bloc ```html : rendu (graphique, visuel) dans un cadre isolé ; les autres blocs restent du code
      blocks.push(lang === "html"
        ? <HtmlFrame key={`c${i}`} html={code.join("\n")} />
        : <pre key={`c${i}`}><code>{code.join("\n")}</code></pre>);
    } else if (/^\s*#{1,6}\s/.test(line)) {
      const level = Math.min(line.trim().match(/^#+/)[0].length + 2, 6);   // # -> h3 : titres raisonnables dans une bulle
      const H = `h${level}`;
      blocks.push(<H key={`h${i}`}>{inline(line.trim().replace(/^#+\s*/, ""), `h${i}`)}</H>);
      i++;
    } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      blocks.push(<hr key={`r${i}`} />);
      i++;
    } else if (LIST_ITEM.test(line)) {
      const [el, next] = list(lines, i);
      blocks.push(el);
      i = next;
    } else if (line.includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const head = cells(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
      blocks.push(
        <table key={`t${i}`}>
          <thead><tr>{head.map((c, j) => <th key={j}>{inline(c, `th${i}${j}`)}</th>)}</tr></thead>
          <tbody>{rows.map((r, a) => <tr key={a}>{r.map((c, b) => <td key={b}>{inline(c, `td${i}${a}${b}`)}</td>)}</tr>)}</tbody>
        </table>,
      );
    } else if (/^\s*>/.test(line)) {
      const quote = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
      blocks.push(<blockquote key={`q${i}`}>{inline(quote.join(" "), `q${i}`)}</blockquote>);
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && !LIST_ITEM.test(lines[i]) && !/^\s*(#{1,6}\s|```|>)/.test(lines[i])) para.push(lines[i++]);
      blocks.push(
        <p key={`p${i}`}>
          {para.map((l, j) => (
            <Fragment key={j}>
              {j > 0 && <br />}
              {inline(l, `p${i}-${j}`)}
            </Fragment>
          ))}
        </p>,
      );
    }
  }
  const rest = recipes.filter((_, k) => !CITED.placed.has(k));   // titres non retrouvés tels quels dans le texte : à la fin
  CITED = null;
  if (rest.length) blocks.push(<RecipeCards key="rc-rest" recipes={rest} servings={servings} />);
  return <div className="md">{blocks}</div>;
}
