import { useEffect, useRef, useState } from "react";
import {
  generateContext,
  getChunks,
  getContextualization,
  getContexts,
  getEmbedded,
  getEmbeddingStatus,
  startEmbedding,
} from "./api.js";

const MAX_DISPLAYED = 200;
const BATCH_SIZE = 10;

// Longueur de la partie commune entre la fin du chunk précédent et le début du chunk courant
function overlapLength(prev, cur, max) {
  const limit = Math.min(prev.length, cur.length, max, 2000);
  for (let k = limit; k > 0; k--) {
    if (prev.endsWith(cur.slice(0, k))) return k;
  }
  return 0;
}

function Explanation({ model }) {
  return (
    <details className="explain" open>
      <summary>🧠 Contextual retrieval : c'est quoi ?</summary>
      <div className="explain-body">
        <p>
          <b>Le problème.</b> Une fois le document découpé, chaque chunk est lu <i>seul</i>. Il perd le contexte qui
          l'entourait : « Le droit de rétention se perd par le dessaisissement volontaire. » Dans quel article ? Dans quel
          livre ? Une recherche risque de ne pas le retrouver, ou de le confondre avec un autre passage.
        </p>
        <p>
          <b>L'idée</b> (Anthropic, 2024). Avant d'indexer, on demande à un LLM de rédiger pour <b>chaque chunk</b> une phrase
          qui le replace dans le document, puis on la <b>colle devant le chunk</b>. C'est ce texte « enrichi » qu'on indexe
          (embeddings, recherche par mots-clés…).
        </p>
        <ol>
          <li>
            On envoie au LLM un <b>message système</b> (les consignes) et un <b>message utilisateur</b> contenant : le chemin de
            titres du chunk, un <b>extrait</b> du document (le chunk et les <b>2 chunks avant et après</b>, sans jamais sortir de
            la <b>section / chapitre</b> qui le contient) et le chunk à situer.
          </li>
          <li>Le LLM répond par <b>une phrase de contexte</b>.</li>
          <li>
            On indexe <code>contexte + chunk</code>. Le chunk affiché à l'utilisateur, lui, ne change pas.
          </li>
        </ol>
        <div className="before-after">
          <div>
            <small>Sans contexte</small>
            <p>« Le droit de rétention se perd par le dessaisissement volontaire. »</p>
          </div>
          <div className="with">
            <small>Avec contexte</small>
            <p>
              <mark className="ctx-mark">Dans l'article 2286 du Code civil, ce passage énumère les cas où le droit de rétention peut être exercé.</mark>{" "}
              « Le droit de rétention se perd par le dessaisissement volontaire. »
            </p>
          </div>
        </div>
        <p className="explain-note">
          Modèle utilisé : <code>{model || "ministral-14b-latest"}</code>. Chaque appel envoie environ 1 000 à 1 500 tokens
          (un extrait, pas le document entier) : un gros document reste tout de même long à traiter en entier. Les contextes générés sont mémorisés en base. Le chatbot
          ne les utilise pas encore : c'est à vous de le faire dans <code>ai.py</code>.
        </p>
      </div>
    </details>
  );
}

function ChunkCard({ index, chunk, overlapChars, context, detail, busy, canGenerate, onGenerate, model, vector }) {
  const [showIndexed, setShowIndexed] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  return (
    <article className={`chunk${context ? " has-context" : ""}`}>
      <div className="chunk-head">
        <b>Chunk {index}</b>
        <span>
          {vector !== undefined && (
            <span className={`vec ${vector ? "set" : "placeholder"}`} title="Vecteur stocké dans PostgreSQL (pgvector)">
              🔢 {vector ? "vecteur calculé" : "vecteur factice"}
            </span>
          )}
          {chunk.length} caractères
        </span>
      </div>

      {context && (
        <div className="ctx">
          <div className="ctx-label">✨ Contexte généré par le LLM</div>
          <p>{context}</p>
        </div>
      )}

      <p>
        {overlapChars > 0 && <mark>{chunk.slice(0, overlapChars)}</mark>}
        {chunk.slice(overlapChars)}
      </p>

      <div className="chunk-actions">
        <button type="button" className="small" disabled={busy || !canGenerate} onClick={onGenerate}>
          {busy ? "Génération…" : context ? "↻ Régénérer" : "✨ Contextualiser"}
        </button>
        {context && (
          <button type="button" className="small ghost" onClick={() => setShowIndexed(!showIndexed)}>
            {showIndexed ? "Masquer" : "Voir"} le texte indexé
          </button>
        )}
        {detail && (
          <button type="button" className="small ghost" onClick={() => setShowPrompt(!showPrompt)}>
            {showPrompt ? "Masquer" : "Voir"} le prompt
          </button>
        )}
        {detail?.usage?.prompt_tokens && (
          <span className="usage">
            {detail.usage.prompt_tokens} tokens en entrée · {detail.usage.completion_tokens} en sortie
          </span>
        )}
      </div>

      {showIndexed && context && (
        <div className="indexed">
          <small>Texte qui serait indexé (contexte + chunk) :</small>
          <pre>
            <span className="ctx-mark">{context}</span>
            {"\n\n"}
            {chunk}
          </pre>
        </div>
      )}

      {showPrompt && detail && (
        <div className="prompt-view">
          <small>
            Message <b>système</b> (les consignes, identiques pour tous les chunks) :
          </small>
          <pre>{detail.system_prompt}</pre>
          <small>
            Message <b>utilisateur</b> (un extrait du document
            {detail.window
              ? ` : ${detail.window.before} chunk(s) avant, ${detail.window.after} après${detail.window.section ? `, dans la section « ${detail.window.section} »` : ""}`
              : ""}
            , puis le chunk) :
          </small>
          <pre>{detail.prompt}</pre>
          <small>
            Réponse du modèle <code>{detail.model || model}</code> : « {detail.context} »
          </small>
        </div>
      )}
    </article>
  );
}

export default function ChunkViewer({ doc, onClose }) {
  const [chunkSize, setChunkSize] = useState(1000);
  const [overlap, setOverlap] = useState(200);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const [contexts, setContexts] = useState({});   // index du chunk -> contexte (mémorisé en base)
  const [details, setDetails] = useState({});     // index -> détails de la dernière génération (prompt, tokens…)
  const [busy, setBusy] = useState({});           // index -> true pendant la génération
  const [ctxError, setCtxError] = useState("");
  const [batch, setBatch] = useState(null);       // { done, total } pendant une génération en série
  const settingsKey = useRef("");
  const [job, setJob] = useState(null);           // état de la contextualisation automatique de ce document
  const [emb, setEmb] = useState(null);           // état du calcul des embeddings (appel de ai.embed)
  const [embTick, setEmbTick] = useState(0);

  // Suivi de la contextualisation automatique : tant qu'elle tourne, on récupère les contextes au fur et à mesure
  useEffect(() => {
    let stopped = false;
    let timer;
    async function poll() {
      try {
        const jobs = await getContextualization();
        if (stopped) return;
        const j = jobs[doc.id] || null;
        setJob(j);
        if (j && j.status === "running" && settingsKey.current === `${j.chunk_size}-${j.overlap}`) {
          const r = await getContexts(doc.id, j.chunk_size, j.overlap);
          if (!stopped && settingsKey.current === `${j.chunk_size}-${j.overlap}`) setContexts((c) => ({ ...c, ...r.contexts }));
        }
        if (!stopped && j && j.status === "running") timer = setTimeout(poll, 2500);
      } catch {
        /* le suivi est facultatif */
      }
    }
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [doc.id]);

  // Fermeture avec Échap
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Recalcul (avec un léger délai) quand on change les réglages
  useEffect(() => {
    const size = Number(chunkSize);
    const ov = Number(overlap);
    if (!Number.isInteger(size) || size <= 0) return setError("chunk_size doit être un entier positif");
    if (!Number.isInteger(ov) || ov < 0 || ov >= size) return setError("overlap doit être compris entre 0 et chunk_size − 1");

    setError("");
    setLoading(true);
    let cancelled = false;
    const timer = setTimeout(() => {
      getChunks(doc.id, size, ov)
        .then((r) => {
          if (cancelled) return;
          settingsKey.current = `${size}-${ov}`;
          setResult(r);
          setContexts(r.contexts || {});
          setDetails({});
        })
        .catch((e) => {
          if (!cancelled) {
            setResult(null);
            setError(e.message);
          }
        })
        .finally(() => !cancelled && setLoading(false));
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [doc.id, chunkSize, overlap]);

  // Suivi du calcul des embeddings : tant qu'il tourne, on rafraîchit l'état et les badges « vecteur calculé »
  useEffect(() => {
    let stopped = false;
    let timer;
    async function poll() {
      try {
        const all = await getEmbeddingStatus();
        if (stopped) return;
        const st = all[doc.id] || null;
        setEmb(st);
        if (settingsKey.current === "1000-200") {
          const r = await getEmbedded(doc.id, 1000, 200);
          if (!stopped) setResult((prev) => (prev ? { ...prev, embedded: r.embedded } : prev));
        }
        if (!stopped && st && st.running) timer = setTimeout(poll, 1500);
      } catch {
        /* le suivi est facultatif */
      }
    }
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [doc.id, embTick]);

  async function runEmbedding(force) {
    try {
      await startEmbedding(doc.id, force);
    } catch (e) {
      setCtxError(e.message);
    }
    setEmbTick((t) => t + 1);   // relance le suivi
  }

  async function generate(i) {
    const key = settingsKey.current;
    setCtxError("");
    setBusy((b) => ({ ...b, [i]: true }));
    try {
      const r = await generateContext(doc.id, i, Number(chunkSize), Number(overlap));
      if (settingsKey.current !== key) return true; // les réglages ont changé entre-temps : on ignore ce résultat
      setContexts((c) => ({ ...c, [i]: r.context }));
      setDetails((d) => ({ ...d, [i]: r }));
      return true;
    } catch (e) {
      setCtxError(e.message);
      return false;
    } finally {
      setBusy((b) => {
        const n = { ...b };
        delete n[i];
        return n;
      });
    }
  }

  async function generateFirst() {
    const todo = [];
    for (let i = 0; i < Math.min(chunks.length, MAX_DISPLAYED) && todo.length < BATCH_SIZE; i++) {
      if (!contexts[i]) todo.push(i);
    }
    let done = 0;
    setBatch({ done, total: todo.length });
    // Tous les appels partent en parallèle (les erreurs éventuelles s'affichent sous le bouton)
    await Promise.all(
      todo.map(async (i) => {
        await generate(i);
        done += 1;
        setBatch({ done, total: todo.length });
      })
    );
    setBatch(null);
  }

  const chunks = result?.chunks ?? [];
  const shown = chunks.slice(0, MAX_DISPLAYED);
  const lengths = chunks.map((c) => c.length);
  const avg = lengths.length ? Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length) : 0;
  const canGenerate = !!result?.api_key_set;
  const embeddedSet = new Set(result?.embedded ?? []);   // chunks dont le vecteur n'est plus le vecteur factice (réglages 1000/200)
  const nContexts = Object.keys(contexts).length;
  const anyBusy = Object.keys(busy).length > 0;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={`Chunking de ${doc.name}`}>
        <div className="modal-header">
          <h2>✂️ {doc.name}</h2>
          <button type="button" className="link" onClick={onClose} aria-label="Fermer">
            ✕
          </button>
        </div>

        <div className="chunk-controls">
          <label>
            chunk_size
            <input type="number" min="1" value={chunkSize} onChange={(e) => setChunkSize(e.target.value)} />
          </label>
          <label>
            overlap
            <input type="number" min="0" value={overlap} onChange={(e) => setOverlap(e.target.value)} />
          </label>
          {result && !error && (
            <span className="stats">
              {result.n_chars.toLocaleString("fr-FR")} caractères → <b>{result.n_chunks}</b> chunks
              (taille moyenne {avg})
            </span>
          )}
        </div>

        <div className="modal-body">
          {error && <p className="error-text">⚠️ {error}</p>}
          {loading && !error && <p className="empty-inline">Calcul…</p>}
          {!error && result && (
            <>
              <Explanation model={result.context_model} />

              {!result.api_key_set && (
                <p className="error-text">
                  ⚠️ <code>MISTRAL_API_KEY</code> est vide : renseignez-la dans le fichier <code>.env</code>, puis relancez{" "}
                  <code>docker compose up</code> pour pouvoir générer des contextes.
                </p>
              )}

              {job && (
                <p className={`auto-banner ${job.status}`}>
                  {job.status === "running" &&
                    `⏳ Contextualisation automatique en cours : ${job.done}/${job.total || "…"} chunks (réglages chunk_size=${job.chunk_size}, overlap=${job.overlap}).`}
                  {job.status === "done" &&
                    `✅ Contextualisation automatique terminée : ${job.done}/${job.total} chunks (réglages chunk_size=${job.chunk_size}, overlap=${job.overlap}). Avec d'autres réglages, les contextes se génèrent à la demande.`}
                  {["partial", "error", "skipped"].includes(job.status) && `⚠️ ${job.message}`}
                </p>
              )}

              {result.vector_store && (
                <div className={`auto-banner ${emb?.error || emb?.failed ? "error" : emb?.running ? "running" : emb?.stub ? "partial" : "done"}`}>
                  <div>
                    {emb?.running &&
                      `⏳ Calcul des embeddings (ai.embed) : ${emb.done}/${emb.total} chunks…`}
                    {!emb?.running && emb?.stub &&
                      "⚠️ ai.embed() renvoie encore le vecteur factice (np.ones) : rien n'est enregistré. Complétez le TODO dans backend/ai.py ; le calcul se relance tout seul quand vous enregistrez le fichier."}
                    {!emb?.running && !emb?.stub && (emb?.error || emb?.failed) &&
                      `⚠️ ${emb.failed || 0} chunk(s) en échec. ${emb.error || ""} (détails : docker compose logs backend)`}
                    {!emb?.running && !emb?.stub && !emb?.error && !emb?.failed &&
                      `🔢 ${embeddedSet.size}/${result.n_chunks} chunks avec un vecteur calculé (réglages 1000 / 200).`}
                  </div>
                  <div className="chunk-actions">
                    <button type="button" className="small" disabled={!!emb?.running} onClick={() => runEmbedding(false)}>
                      🔢 Calculer les embeddings manquants
                    </button>
                    <button type="button" className="small ghost" disabled={!!emb?.running} onClick={() => runEmbedding(true)}>
                      ↻ Tout recalculer
                    </button>
                  </div>
                </div>
              )}

              <div className="batch-bar">
                <button type="button" className="small" disabled={!canGenerate || anyBusy || batch} onClick={generateFirst}>
                  ✨ Contextualiser les {BATCH_SIZE} premiers chunks sans contexte
                </button>
                <span className="usage">
                  {batch ? `Génération ${batch.done}/${batch.total}…` : `${nContexts} chunk${nContexts > 1 ? "s" : ""} contextualisé${nContexts > 1 ? "s" : ""}`}
                </span>
              </div>
              {ctxError && <p className="error-text">⚠️ {ctxError}</p>}

              <p className="legend">
                Le <mark>texte surligné</mark> est le chevauchement avec le chunk précédent.
              </p>
              {shown.map((c, i) => (
                <ChunkCard
                  key={i}
                  index={i}
                  chunk={c}
                  overlapChars={i === 0 ? 0 : overlapLength(chunks[i - 1], c, Number(overlap))}
                  context={contexts[i]}
                  detail={details[i]}
                  busy={!!busy[i]}
                  canGenerate={canGenerate}
                  onGenerate={() => generate(i)}
                  model={result.context_model}
                  vector={
                    result.vector_store && Number(chunkSize) === 1000 && Number(overlap) === 200
                      ? embeddedSet.has(i)
                      : undefined
                  }
                />
              ))}
              {chunks.length > MAX_DISPLAYED && (
                <p className="empty-inline">… et {chunks.length - MAX_DISPLAYED} autres chunks non affichés.</p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
