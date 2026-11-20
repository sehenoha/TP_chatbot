import { useEffect, useRef, useState } from "react";
import {
  deleteDocument,
  downloadUrl,
  formatSize,
  getContextualization,
  getDocuments,
  restartContextualization,
  uploadDocuments,
} from "./api.js";
import ChunkViewer from "./ChunkViewer.jsx";

// Affiche l'état de la contextualisation automatique d'un document
function ContextStatus({ job, onRestart }) {
  if (!job) {
    return (
      <button type="button" className="small ghost" onClick={onRestart} title="Générer le contexte de tous les chunks">
        ✨ Contextualiser
      </button>
    );
  }
  const { status, total, done, message } = job;
  const pct = total ? Math.round((done / total) * 100) : 0;
  if (status === "running") {
    return (
      <div className="ctx-status" title="Contextualisation en cours (génération du contexte de chaque chunk par le LLM)">
        <div className="progress">
          <div style={{ width: `${pct}%` }} />
        </div>
        <small>{total ? `⏳ ${done}/${total}` : "⏳ préparation…"}</small>
      </div>
    );
  }
  if (status === "done") {
    return <small className="ok" title="Tous les chunks ont leur contexte">✅ {done}/{total}</small>;
  }
  return (
    <div className="ctx-status" title={message || ""}>
      <small className={status === "skipped" ? "warn" : "bad"}>
        {status === "skipped" ? "⏭️ trop volumineux" : status === "partial" ? `⚠️ ${done}/${total}` : "⚠️ erreur"}
      </small>
      {status !== "skipped" && (
        <button type="button" className="small ghost" onClick={onRestart}>
          ↻ Relancer
        </button>
      )}
      {message && <small className="msg">{message}</small>}
    </div>
  );
}

export default function Documents({ active }) {
  const [documents, setDocuments] = useState([]);
  const [jobs, setJobs] = useState({});             // id du document -> état de la contextualisation
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);   // document dont on visualise le chunking
  const inputRef = useRef(null);

  const refreshJobs = () => getContextualization().then(setJobs).catch(() => {});
  const anyRunning = Object.values(jobs).some((j) => j.status === "running");

  useEffect(() => {
    if (active) {
      getDocuments().then(setDocuments).catch((e) => setError(e.message));
      refreshJobs();
    }
  }, [active]);

  // Tant qu'une contextualisation est en cours, on actualise la progression toutes les 2 secondes
  useEffect(() => {
    if (!active || !anyRunning) return;
    const timer = setInterval(refreshJobs, 2000);
    return () => clearInterval(timer);
  }, [active, anyRunning]);

  async function upload(files) {
    if (!files || files.length === 0) return;
    setError("");
    setUploading(true);
    try {
      const data = await uploadDocuments(files);
      setDocuments(data.documents);
      refreshJobs();       // la contextualisation automatique démarre côté backend
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function remove(id) {
    setError("");
    try {
      const data = await deleteDocument(id);
      setDocuments(data.documents);
      if (selected?.id === id) setSelected(null);
      refreshJobs();
    } catch (err) {
      setError(err.message);
    }
  }

  async function restart(id) {
    setError("");
    try {
      await restartContextualization(id);
      refreshJobs();
    } catch (err) {
      setError(err.message);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    upload(e.dataTransfer.files);
  }

  return (
    <main className="documents">
      <div
        className={`dropzone${dragging ? " dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            upload(e.target.files);
            e.target.value = "";
          }}
        />
        <div className="icon">📤</div>
        <p>{uploading ? "Chargement…" : "Glissez vos fichiers ici, ou cliquez pour les choisir"}</p>
        <small>Stockés dans PostgreSQL · 20 Mo maximum par fichier · contexte de chaque chunk généré automatiquement</small>
      </div>

      {error && <p className="error-text">⚠️ {error}</p>}

      <h2>Documents chargés ({documents.length})</h2>
      {documents.length > 0 && (
        <p className="hint">Cliquez sur un document pour visualiser son découpage en chunks et leurs contextes.</p>
      )}
      {documents.length === 0 ? (
        <p className="empty-inline">Aucun document pour l'instant.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Nom</th>
              <th>Taille</th>
              <th>Contexte (LLM)</th>
              <th>Ajouté le</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {documents.map((d) => (
              <tr key={d.id}>
                <td className="name">
                  <button type="button" className="doc-link" onClick={() => setSelected(d)} title="Visualiser le chunking">
                    📄 {d.name}
                  </button>
                </td>
                <td>{formatSize(d.size)}</td>
                <td>
                  <ContextStatus job={jobs[d.id]} onRestart={() => restart(d.id)} />
                </td>
                <td>{new Date(d.created_at).toLocaleString("fr-FR")}</td>
                <td className="actions">
                  <a href={downloadUrl(d.id)} title="Télécharger">⬇️</a>
                  <button type="button" className="link" title="Supprimer" onClick={() => remove(d.id)}>
                    🗑️
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selected && <ChunkViewer doc={selected} onClose={() => setSelected(null)} />}
    </main>
  );
}
