import { useEffect, useRef, useState } from "react";
import { deleteDocument, formatSize, getDocuments, getModels, sendChat, uploadDocuments } from "./api.js";
import Markdown from "./Markdown.jsx";
import EffortBar from "./EffortBar.jsx";

export default function Chat({ active }) {
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const bottomRef = useRef(null);
  const fileInputRef = useRef(null);
  const textRef = useRef(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [levels, setLevels] = useState(null);      // niveaux de raisonnement « maison » (backend/models.py)
  const [effort, setEffort] = useState(null);
  const native = models.find((m) => m.id === model)?.native_reasoning || null;   // niveau -> reasoning_effort de l'API

  function chooseEffort(level) {
    setEffort(level);
    try {
      localStorage.setItem("chat-reasoning", level);
    } catch {
      /* stockage indisponible */
    }
  }

  // Modèles proposés (backend/models.py) ; le dernier choisi est retenu dans le navigateur
  useEffect(() => {
    getModels()
      .then((data) => {
        setModels(data.models);
        setLevels(data.reasoning);
        let savedLevel = null;
        try {
          savedLevel = localStorage.getItem("chat-reasoning");
        } catch {
          /* stockage indisponible */
        }
        setEffort(data.reasoning.levels.includes(savedLevel) ? savedLevel : data.reasoning.default);
        let saved = null;
        try {
          saved = localStorage.getItem("chat-model");
        } catch {
          /* stockage indisponible */
        }
        setModel(data.models.some((m) => m.id === saved) ? saved : data.default);
      })
      .catch(() => {});
  }, []);

  function chooseModel(id) {
    setModel(id);
    try {
      localStorage.setItem("chat-model", id);
    } catch {
      /* stockage indisponible */
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // La zone de saisie grandit avec le texte (jusqu'à une hauteur maximale, puis elle défile)
  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  // Recharge la liste des documents à chaque retour sur le chat (ils ont pu changer dans la page Documents)
  useEffect(() => {
    if (active) getDocuments().then(setDocuments).catch(() => {});
  }, [active]);

  function addError(text) {
    setMessages((prev) => [...prev, { role: "assistant", content: `⚠️ ${text}`, error: true }]);
  }

  async function onFilesChosen(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    try {
      const data = await uploadDocuments(files);
      setDocuments(data.documents);
    } catch (err) {
      addError(err.message);
    } finally {
      e.target.value = ""; // permet de re-sélectionner le même fichier
      setUploading(false);
    }
  }

  async function removeDocument(id) {
    try {
      const data = await deleteDocument(id);
      setDocuments(data.documents);
    } catch (err) {
      addError(err.message);
    }
  }

  async function send(e) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const history = [...messages, { role: "user", content: text }];
    setMessages(history);
    setInput("");
    setLoading(true);

    try {
      const data = await sendChat(history.filter((m) => !m.error).map(({ role, content }) => ({ role, content })), model, effort);
      setMessages([...history, { role: "assistant", content: data.reply, recipes: data.recipes || [], servings: data.servings || null, thinking: data.thinking || null }]);
    } catch (err) {
      setMessages([...history, { role: "assistant", content: `⚠️ ${err.message}`, error: true }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <main>
        {messages.length === 0 && <p className="empty">Écrivez un message pour commencer.</p>}
        {messages.map((m, i) => (
          <div key={i} className={`turn ${m.role}`}>
            {m.thinking && (
              <details className="thinking">
                <summary>💭 Réflexion du modèle</summary>
                <div>{m.thinking}</div>
              </details>
            )}
            <div className={`bubble ${m.role}${m.error ? " error" : ""}${m.role === "assistant" && !m.error ? " rich" : ""}`}>
              {m.role === "assistant" && !m.error ? <Markdown text={m.content} recipes={m.recipes || []} servings={m.servings} /> : m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="bubble assistant typing" role="status" aria-label="Le chatbot écrit…">
            <span /><span /><span />
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      {(documents.length > 0 || uploading) && (
        <div className="docs">
          {documents.map((d) => (
            <span key={d.id} className="chip" title={formatSize(d.size)}>
              📄 {d.name}
              <button type="button" aria-label={`Retirer ${d.name}`} onClick={() => removeDocument(d.id)}>
                ×
              </button>
            </span>
          ))}
          {uploading && <span className="chip loading">Chargement…</span>}
        </div>
      )}

      <form onSubmit={send}>
        <input ref={fileInputRef} type="file" multiple hidden onChange={onFilesChosen} />
        <button
          type="button"
          className="plus"
          title="Ajouter des documents"
          aria-label="Ajouter des documents"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          +
        </button>
        <textarea
          ref={textRef}
          className="text"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // Entrée envoie ; Maj+Entrée passe à la ligne
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) send(e);
          }}
          placeholder="Votre message… (Maj+Entrée pour aller à la ligne)"
        />
        {models.length > 0 && (
          <select className="model" value={model} onChange={(e) => chooseModel(e.target.value)} aria-label="Modèle" title="Modèle utilisé pour répondre">
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        )}
        {levels && effort && (
          <EffortBar
            levels={levels}
            value={effort}
            onChange={chooseEffort}
            hint={`Jusqu'à ${levels.agent_steps[effort]} étapes d'agent` +
              (native && native[effort] !== "none"
                ? ` ; réflexion du modèle (${native[effort]}, ${levels.budget[effort]} jetons max)`
                : native ? " ; sans réflexion du modèle à ce niveau" : " ; ce modèle n'a pas de réflexion native")}
          />
        )}
        <button type="submit" disabled={loading || !input.trim()}>
          Envoyer
        </button>
      </form>
    </>
  );
}
