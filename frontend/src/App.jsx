import { useEffect, useState } from "react";
import Chat from "./Chat.jsx";
import Documents from "./Documents.jsx";

const PAGES = { chat: "💬 Chat", documents: "📄 Documents" };

function pageFromHash() {
  const p = window.location.hash.replace("#/", "");
  return p in PAGES ? p : "chat";
}

export default function App() {
  const [page, setPage] = useState(pageFromHash);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <header>
        <span className="brand">🤖 Chatbot</span>
        <nav>
          {Object.entries(PAGES).map(([key, label]) => (
            <a key={key} href={`#/${key}`} className={page === key ? "active" : ""}>
              {label}
            </a>
          ))}
        </nav>
      </header>

      {/* Les deux pages restent montées pour conserver la conversation en cours */}
      <div className="page" hidden={page !== "chat"}>
        <Chat active={page === "chat"} />
      </div>
      <div className="page" hidden={page !== "documents"}>
        <Documents active={page === "documents"} />
      </div>
    </div>
  );
}
