// Affiche un fragment HTML produit par le modèle (graphique Chart.js, SVG, tableau stylé...) dans un cadre ISOLÉ :
// iframe `sandbox="allow-scripts"` (origine opaque : aucun accès à la page, aux cookies ni à l'API) + politique de
// sécurité qui n'autorise que les scripts en ligne et ceux de cdn.jsdelivr.net, et aucune requête réseau.
import { useEffect, useId, useRef, useState } from "react";

const CSP = "default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline'; img-src data: https:; font-src data:;";

function srcdoc(html, id) {
  return `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="${CSP}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>html,body{margin:0;padding:0;background:transparent;font-family:system-ui,-apple-system,sans-serif;color:#111827}
body{padding:4px}img,svg,canvas,table{max-width:100%}</style></head><body>${html}
<script>
  const send = () => parent.postMessage({ htmlFrame: ${JSON.stringify(id)}, height: document.documentElement.scrollHeight }, "*");
  new ResizeObserver(send).observe(document.body);
  addEventListener("load", send); setTimeout(send, 300); setTimeout(send, 1500);
</script></body></html>`;
}

export default function HtmlFrame({ html }) {
  const id = useId();
  const ref = useRef(null);
  const [height, setHeight] = useState(120);

  useEffect(() => {
    const onMessage = (e) => {
      if (e.source === ref.current?.contentWindow && e.data?.htmlFrame === id) {
        setHeight(Math.min(Math.max(40, Math.ceil(e.data.height) + 8), 900));
      }
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [id]);

  return (
    <div className="html-frame">
      <iframe ref={ref} title="Visuel généré par le chatbot" sandbox="allow-scripts" srcDoc={srcdoc(html, id)} style={{ height }} />
      <details>
        <summary>Code HTML</summary>
        <pre><code>{html}</code></pre>
      </details>
    </div>
  );
}
