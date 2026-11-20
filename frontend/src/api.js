async function request(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail || "";
    } catch {
      /* pas de JSON */
    }
    throw new Error(detail || `Erreur ${res.status}`);
  }
  return res.json();
}

export const getDocuments = () => request("/api/documents");

export function uploadDocuments(files) {
  const form = new FormData();
  Array.from(files).forEach((f) => form.append("files", f));
  return request("/api/documents", { method: "POST", body: form });
}

export const deleteDocument = (id) => request(`/api/documents/${id}`, { method: "DELETE" });

export const downloadUrl = (id) => `/api/documents/${id}/download`;

export const getContextualization = () => request("/api/contextualization");

export const restartContextualization = (id) => request(`/api/documents/${id}/contextualize`, { method: "POST" });

export const getContexts = (id, chunkSize, overlap) =>
  request(`/api/documents/${id}/contexts?chunk_size=${chunkSize}&overlap=${overlap}`);

export const getEmbeddingStatus = () => request("/api/embedding");

export const startEmbedding = (id, force = false) =>
  request(`/api/documents/${id}/embed?force=${force}`, { method: "POST" });

export const getEmbedded = (id, chunkSize, overlap) =>
  request(`/api/documents/${id}/embedded?chunk_size=${chunkSize}&overlap=${overlap}`);

export const generateContext =(id, index, chunkSize, overlap) =>
  request(`/api/documents/${id}/chunks/${index}/context?chunk_size=${chunkSize}&overlap=${overlap}`, { method: "POST" });

export const getChunks = (id, chunkSize, overlap) =>
  request(`/api/documents/${id}/chunks?chunk_size=${chunkSize}&overlap=${overlap}`);

export const sendChat = (messages, model, reasoning) =>
  request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, model: model || null, reasoning: reasoning || null }),
  });

export const getModels = () => request("/api/models");

export function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

export const getRecipe = (docId, chunkIndex, servings) =>
  request(`/api/recipes/${docId}/${chunkIndex}${servings ? `?servings=${servings}` : ""}`);

export const recipeImageUrl = (docId, chunkIndex) => `/api/recipes/${docId}/${chunkIndex}/image`;
