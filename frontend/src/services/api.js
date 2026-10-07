const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

export const api = {
  health: () => request("/api/health"),
  latest: () => request("/api/model/latest"),
  history: () => request("/api/model/history"),
  comparison: () => request("/api/model/comparison"),
  weatherHistory: () => request("/api/weather/history"),
  predict: (payload) => request("/api/predict", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  }),
  train: () => request("/api/train", { method: "POST" }),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/upload-dataset", { method: "POST", body: form });
  },
};
