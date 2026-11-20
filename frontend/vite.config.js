import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Les appels /api/... sont redirigés vers le backend (service "backend" de docker-compose)
    proxy: {
      "/api": { target: process.env.BACKEND_URL || "http://backend:8000", changeOrigin: true },
    },
  },
});
