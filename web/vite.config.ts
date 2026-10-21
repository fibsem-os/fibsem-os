import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// `bun run dev` proxies the App Server's routes to a server on this machine.
const SERVER = process.env.FIBSEM_SERVER ?? "http://127.0.0.1:8001";
const ROUTES = ["/app", "/microscope", "/services", "/pair", "/pairing", "/capabilities", "/chat"];

export default defineConfig({
  base: "/ui/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  build: {
    outDir: "../fibsem/server/ui",
    emptyOutDir: true,
    // Committed: names without content hashes keep the diff of a rebuild small.
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name].js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  server: { proxy: Object.fromEntries(ROUTES.map((route) => [route, SERVER])) },
});
