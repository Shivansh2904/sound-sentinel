import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

/**
 * Vite Configuration for SoundSentinel
 * ======================================
 *
 * Key requirements:
 *
 * 1. SharedArrayBuffer support — ONNX Runtime Web's multi-threaded WASM backend
 *    requires SharedArrayBuffer, which is only available in cross-origin isolated
 *    contexts. We set the required COOP/COEP headers on the dev and preview
 *    servers below. A static host has to send the same headers itself.
 *
 * 2. Web Worker — The inference worker is bundled as a separate ES module chunk
 *    using Vite's built-in worker support (type: 'module').
 *
 * 3. ONNX Runtime Web's WASM files — `vite build` bundles onnxruntime-web into
 *    the worker chunk and emits its .wasm binary into dist/assets/ under a
 *    hashed name. The worker then sets `ort.env.wasm.wasmPaths = "/"`, which makes
 *    ONNX Runtime Web load its .mjs and .wasm files from the site root instead,
 *    and nothing puts them there yet (there is no public/ directory). Files in
 *    public/, such as public/model.onnx once it exists, are copied to the build
 *    output root unchanged.
 */
export default defineConfig({
  plugins: [
    react(),
  ],

  // Resolve aliases
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },

  // Web Worker configuration
  worker: {
    // Bundle workers as ES modules so they can use import syntax
    format: "es",
    plugins: () => [react()],
  },

  // Dev server
  server: {
    port: 5173,
    headers: {
      // Required for SharedArrayBuffer (used by ONNX Runtime Web multi-threaded WASM)
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },

  // Preview server (for `npm run preview`)
  preview: {
    port: 4173,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },

  // Build configuration
  build: {
    target: "es2020",
    outDir: "dist",
    sourcemap: true,

    rollupOptions: {
      output: {
        // Split vendor chunks for better caching
        manualChunks: {
          "react-vendor": ["react", "react-dom"],
          "ort": ["onnxruntime-web"],
        },
      },
    },
  },

  // Optimize dependencies
  optimizeDeps: {
    // onnxruntime-web uses dynamic imports internally — exclude from pre-bundling
    exclude: ["onnxruntime-web"],
  },
});
