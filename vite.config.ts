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
 *    contexts. We set the required COOP/COEP headers in the dev server.
 *
 * 2. Web Worker — The inference worker is bundled as a separate ES module chunk
 *    using Vite's built-in worker support (type: 'module').
 *
 * 3. ONNX WASM files — onnxruntime-web ships WASM binaries that must be served
 *    from the public directory. Vite automatically copies everything in /public
 *    to the build output root, so placing model.onnx there is sufficient.
 *    The WASM files from onnxruntime-web are referenced as /ort-wasm*.wasm URLs,
 *    so we exclude them from the bundle and let the CDN/public dir serve them.
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
