// Browsers only honour COOP/COEP as HTTP response headers. As <meta http-equiv>
// tags they do nothing, so index.html must not rely on them.
// Run with: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

test("index.html does not set COOP/COEP through meta tags", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /<meta[^>]*http-equiv=["']?Cross-Origin-(Opener|Embedder)-Policy/i);
});
