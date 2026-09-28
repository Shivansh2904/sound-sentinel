// Links back to this repository must point at the account that owns it.
// Run with: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const REPO_URL = "https://github.com/Shivansh2904/sound-sentinel";
const REPO_LINK = /https?:\/\/github\.com\/[\w.-]+\/sound-sentinel/g;

function repoLinks(file) {
  return readFileSync(new URL(`../${file}`, import.meta.url), "utf8").match(REPO_LINK) ?? [];
}

test("the app header links to this repository", () => {
  const links = repoLinks("src/App.tsx");
  assert.ok(links.length > 0, "src/App.tsx has no link to the repository");
  for (const link of links) assert.equal(link, REPO_URL);
});

test("the README links to this repository", () => {
  const links = repoLinks("README.md");
  assert.ok(links.length > 0, "README.md has no link to the repository");
  for (const link of links) assert.equal(link, REPO_URL);
});
