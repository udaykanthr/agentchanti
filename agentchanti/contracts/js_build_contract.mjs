// A deterministic acceptance check for a JavaScript or TypeScript project.
//
// WHY THIS FILE EXISTS
//
// The seeder can ask a model to write a contract before any code exists,
// and for Python that works. For a JS/TS build it did not. Measured over
// four consecutive live runs on one Next.js task, each contract failed a
// project that was correct, and each for a different reason:
//
//   1. spawnSync("npm", ...)                  ENOENT — npm is a .cmd
//   2. "project root must contain package.json" — the app was in my-app/
//   3. "the development server did not serve the home page"
//   4. it read .next/page.js instead of .next/server/app/index.html
//
// Every one of those is a way of OBSERVING a build wrongly, on a machine
// the author never saw, guessing at a layout that did not exist yet.
// Screens now catch (1)-(3); (4) showed the space is larger than a prompt
// can enumerate.
//
// So this file is written once, by people who can run it, and shipped.
// It is still independent of the run — nothing here is authored by the
// model whose work it judges, and the hash check still refuses it as
// evidence if the run edits it.
//
// WHAT IT PROVES, AND WHAT IT DOES NOT
//
// That the project builds and emits a non-trivial page for its root
// route. It knows nothing about the task, so it cannot tell a beautiful
// home page from an ugly one — the run reports it as SHALLOW for exactly
// that reason. It is a floor, not a ceiling: it catches a project that
// does not build, emits nothing, or emits an empty shell, which is the
// class of failure that matters most and the one a run can otherwise
// declare "complete".

import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const SKIP = new Set([
  "node_modules", ".git", ".agentchanti", ".next", "dist", "build", "out",
  ".turbo", "coverage",
]);

let checks = 0;
function check(condition, message) {
  checks += 1;
  assert.ok(condition, message);
}

// ── 1. Find the app. It is often a subdirectory: `my-app/package.json`.
function findApp(root) {
  const candidates = [];
  if (fs.existsSync(path.join(root, "package.json"))) candidates.push(root);
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory() || SKIP.has(entry.name)) continue;
    const manifest = path.join(root, entry.name, "package.json");
    if (fs.existsSync(manifest)) candidates.push(path.join(root, entry.name));
  }
  for (const dir of candidates) {
    try {
      const pkg = JSON.parse(
        fs.readFileSync(path.join(dir, "package.json"), "utf8"),
      );
      if (pkg.scripts && typeof pkg.scripts.build === "string") {
        return { dir, pkg };
      }
    } catch {
      // an unreadable manifest is not the app
    }
  }
  return null;
}

const app = findApp(ROOT);
check(app !== null, `no package.json defining a build script was found under ${ROOT}`);

// ── 2. It builds. `shell: true` because npm is a .cmd on Windows.
let buildOutput = "";
try {
  buildOutput = execSync("npm run build", {
    cwd: app.dir,
    encoding: "utf8",
    timeout: 600000,
    shell: true,
    stdio: "pipe",
  });
} catch (error) {
  const detail = `${error.stdout || ""}${error.stderr || ""}`.slice(-1500);
  check(false, `the production build failed:\n${detail}`);
}
check(
  !/failed to compile|build failed/i.test(buildOutput),
  "the build reported a compilation failure",
);

// ── 3. It emitted something to serve. Every plausible location, because
//      the layout differs per framework and per version — the mistake
//      that made a model-written contract read `.next/page.js`.
function collectHtml(dir, depth = 0) {
  if (depth > 4 || !fs.existsSync(dir)) return [];
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...collectHtml(full, depth + 1));
    } else if (entry.name.endsWith(".html")) {
      out.push(full);
    }
  }
  return out;
}

const outputDirs = [".next/server/app", ".next/server/pages", "out", "dist",
                    "build", "public"].map((d) => path.join(app.dir, d));
let html = [];
for (const dir of outputDirs) html.push(...collectHtml(dir));
// An error page is not the site. `not-found` matters as much as `404`:
// measured, a page that rendered `null` PASSED this contract because
// Next's 8.5KB `_not-found.html` was larger than the 6.4KB `index.html`
// and "the largest page" picked it. Grading the error page is how a
// check manufactures its own proof.
html = html.filter(
  (f) => !/error|404|500|not[-_]?found/i.test(path.basename(f)),
);
check(html.length > 0, "the build emitted no HTML to serve");

// ── 4. The emitted page is a real page, not an empty shell.
//      The ROOT route is what the task is about, so prefer it by name and
//      fall back to the largest only when nothing is named like one.
const ROOT_NAMES = ["index.html", "page.html", "home.html"];
const rootFirst = [
  ...html.filter((f) => ROOT_NAMES.includes(path.basename(f).toLowerCase())),
  ...html.filter((f) => !ROOT_NAMES.includes(path.basename(f).toLowerCase())),
];
const rendered = (() => {
  for (const f of rootFirst) {
    try {
      return fs.readFileSync(f, "utf8");
    } catch {
      // unreadable: try the next candidate
    }
  }
  return "";
})();

check(rendered.length > 500,
      `the largest emitted page is ${rendered.length} bytes — too small to be a page`);
check(/<body[\s>]/i.test(rendered), "the emitted page has no <body>");
check(
  /<(h1|main|header|section|article|nav)[\s>]/i.test(rendered),
  "the emitted page has no content regions (h1/main/header/section/article/nav)",
);

const text = rendered
  .replace(/<script[\s\S]*?<\/script>/gi, " ")
  .replace(/<style[\s\S]*?<\/style>/gi, " ")
  .replace(/<[^>]+>/g, " ")
  .replace(/\s+/g, " ")
  .trim();
check(text.length > 50,
      `the emitted page renders only ${text.length} characters of text`);

console.log(`ACCEPTANCE: ${checks} checks passed (app: ${path.relative(ROOT, app.dir) || "."})`);
