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

// A build script alone does not make a web app. Measured 2026-09-27: a
// command-line todo manager declared
//
//     "build": "node -e \"console.log('No build step required for the CLI')\""
//
// so "has a build script" matched it and the page checks failed a correct
// program. A web build is a build script AND something that says the
// output is a page: a framework or bundler dependency, or an index.html
// to build from.
const WEB_DEPS = /^(next|nuxt|astro|gatsby|vite|parcel|webpack|react|react-dom|react-scripts|vue|svelte|@sveltejs\/kit|@angular\/core|remix|@remix-run\/|preact|solid-js|qwik)/;

function isWebProject(dir, pkg) {
  const deps = Object.keys({ ...(pkg.dependencies || {}),
                             ...(pkg.devDependencies || {}) });
  if (deps.some((d) => WEB_DEPS.test(d))) return true;
  for (const html of ["index.html", "public/index.html", "src/index.html",
                      "app/index.html"]) {
    if (fs.existsSync(path.join(dir, html))) return true;
  }
  return false;
}

// The floor for a project with no build step. Deliberately thin: this
// file is generic and knows nothing about the task, so it checks the
// things true of EVERY runnable Node project and nothing more. A thin
// check that is right about a correct project beats a strong one that
// fails it - the trade `desktop_introspection_reason` already makes.
function runNonWebContract(app) {
  const pkg = app.pkg || {};
  const named = [];
  if (typeof pkg.main === "string") named.push(pkg.main);
  if (typeof pkg.bin === "string") named.push(pkg.bin);
  else if (pkg.bin && typeof pkg.bin === "object") named.push(...Object.values(pkg.bin));
  for (const fallback of ["index.js", "index.mjs", "index.cjs", "main.js",
                          "cli.js", "src/index.js"]) {
    named.push(fallback);
  }
  const entry = named
    .filter((p) => typeof p === "string" && p)
    .map((p) => path.join(app.dir, p))
    .find((p) => fs.existsSync(p) && fs.statSync(p).isFile());

  check(entry !== undefined,
        `no entry point exists: package.json names ${JSON.stringify(
          pkg.main || pkg.bin || null)} and no index.js was found in ${app.dir}`);

  // Parses as JavaScript. `node --check` runs nothing, so a CLI with side
  // effects is never executed by this contract.
  try {
    execSync(`node --check ${JSON.stringify(entry)}`,
             { encoding: "utf8", timeout: 60000, shell: true, stdio: "pipe" });
  } catch (error) {
    const detail = `${error.stdout || ""}${error.stderr || ""}`.slice(-600);
    check(false, `the entry point is not valid JavaScript:\n${detail}`);
  }
  check(true, "entry point parses");

  const body = fs.readFileSync(entry, "utf8");
  check(body.trim().length > 200,
        `the entry point is ${body.trim().length} bytes — too small to be a program`);
  check(/function|=>|class|require\(|import\s/.test(body),
        "the entry point declares no functions, imports or requires");

  console.log(`ACCEPTANCE: ${checks} checks passed (no build script — ` +
              `judged as a CLI/library at ${path.relative(ROOT, app.dir) || "."})`);
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
      if (pkg.scripts && typeof pkg.scripts.build === "string"
          && isWebProject(dir, pkg)) {
        return { dir, pkg };
      }
    } catch {
      // an unreadable manifest is not the app
    }
  }
  return null;
}

// A project with NO build script is not a broken web app - it is a CLI or
// a library, and this contract's page checks cannot apply to it.
//
// Measured 2026-09-27 on the `todo-node` benchmark case: the seeder
// installs this file for ANY JavaScript project, so a command-line todo
// manager was judged by "the project builds and emits a real page". It has
// no build and no page, so the contract could never pass, all three runs
// reported `self-authored` where the identical task in Python reported
// `independent`, and the artifacts passed an external 11-step behavioural
// probe. An instrument that cannot apply must say so, not fail.
function findAnyPackage(root) {
  const candidates = [];
  if (fs.existsSync(path.join(root, "package.json"))) candidates.push(root);
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory() || SKIP.has(entry.name)) continue;
    const manifest = path.join(root, entry.name, "package.json");
    if (fs.existsSync(manifest)) candidates.push(path.join(root, entry.name));
  }
  for (const dir of candidates) {
    try {
      return { dir, pkg: JSON.parse(
        fs.readFileSync(path.join(dir, "package.json"), "utf8")) };
    } catch {
      // an unreadable manifest is not the app
    }
  }
  return null;
}

const app = findApp(ROOT);
if (app === null) {
  // No manifest at all is a legitimate shape for a small Node CLI - the
  // task asked for `node index.js`, not for a package.json - so the entry
  // point is judged directly. A directory with neither still fails, in
  // runNonWebContract, naming what is missing.
  runNonWebContract(findAnyPackage(ROOT) || { dir: ROOT, pkg: {} });
  process.exit(0);
}

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
let renderedFile = "";
const rendered = (() => {
  for (const f of rootFirst) {
    try {
      const body = fs.readFileSync(f, "utf8");
      renderedFile = f;
      return body;
    } catch {
      // unreadable: try the next candidate
    }
  }
  return "";
})();

check(/<body[\s>]/i.test(rendered), "the emitted page has no <body>");

const text = rendered
  .replace(/<script[\s\S]*?<\/script>/gi, " ")
  .replace(/<style[\s\S]*?<\/style>/gi, " ")
  .replace(/<[^>]+>/g, " ")
  .replace(/\s+/g, " ")
  .trim();

// ── 4b. Where the page's content LIVES depends on the rendering strategy,
//      and grading only the emitted HTML judged that choice rather than the
//      app. Next prerenders, so its HTML carries the copy; a Vite/CRA SPA
//      emits `<div id="root"></div>` and renders in the browser.
//
//      Measured 2026-09-25 across three runs that each produced a working
//      responsive home page: 463 bytes failed "too small to be a page",
//      637 bytes failed "no content regions", and the third PASSED only
//      because the model happened to add a <noscript> fallback — markup
//      that renders exactly when JavaScript does NOT. The verdict was
//      uncorrelated with whether the app worked.
//
//      So: if the page renders server-side, judge the page. If it is a
//      shell that loads a bundle, judge that bundle's copy. A shell that
//      loads nothing, or a bundle with no human text in it, still fails.
function bundleCopy(htmlFile, htmlText) {
  if (!htmlFile) return 0;
  const base = path.dirname(htmlFile);
  const srcs = [...htmlText.matchAll(/<script[^>]+src=["']([^"']+)["']/gi)]
    .map((m) => m[1])
    .filter((s) => !/^https?:/i.test(s));
  let total = 0;
  for (const src of srcs) {
    const candidates = [
      path.join(base, src.replace(/^\//, "")),
      path.join(base, src),
      path.join(app.dir, src.replace(/^\//, "")),
    ];
    for (const cand of candidates) {
      let body;
      try {
        body = fs.readFileSync(cand, "utf8");
      } catch {
        continue;
      }
      // Literal strings of three or more words: headings, copy, labels —
      // never minified identifiers.
      for (const m of body.matchAll(/["'`]([^"'`<>{}\\]{12,200})["'`]/g)) {
        if (m[1].trim().split(/\s+/).length >= 3) total += m[1].length;
      }
      break;
    }
  }
  return total;
}

const serverRendered =
  text.length > 50 &&
  /<(h1|main|header|section|article|nav)[\s>]/i.test(rendered);

if (serverRendered) {
  check(rendered.length > 500,
        `the largest emitted page is ${rendered.length} bytes — too small to be a page`);
  check(text.length > 50,
        `the emitted page renders only ${text.length} characters of text`);
} else {
  const copy = bundleCopy(renderedFile, rendered);
  check(
    copy > 200,
    `the emitted page is a shell (${text.length} chars of text, no content ` +
    `regions) and the scripts it loads carry only ${copy} characters of ` +
    `copy — so nothing renders it either server-side or client-side`,
  );
}

console.log(`ACCEPTANCE: ${checks} checks passed (app: ${path.relative(ROOT, app.dir) || "."})`);
