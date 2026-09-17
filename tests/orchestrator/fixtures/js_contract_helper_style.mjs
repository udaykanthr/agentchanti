import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';

let checks = 0;
function check(condition, message) {
  checks += 1;
  assert.ok(condition, message);
}

const root = process.cwd();
const packagePath = path.join(root, 'package.json');

check(fs.existsSync(packagePath), 'A runnable home page project must provide package.json.');

const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
check(pkg.scripts && typeof pkg.scripts === 'object', 'package.json must define runnable scripts.');

check(typeof pkg.scripts.build === 'string' && pkg.scripts.build.trim().length > 0,
  'The project must provide a build script.');

execFileSync('npm', ['run', 'build'], {
  cwd: root,
  stdio: 'pipe',
  timeout: 120000,
  env: { ...process.env, CI: 'true' }
});
check(true, 'The production build completed successfully.');

const serverScript = ['preview', 'start', 'dev'].find(
  (name) => typeof pkg.scripts[name] === 'string' && pkg.scripts[name].trim().length > 0
);

check(Boolean(serverScript),
  'The project must provide a preview, start, or dev script so the home page can be requested.');

const requestedPort = 41791;
let serverOutput = '';
let server;

function pause(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function requestPage(port, pathname = '/') {
  try {
    return execFileSync(
      'curl',
      [
        '--fail',
        '--silent',
        '--show-error',
        '--location',
        '--compressed',
        '--max-time',
        '3',
        `http://127.0.0.1:${port}${pathname}`
      ],
      { cwd: root, encoding: 'utf8', timeout: 5000 }
    );
  } catch {
    return null;
  }
}

try {
  server = spawn(
    'npm',
    ['run', serverScript, '--', '--host', '127.0.0.1', '--port', String(requestedPort)],
    {
      cwd: root,
      env: {
        ...process.env,
        CI: 'true',
        HOST: '127.0.0.1',
        PORT: String(requestedPort)
      },
      stdio: ['ignore', 'pipe', 'pipe']
    }
  );

  server.stdout.on('data', (chunk) => { serverOutput += chunk.toString(); });
  server.stderr.on('data', (chunk) => { serverOutput += chunk.toString(); });

  let html = null;
  let activePort = null;

  for (let attempt = 0; attempt < 30 && !html; attempt += 1) {
    const loggedPorts = [...serverOutput.matchAll(/https?:\/\/[^\s:]+:(\d+)/gi)]
      .map((match) => Number(match[1]));

    const ports = [...new Set([
      requestedPort,
      ...loggedPorts,
      4173,
      5173,
      3000,
      8080
    ])];

    for (const port of ports) {
      const page = requestPage(port);
      if (page) {
        html = page;
        activePort = port;
        break;
      }
    }

    if (!html) await pause(500);
  }

  check(Boolean(html),
    `The ${serverScript} server must start and serve the home page over HTTP. Server output: ${serverOutput.slice(0, 500)}`);

  let rendered = html;
  const browserCandidates = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
  let browserPath = null;

  for (const candidate of browserCandidates) {
    try {
      browserPath = execFileSync('which', [candidate], { encoding: 'utf8', timeout: 2000 }).trim();
      if (browserPath) break;
    } catch {
      // A browser is optional; the server response remains testable without one.
    }
  }

  if (browserPath) {
    try {
      const dom = execFileSync(
        browserPath,
        [
          '--headless',
          '--no-sandbox',
          '--disable-gpu',
          '--dump-dom',
          '--virtual-time-budget=5000',
          `http://127.0.0.1:${activePort}/`
        ],
        { cwd: root, encoding: 'utf8', timeout: 15000, stdio: ['ignore', 'pipe', 'pipe'] }
      );
      if (dom && dom.trim().length > 0) rendered = dom;
    } catch {
      // Some environments intentionally do not permit headless browser execution.
    }
  }

  check(/<title[^>]*>\s*\S[\s\S]*?<\/title>/i.test(rendered),
    'The home page must expose a meaningful document title.');

  check(/<meta[^>]+name=["']viewport["'][^>]*content=["'][^"']*width\s*=\s*device-width/i.test(rendered),
    'The home page must declare a mobile viewport.');

  check(/<(header|nav)\b[\s\S]*?<\/(header|nav)>/i.test(rendered),
    'The home page must provide visible site navigation.');

  check(/<main\b[\s\S]*?<\/main>/i.test(rendered),
    'The home page must provide a main content landmark.');

  check(/<h1\b[^>]*>\s*\S[\s\S]*?<\/h1>/i.test(rendered),
    'The home page must present a primary headline.');

  check(/<(a|button)\b[^>]*>[\s\S]*?(get started|learn more|contact|explore|shop|sign up|book|view|discover)[\s\S]*?<\/(a|button)>/i.test(rendered),
    'The home page must include a clear call to action.');

  let styleText = rendered;
  const stylesheetHrefs = [...rendered.matchAll(/<link[^>]+rel=["'][^"']*stylesheet[^"']*["'][^>]+href=["']([^"']+)["']/gi)]
    .map((match) => match[1]);

  for (const href of stylesheetHrefs) {
    if (/^(https?:|\/\/)/i.test(href)) continue;
    const cssPath = href.startsWith('/') ? href : `/${href.replace(/^\.\//, '')}`;
    const css = requestPage(activePort, cssPath);
    if (css) styleText += `\n${css}`;
  }

  check(/@media\s*(?:only\s*)?\([^)]*(max-width|min-width|width)[^)]*\)/i.test(styleText),
    'The delivered page must include responsive breakpoint styling for different screen widths.');
} finally {
  if (server && !server.killed) {
    server.kill('SIGTERM');
    await pause(200);
    if (!server.killed) server.kill('SIGKILL');
  }
}

console.log(`ACCEPTANCE: ${checks} checks passed`);