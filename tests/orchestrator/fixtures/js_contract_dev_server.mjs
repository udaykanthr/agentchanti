import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execSync, spawn } from 'node:child_process';

const root = process.cwd();
let checks = 0;

function check(condition, message) {
  assert.ok(condition, message);
  checks += 1;
}

function run(command, args, options = {}) {
  return execSync([command, ...args].join(' '), {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    timeout: 120000,
    ...options,
  });
}

function findIndexFiles(directory, depth = 0) {
  if (depth > 5 || !fs.existsSync(directory)) return [];

  const ignored = new Set([
    'node_modules',
    '.git',
    '.next',
    '.cache',
    'coverage',
  ]);

  const results = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (ignored.has(entry.name)) continue;

    const fullPath = path.join(directory, entry.name);
    if (entry.isFile() && entry.name.toLowerCase() === 'index.html') {
      results.push(fullPath);
    } else if (entry.isDirectory()) {
      results.push(...findIndexFiles(fullPath, depth + 1));
    }
  }
  return results;
}

function readBestIndex() {
  const preferredDirectories = ['dist', 'build', 'out', 'public', '.next'];
  const candidates = [];

  for (const directory of preferredDirectories) {
    candidates.push(...findIndexFiles(path.join(root, directory)));
  }

  candidates.push(...findIndexFiles(root));

  const unique = [...new Set(candidates)].filter((file) => {
    return !file.split(path.sep).includes('node_modules');
  });

  unique.sort((a, b) => {
    const preferred = (file) => {
      const relative = path.relative(root, file);
      return preferredDirectories.some(
        (directory) =>
          relative === path.join(directory, 'index.html') ||
          relative.startsWith(`${directory}${path.sep}`)
      )
        ? 1
        : 0;
    };

    return preferred(b) - preferred(a) || fs.statSync(b).size - fs.statSync(a).size;
  });

  return unique.length
    ? { file: unique[0], html: fs.readFileSync(unique[0], 'utf8') }
    : null;
}

function curlPage(port) {
  try {
    const output = execSync(
      `curl -sS -L --max-time 2 -w "\\n%{http_code}" http://127.0.0.1:${port}/`,
      {
        cwd: root,
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
        timeout: 5000,
      }
    );

    const lastNewline = output.lastIndexOf('\n');
    const body = output.slice(0, lastNewline);
    const status = Number(output.slice(lastNewline + 1).trim());

    if (status >= 200 && status < 400 && body.trim()) {
      return body;
    }
  } catch {
    // The server may not be ready yet; the caller retries.
  }

  return null;
}

const packagePath = path.join(root, 'package.json');
let packageJson = null;

if (fs.existsSync(packagePath)) {
  try {
    packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
    check(
      packageJson && typeof packageJson === 'object',
      'package.json must contain a valid project manifest'
    );
  } catch {
    check(false, 'package.json must be valid JSON');
  }
}

const scripts = packageJson && typeof packageJson.scripts === 'object'
  ? packageJson.scripts
  : {};

if (typeof scripts.build === 'string') {
  let buildSucceeded = false;
  try {
    run('npm', ['run', 'build']);
    buildSucceeded = true;
  } catch {
    buildSucceeded = false;
  }

  check(buildSucceeded, 'the production build must complete successfully');
}

let servedHtml = null;
let server = null;

const serverScript = ['preview', 'start', 'dev'].find(
  (name) => typeof scripts[name] === 'string'
);

try {
  if (serverScript) {
    const port = 43123;
    server = spawn(
      'npm',
      ['run', serverScript, '--', '--port', String(port)],
      {
        cwd: root,
        detached: process.platform !== 'win32',
        env: { ...process.env, PORT: String(port) },
        stdio: 'ignore',
      }
    );

    const deadline = Date.now() + 20000;
    const ports = [port, 3000, 4173, 5173, 8080, 8000];

    while (Date.now() < deadline && !servedHtml) {
      for (const candidatePort of ports) {
        servedHtml = curlPage(candidatePort);
        if (servedHtml) break;
      }
    }

    check(
      Boolean(servedHtml),
      'the home page must be reachable from the application server'
    );
  } else {
    const localIndex = readBestIndex();
    check(
      Boolean(localIndex),
      'the project must provide a home page even when no server script is configured'
    );
    servedHtml = localIndex.html;
  }
} finally {
  if (server && !server.killed) {
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /pid ${server.pid} /T /F`, {
          stdio: 'ignore',
          timeout: 5000,
        });
      } else {
        process.kill(-server.pid, 'SIGTERM');
      }
    } catch {
      try {
        server.kill('SIGTERM');
      } catch {
        // Nothing further is required if the process already exited.
      }
    }
  }
}

const html = String(servedHtml || '');

check(
  /<!doctype\s+html|<html[\s>]/i.test(html),
  'the home page response must be an HTML document'
);

check(
  /<meta\b[^>]*\bname\s*=\s*["']viewport["'][^>]*\bcontent\s*=\s*["'][^"']*width\s*=\s*device-width/i.test(
    html
  ),
  'the home page must declare a device-width viewport for responsive layouts'
);

const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
const title = titleMatch ? titleMatch[1].replace(/<[^>]+>/g, '').trim() : '';

check(
  title.length >= 3 &&
    !/^(vite|react|next\.js|app|application|website)$/i.test(title),
  'the home page must provide a descriptive, project-specific document title'
);

check(
  /<(main|header|nav|section|article|h1)\b/i.test(html) ||
    /<(div|body)\b[^>]*(id|class)=["'][^"']*(root|app|home|landing)[^"']*["']/i.test(html),
  'the home page must expose a page or application content region'
);

check(
  /<link\b[^>]*\brel\s*=\s*["'][^"']*stylesheet[^"']*["'][^>]*>|<style\b/i.test(html) ||
    /<(script)\b[^>]*\bsrc\s*=/i.test(html),
  'the home page must load presentation or application assets rather than being an unstyled document'
);

console.log(`ACCEPTANCE: ${checks} checks passed`);