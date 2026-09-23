"""
Project scanner — reads existing project structure and key files
so the planner agent has awareness of the current codebase.
"""

import os
import subprocess

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv", "env",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    "target", "bin", "obj", ".idea", ".vscode", ".eggs",
    "site-packages", ".next", ".nuxt", "coverage", "htmlcov",
    ".agentchanti", "logs", ".claude",
}

SKIP_FILES = {
    ".agentchanti.yaml", ".agentchanti_checkpoint.json", ".agentchanti.yml",
    "prompt.txt", "CLAUDE.md",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib", ".o", ".obj",
    ".class", ".jar", ".war", ".zip", ".tar", ".gz", ".bz2",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".bmp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".woff", ".woff2", ".ttf", ".eot",
    ".lock", ".db", ".sqlite", ".sqlite3",
}

# Extensions recognized as source code files worth pre-loading into memory
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm", ".css", ".scss",
    ".sass", ".less", ".vue", ".svelte", ".php", ".rb", ".go", ".rs",
    ".java", ".kt", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".m",
    ".lua", ".sh", ".bash", ".zsh", ".sql", ".graphql", ".yaml", ".yml",
    ".toml", ".json", ".xml", ".md", ".txt", ".env.example",
}

_MAX_SOURCE_FILES = 50
_MAX_FILE_SIZE_BYTES = 32_000
_MAX_TOTAL_CHARS = 200_000

KEY_FILENAMES = {
    "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
    "package.json", "tsconfig.json",
    "Cargo.toml", "go.mod", "go.sum",
    "Gemfile", "pom.xml", "build.gradle",
    "Makefile", "Dockerfile", "docker-compose.yml",
    "main.py", "app.py", "index.js", "index.ts",
    "manage.py", "settings.py",
    ".env.example", "README.md",
}

_MAX_KEY_FILES = 15
_MAX_LINES_PER_FILE = 100


def _get_git_tracked_files(directory: str) -> set[str] | None:
    """Files git considers part of the project — tracked AND untracked.

    Two questions, and only their union describes a working tree: `git
    ls-files` lists what is committed, and `git ls-files --others
    --exclude-standard` lists what exists and is *not* ignored. A project
    the user created this morning is entirely in the second set. See
    :func:`_is_gitignored` for the run this cost.

    Returns None if not a git repo or git is unavailable, so callers
    can fall back to the hardcoded skip lists.
    """
    visible: set[str] = set()
    saw_git = False
    for args in (["git", "ls-files"],
                 ["git", "ls-files", "--others", "--exclude-standard"]):
        try:
            result = subprocess.run(
                args, cwd=directory, capture_output=True, text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            continue
        saw_git = True
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                visible.add(line.replace("\\", "/"))
    if not saw_git:
        return None
    visible = _expand_nested_repos(directory, visible)
    return visible if visible else None


# Build output and dependency trees are never "the project" — the same
# directories every scan already skips, needed here only for a nested
# repository whose own ignore rules could not be asked.
_NESTED_SKIP_DIRS = {".git", "node_modules", ".next", "dist", "build",
                     "__pycache__", "venv", ".venv", ".turbo", "coverage"}


def _expand_nested_repos(directory: str, visible: set[str]) -> set[str]:
    """Replace every entry that is a DIRECTORY with the files inside it.

    `git ls-files` lists files, with one exception: a nested repository.
    Committed, it is a gitlink — a single entry naming the directory
    (`160000 ... my-app`); untracked, `--others` reports it as `my-app/`.
    Either way git stops at its boundary and says nothing of the files.

    Measured 2026-09-21. `create-next-app` runs `git init` inside
    `my-app/`, the run's own snapshot repo at the parent then committed
    `my-app` as a gitlink, and the next run's scan reported `0 files
    detected` over a 22-file Next.js app. The planner was told the
    directory was BLANK and had to argue with that instruction for a
    whole plan before settling on `npm install` rather than a fresh
    scaffold — the premise that deleted `my-app` on 2026-09-17.
    """
    out: set[str] = set()
    for rel in visible:
        path = os.path.join(directory, rel.rstrip("/"))
        if not os.path.isdir(path):
            out.add(rel)
            continue
        prefix = rel.rstrip("/") + "/"
        inner = None
        if os.path.exists(os.path.join(path, ".git")):
            inner = _get_git_tracked_files(path)
        if inner is None:
            inner = set()
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in _NESTED_SKIP_DIRS]
                for name in files:
                    inner.add(os.path.relpath(os.path.join(root, name),
                                              path).replace("\\", "/"))
        out.update(prefix + f for f in inner)
    return out


def _is_gitignored(rel_path: str, visible: set[str] | None) -> bool:
    """Check if a file is excluded by git — NOT merely absent from the index.

    This function used to answer "is it tracked?" while being named, and
    read, as "is it ignored?". Those differ for exactly one case, and it
    is the most common state a project is ever in: **files the user has
    created and not yet committed.**

    Measured 2026-09-17. A user made a Next.js + TypeScript app in
    `my-app/` by hand, never committed it, and asked for a home page. The
    scan ran at 09:39:03 and reported `0 files detected`; the run's own
    git checkpoint — which is what made those files tracked, and what
    later allowed them to be recovered — ran at 09:39:07, four seconds
    too late. The planner was handed a project containing an empty
    `my-app/` directory and reasonably concluded it was a scaffold in
    waiting:

        --STEP 1.1 [CMD] Initialize a JavaScript Next.js App Router
        project ... in the blank `my-app` shell, replacing only the empty
        directory structure
        > rmdir /s /q my-app && npm create next-app@latest my-app -- --js

    Nineteen files were deleted and replaced with a JavaScript scaffold.
    The intent agent had read `layout.tsx` moments earlier and knew the
    app was there — one subsystem knew while another asserted the
    opposite, and the empty scan is what reached the plan.

    So the question asked here is now the one git actually answers:
    tracked files PLUS untracked-but-not-ignored ones are the project.
    Only genuinely ignored paths are skipped, which is what the name
    always promised.
    """
    if visible is None:
        return False  # No git info available, rely on skip lists
    return rel_path.replace("\\", "/") not in visible


def scan_project(directory: str = ".") -> dict:
    """Walk *directory* and return project structure info.

    Returns::

        {
            "tree": str,          # indented directory tree
            "key_files": dict,    # filename → first N lines of content
            "file_count": int,
            "languages": dict,    # extension → count
        }
    """
    tree_lines: list[str] = []
    key_files: dict[str, str] = {}
    file_count = 0
    lang_counts: dict[str, int] = {}

    abs_dir = os.path.abspath(directory)
    git_tracked = _get_git_tracked_files(abs_dir)

    for root, dirs, files in os.walk(abs_dir):
        # Filter out skipped directories (in-place so os.walk respects it)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)

        rel_root = os.path.relpath(root, abs_dir)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        indent = "  " * depth
        dir_name = os.path.basename(root) if rel_root != "." else "."
        tree_lines.append(f"{indent}{dir_name}/")

        for fname in sorted(files):
            if fname in SKIP_FILES:
                continue
            _, ext = os.path.splitext(fname)
            if ext in SKIP_EXTENSIONS:
                continue

            # Skip gitignored files
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, abs_dir).replace("\\", "/")
            if _is_gitignored(rel_path, git_tracked):
                continue

            file_count += 1
            tree_lines.append(f"{indent}  {fname}")

            # Count language extensions
            if ext:
                lang_counts[ext] = lang_counts.get(ext, 0) + 1

            # Read key files (up to limit)
            if fname in KEY_FILENAMES and len(key_files) < _MAX_KEY_FILES:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        lines = []
                        for line_no, line in enumerate(f):
                            if line_no >= _MAX_LINES_PER_FILE:
                                lines.append(f"... (truncated at {_MAX_LINES_PER_FILE} lines)")
                                break
                            lines.append(line.rstrip())
                        key_files[rel_path] = "\n".join(lines)
                except OSError:
                    pass

    return {
        "tree": "\n".join(tree_lines),
        "key_files": key_files,
        "file_count": file_count,
        "languages": lang_counts,
    }


def collect_source_files(directory: str = ".") -> dict[str, str]:
    """Read actual source files from the project directory.

    Returns a dict of ``{relative_path: content}`` for source files,
    respecting size limits to avoid overwhelming memory.
    """
    abs_dir = os.path.abspath(directory)
    source_files: dict[str, str] = {}
    total_chars = 0
    git_tracked = _get_git_tracked_files(abs_dir)

    for root, dirs, files in os.walk(abs_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)

        for fname in sorted(files):
            if fname in SKIP_FILES:
                continue
            _, ext = os.path.splitext(fname)
            if ext not in SOURCE_EXTENSIONS:
                continue
            if ext in SKIP_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)

            # Skip files that are too large (likely generated/bundled)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue
            if size > _MAX_FILE_SIZE_BYTES or size == 0:
                continue

            rel_path = os.path.relpath(fpath, abs_dir).replace("\\", "/")

            # Skip gitignored files
            if _is_gitignored(rel_path, git_tracked):
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue

            total_chars += len(content)
            if total_chars > _MAX_TOTAL_CHARS:
                break

            source_files[rel_path] = content

            if len(source_files) >= _MAX_SOURCE_FILES:
                break

        if (len(source_files) >= _MAX_SOURCE_FILES
                or total_chars > _MAX_TOTAL_CHARS):
            break

    return source_files


def format_scan_for_planner(
    scan_result: dict,
    max_chars: int = 6000,
    source_files: dict[str, str] | None = None,
) -> str:
    """Format scan results into a compact context string for the planner prompt.

    When *source_files* is provided, their content is included alongside the
    tree and key files so the planner can see actual code.
    Truncates to *max_chars* to avoid overwhelming the context window.
    """
    has_sources = bool(source_files)
    parts: list[str] = []

    # Budget allocation depends on whether source files are provided
    if has_sources:
        tree_budget = max_chars * 3 // 10       # 30%
        key_budget = max_chars * 2 // 10        # 20%
        source_budget = max_chars * 5 // 10     # 50%
    else:
        tree_budget = max_chars * 2 // 5        # 40%
        key_budget = max_chars * 3 // 5         # 60%
        source_budget = 0

    # Project tree
    tree = scan_result.get("tree", "")
    if len(tree) > tree_budget:
        tree = tree[:tree_budget] + "\n... (tree truncated)"
    parts.append(f"## Project Structure\n```\n{tree}\n```")

    # Key files (config/build files)
    parts.append("\n## Key Files")
    remaining_key = key_budget
    for fpath, content in scan_result.get("key_files", {}).items():
        entry = f"\n### {fpath}\n```\n{content}\n```"
        if len(entry) > remaining_key:
            break
        parts.append(entry)
        remaining_key -= len(entry)

    # Source files (actual code)
    if has_sources:
        parts.append("\n## Existing Source Files")
        remaining_src = source_budget
        for fpath, content in source_files.items():
            # Truncate individual files if needed
            max_file_chars = min(remaining_src, 4000)
            if len(content) > max_file_chars:
                content = content[:max_file_chars] + "\n... (truncated)"
            entry = f"\n### {fpath}\n```\n{content}\n```"
            if len(entry) > remaining_src:
                break
            parts.append(entry)
            remaining_src -= len(entry)

    # Stats
    fc = scan_result.get("file_count", 0)
    src_count = len(source_files) if source_files else 0
    stats = f"\n**{fc} files detected."
    if src_count:
        stats += f" {src_count} source files loaded.**"
    else:
        stats += "**"
    parts.append(stats)

    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"
    return result
