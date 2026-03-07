"""
Updater — fetches knowledge base updates from a GitHub-hosted registry.

Checks for new releases, downloads update archives, and applies them
atomically.  Respects HTTP_PROXY/HTTPS_PROXY environment variables and
optional GITHUB_TOKEN for private repos.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_GLOBAL_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.join(_GLOBAL_DIR, "core")
_REGISTRY_DIR = os.path.join(_GLOBAL_DIR, "registry")
_MANIFEST_PATH = os.path.join(_CORE_DIR, "manifest.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UpdateStatus:
    """Result of an update check."""

    current_version: str
    latest_version: str
    update_available: bool
    changelog: str = ""


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _parse_semver(version: str) -> tuple[int, ...]:
    """Parse a semver string like '1.2.3' into a comparable tuple."""
    parts = version.lstrip("v").split(".")
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            result.append(0)
    return tuple(result)


def _load_local_manifest() -> dict:
    """Load the local core/manifest.json."""
    if not os.path.isfile(_MANIFEST_PATH):
        return {"version": "0.0.0", "categories": []}
    try:
        with open(_MANIFEST_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"version": "0.0.0", "categories": []}


def _save_local_manifest(data: dict) -> None:
    """Save the local core/manifest.json."""
    os.makedirs(os.path.dirname(_MANIFEST_PATH), exist_ok=True)
    with open(_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _github_headers() -> dict[str, str]:
    """Build GitHub API headers, including token if available."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "agentchanti-kb-updater",
    }
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _http_get(url: str, headers: Optional[dict] = None) -> bytes:
    """
    Perform an HTTP GET request.

    Respects HTTP_PROXY / HTTPS_PROXY environment variables.

    Raises
    ------
    ConnectionError
        On any network failure.
    """
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, method="GET")
    if headers:
        for key, val in headers.items():
            req.add_header(key, val)

    try:
        # urllib automatically reads proxy from env vars
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise ConnectionError(
            f"HTTP {exc.code} for {url}: {exc.reason}"
        ) from exc
    except Exception as exc:
        raise ConnectionError(f"Network error fetching {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean() -> dict:
    """
    Remove all global KB data: vector store, errors DB, registry files,
    and manifest.

    After running this, ``kb seed`` and ``kb update`` can repopulate the
    KB from scratch.

    Returns
    -------
    dict
        Summary with keys: files_removed, dbs_removed.
    """
    summary = {"files_removed": 0, "dbs_removed": 0}

    # Remove registry markdown files (preserve .gitignore)
    if os.path.isdir(_REGISTRY_DIR):
        for dirpath, _, filenames in os.walk(_REGISTRY_DIR):
            for fname in filenames:
                if fname == ".gitignore":
                    continue
                try:
                    os.remove(os.path.join(dirpath, fname))
                    summary["files_removed"] += 1
                except OSError as exc:
                    logger.warning("Failed to remove %s: %s", fname, exc)
        # Remove empty subdirectories but keep registry/ itself
        for dirpath, dirnames, filenames in os.walk(_REGISTRY_DIR, topdown=False):
            if dirpath == _REGISTRY_DIR:
                continue
            try:
                os.rmdir(dirpath)  # only removes if empty
            except OSError:
                pass

    # Remove SQLite databases and manifest
    for db_name in ("errors.db", "global_kb.db", "manifest.json"):
        db_path = os.path.join(_CORE_DIR, db_name)
        if os.path.isfile(db_path):
            try:
                os.remove(db_path)
                summary["dbs_removed"] += 1
            except OSError as exc:
                logger.warning("Failed to remove %s: %s", db_name, exc)

    return summary


def get_version() -> str:
    """Return the current local KB version string."""
    manifest = _load_local_manifest()
    return manifest.get("version", "0.0.0")


def get_manifest_info() -> dict:
    """Return the full local manifest as a dict."""
    return _load_local_manifest()


def check_for_updates(owner: str, repo: str) -> UpdateStatus:
    """
    Check whether a newer version is available on GitHub releases.

    Parameters
    ----------
    owner:
        GitHub repository owner / organisation.
    repo:
        GitHub repository name.

    Returns
    -------
    UpdateStatus
        Comparison of local vs. remote versions.
    """
    local_manifest = _load_local_manifest()
    current_version = local_manifest.get("version", "0.0.0")

    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    try:
        data = json.loads(_http_get(url, _github_headers()))
    except ConnectionError as exc:
        logger.warning("Could not check for updates: %s", exc)
        return UpdateStatus(
            current_version=current_version,
            latest_version=current_version,
            update_available=False,
            changelog=f"Check failed: {exc}",
        )

    latest_version = data.get("tag_name", "").lstrip("v")
    changelog = data.get("body", "")

    update_available = (
        _parse_semver(latest_version) > _parse_semver(current_version)
    )

    return UpdateStatus(
        current_version=current_version,
        latest_version=latest_version,
        update_available=update_available,
        changelog=changelog,
    )


def download_update(
    owner: str,
    repo: str,
    version: Optional[str] = None,
    categories: Optional[list[str]] = None,
) -> dict:
    """
    Download and apply an update from GitHub releases.

    The update is applied atomically — on any failure the previous
    state is fully restored.

    Parameters
    ----------
    owner:
        GitHub repository owner.
    repo:
        GitHub repository name.
    version:
        Specific version to download.  Defaults to latest.
    categories:
        Optional list of categories to update.  Defaults to all.

    Returns
    -------
    dict
        Summary: {version, files_updated, errors_updated}.
    """
    summary: dict = {
        "version": "",
        "files_updated": 0,
        "errors_updated": 0,
        "md_files": [],
    }

    # Determine which release to fetch
    if version:
        url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/v{version}"
    else:
        url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"

    try:
        release_data = json.loads(_http_get(url, _github_headers()))
    except ConnectionError as exc:
        raise ConnectionError(f"Failed to fetch release info: {exc}") from exc

    tag = release_data.get("tag_name", "").lstrip("v")
    summary["version"] = tag

    # Find the zip asset
    assets = release_data.get("assets", [])
    zip_url = None
    for asset in assets:
        name = asset.get("name", "")
        if name.startswith("kb-registry-") and name.endswith(".zip"):
            zip_url = asset.get("browser_download_url")
            break

    if not zip_url:
        # Fall back to source zipball
        zip_url = release_data.get("zipball_url")

    if not zip_url:
        raise ConnectionError("No downloadable asset found in release")

    # Download to temp file
    tmpdir = tempfile.mkdtemp(prefix="agentchanti_kb_update_")
    try:
        zip_path = os.path.join(tmpdir, "update.zip")
        zip_data = _http_get(zip_url, _github_headers())
        with open(zip_path, "wb") as fh:
            fh.write(zip_data)

        # Extract
        extract_dir = os.path.join(tmpdir, "extracted")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Find the root of extracted content
        entries = os.listdir(extract_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            extract_dir = os.path.join(extract_dir, entries[0])

        # Backup current registry
        backup_dir = os.path.join(tmpdir, "backup")
        if os.path.isdir(_REGISTRY_DIR):
            shutil.copytree(_REGISTRY_DIR, backup_dir)

        # Apply update atomically
        try:
            files_updated, md_files = _apply_update(extract_dir, categories)
            summary["files_updated"] = files_updated
            summary["md_files"] = md_files

            # Update manifest version
            manifest = _load_local_manifest()
            manifest["version"] = tag
            _save_local_manifest(manifest)

        except Exception as exc:
            # Rollback
            logger.error("Update failed, rolling back: %s", exc)
            if os.path.isdir(backup_dir):
                if os.path.isdir(_REGISTRY_DIR):
                    shutil.rmtree(_REGISTRY_DIR)
                shutil.copytree(backup_dir, _REGISTRY_DIR)
            raise

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return summary


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

_CATEGORY_DIRS = {
    "patterns": "patterns",
    "adrs": "adrs",
    "docs": "docs",
    "behavioral": "behavioral",
    "errors": "errors",
}


def _apply_update(
    source_dir: str,
    categories: Optional[list[str]] = None,
) -> tuple[int, list[tuple[str, str, str]]]:
    """
    Copy files from extracted update into registry/.

    Parameters
    ----------
    source_dir:
        Path to the extracted update directory.
    categories:
        Optional filter.

    Returns
    -------
    tuple[int, list[tuple[str, str, str]]]
        A tuple of (number of files copied, list of (path, category, title)
        for each markdown file that was copied).  The second element can be
        passed directly to the seeder's ``_embed_md_files`` to index the
        newly downloaded documents.
    """
    from .seeder import _parse_frontmatter

    # Map directory names to the KB category names used in the vector store
    _DIR_TO_CATEGORY = {
        "patterns": "pattern",
        "adrs": "adr",
        "docs": "doc",
        "behavioral": "behavioral",
    }

    files_copied = 0
    md_files: list[tuple[str, str, str]] = []

    for src_name, dest_name in _CATEGORY_DIRS.items():
        if categories and src_name not in categories:
            continue

        src_path = os.path.join(source_dir, src_name)
        if not os.path.isdir(src_path):
            continue

        if src_name == "errors":
            # SQL patches or full errors.db
            _apply_error_updates(src_path)
            files_copied += 1
            continue

        dest_path = os.path.join(_REGISTRY_DIR, dest_name)
        os.makedirs(dest_path, exist_ok=True)

        # Walk recursively — the registry zip may have nested
        # subdirectories (e.g. docs/frameworks/guide.md).  Flatten
        # all .md files into the single registry category directory.
        for dirpath, _, filenames in os.walk(src_path):
            for fname in filenames:
                src_file = os.path.join(dirpath, fname)
                dest_file = os.path.join(dest_path, fname)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dest_file)
                    files_copied += 1

                    # Track markdown files for embedding
                    if fname.endswith(".md"):
                        try:
                            with open(dest_file, encoding="utf-8") as fh:
                                content = fh.read(500)
                            meta = _parse_frontmatter(content)
                            title = meta.get("title", fname)
                        except OSError:
                            title = fname
                        category = _DIR_TO_CATEGORY.get(dest_name, dest_name)
                        md_files.append((dest_file, category, title))

    return files_copied, md_files


def _apply_error_updates(errors_dir: str) -> None:
    """
    Apply error database updates from .sql patch files or full errors.db.

    Parameters
    ----------
    errors_dir:
        Directory containing error update files.
    """
    from .error_dict import ErrorDict

    db_path = os.path.join(_CORE_DIR, "errors.db")
    edict = ErrorDict(db_path)

    for fname in sorted(os.listdir(errors_dir)):
        filepath = os.path.join(errors_dir, fname)

        if fname.endswith(".sql"):
            # Apply SQL patch
            import sqlite3
            try:
                with open(filepath, encoding="utf-8") as fh:
                    sql = fh.read()
                conn = sqlite3.connect(db_path, timeout=10)
                conn.executescript(sql)
                conn.close()
                logger.info("Applied SQL patch: %s", fname)
            except Exception as exc:
                logger.warning("Failed to apply SQL patch %s: %s", fname, exc)

        elif fname == "errors.db":
            # Full replacement (backup first)
            backup = db_path + ".bak"
            if os.path.isfile(db_path):
                shutil.copy2(db_path, backup)
            shutil.copy2(filepath, db_path)
            logger.info("Replaced errors.db from update")
