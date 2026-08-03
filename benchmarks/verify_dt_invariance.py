"""Independent dt-invariance check for a generated tile-maze game.

Ground truth that does not trust the pipeline's own claim: drive the
generated game at several timestep profiles and assert that no entity
ever occupies a wall tile. A game that only holds together at a fixed
1/60 dt fails here, which is the bug this exists to catch.

    cd <generated-project> && python verify_dt_invariance.py

Exit code 0 = PASS, 1 = FAIL, 2 = could not verify.

WHY SO MUCH OF THIS FILE IS REFUSALS
------------------------------------
Generated projects share no vocabulary. Across one benchmark session the
same prompt produced, among others:

    game.map / game.maze / game._game_map
    is_wall(col,row) / is_wall(tile_x,tile_y) / is_walkable(row,col)
    is_wall_pixel / is_wall_at_pixel / is_wall_tile
    positions in pixels, and positions in TILE UNITS as floats
    entity.tile() / .tile_pos / .grid_col+.grid_row / .tile_col+.tile_row

Every fixed assumption written here was wrong within a run or two, and a
wrong assumption does not produce an error — it produces a confident,
false verdict. Three separate times this harness reported a working game
as broken (once with 15,600 fabricated wall-frames, from feeding pixel
coordinates to a tile-indexed query).

So the rule throughout: derive the mapping from the artifact's own
behaviour, verify it against something the artifact itself reports, and
raise SystemExit rather than guess. A refusal is a usable result; a
fabricated number is not.
"""

from __future__ import annotations

import importlib
import inspect
import os
import random
import sys

# Profiles: the spec asks for ~0.008-0.05, the rest are stress.
PROFILES = (
    ("fixed 1/60", lambda r: 1 / 60, 900),
    ("fixed 1/30", lambda r: 1 / 30, 900),
    ("jittery .008-.05 (spec)", lambda r: r.uniform(0.008, 0.05), 900),
    ("hostile .001-.2", lambda r: r.uniform(0.001, 0.2), 900),
    ("single dt 1.0", lambda r: 1.0, 300),
)

_TILE = 0
_TILE_MODE = "floor"          # proved in _solve_tile_size()


# ---------------------------------------------------------------------------
# Locating the game
# ---------------------------------------------------------------------------

def _find_game():
    """The Game class, wherever this run decided to put it."""
    for mod in ("pacman", "pacman.game", "src.game", "game", "game.game",
                "src.main", "main"):
        try:
            g = getattr(importlib.import_module(mod), "Game")
            print(f"module: {mod}")
            return g
        except Exception:
            continue
    # Layouts vary (game/game.py, pacman_game.py, src/...). A hardcoded
    # list reports "no Game class" for a working artifact — a false
    # negative, not a failing game. Walk the tree instead.
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", ".agentchanti", ".git",
                                "venv", ".venv")]
        for f in files:
            if not f.endswith(".py") or f.startswith("test_"):
                continue
            rel = os.path.relpath(os.path.join(root, f), ".")
            name = rel[:-3].replace(os.sep, ".").lstrip(".")
            try:
                g = getattr(importlib.import_module(name), "Game")
                print(f"module: {name}")
                return g
            except Exception:
                continue
    raise SystemExit("no Game class found")


def _new_game(Game, kw):
    g = Game(**kw)
    for starter in ("start_game", "start"):
        if hasattr(g, starter):
            getattr(g, starter)()
            break
    return g


# ---------------------------------------------------------------------------
# Coordinate space
# ---------------------------------------------------------------------------

def _to_tile(coord, scale, mode):
    return round(coord / scale) if mode == "round" else int(coord // scale)


def _tuple_tile(e):
    """(a, b) from a tuple accessor — ORDER UNKNOWN, see _entity_tile."""
    for attr in ("tile", "tile_pos", "get_tile_pos", "grid_pos",
                 "tile_position"):
        v = getattr(e, attr, None)
        if v is None:
            continue
        try:
            t = tuple(v() if callable(v) else v)
        except Exception:
            continue
        if len(t) == 2:
            return t
    return None


def _entity_tile(e):
    """(col, row) from NAMED fields, or None.

    Named pairs only. A tuple accessor cannot tell you which element is
    the row — one artifact returned (row, col) where this harness assumed
    (col, row), which placed a ghost in a wall at spawn and would have
    reported a working game as broken.
    """
    for cx, cy in (("grid_col", "grid_row"), ("tile_col", "tile_row"),
                   ("col", "row"), ("tile_x", "tile_y"), ("tx", "ty")):
        if hasattr(e, cx) and hasattr(e, cy):
            return int(getattr(e, cx)), int(getattr(e, cy))
    return None


def _entity_pixels(e):
    for px, py in (("x", "y"), ("pixel_x", "pixel_y"), ("px", "py")):
        if hasattr(e, px) and hasattr(e, py):
            return getattr(e, px), getattr(e, py)
    return None


def _solve_tile_size(Game, kw):
    """Derive the pixel-per-tile transform from observed behaviour.

    Two unknowns, not one: the scale AND whether the artifact floors or
    rounds. One run stored positions in pixels and floored; another stored
    them in tile units as floats and rounded (x=9.0 -> tile 9), which no
    floor-only search can match. Returns 0 unless exactly one transform
    reproduces the artifact's own accessor across the samples.
    """
    global _TILE_MODE
    try:
        g = _new_game(Game, kw)
        p = g.player
        if _tuple_tile(p) is None:
            return 0
        advance = g.step if hasattr(g, "step") else g.update

        # The player must MOVE, or every sample sits on an exact integer
        # where floor and round agree and carry no information. A
        # stationary sweep once "proved" floor for an artifact that rounds.
        for setter in ("set_direction", "set_player_direction"):
            fn_set = getattr(g, setter, None) or getattr(p, setter, None)
            if fn_set:
                try:
                    fn_set(1, 0)
                except TypeError:
                    try:
                        fn_set((1, 0))
                    except Exception:
                        pass
                break

        samples = []
        for _ in range(240):
            t = _tuple_tile(p)
            if t is not None:
                samples.append((p.x, p.y, t))
            advance(1 / 60)
        if not [s for s in samples if s[2][0] or s[2][1]]:
            return 0
        if not [s for s in samples if s[0] % 1 or s[1] % 1]:
            print("  player never left exact tile centres — cannot tell "
                  "floor from round; refusing to guess")
            return 0

        fits = [(s, m)
                for s in [1] + list(range(4, 129))
                for m in ("floor", "round")
                if all(_to_tile(x, s, m) == t[0] and _to_tile(y, s, m) == t[1]
                       for x, y, t in samples)]
        if len(fits) != 1:
            print(f"  tile transform unresolved ({len(fits)} fit "
                  f"{len(samples)} samples) — refusing to guess")
            return 0
        scale, mode = fits[0]
        _TILE_MODE = mode
        print(f"  tile transform solved: {mode}(coord/{scale}) — verified "
              f"against {len(samples)} observed player positions")
        return scale
    except Exception:
        return 0


def _tile_size(Game, kw, mod):
    """Pixel size of a tile, but ONLY if the conversion can be proven.

    Feeding pixel coordinates to a tile-indexed is_wall() once reported
    every entity inside a wall on every frame — a catastrophic-looking
    result that was purely this harness's bug. No proof, no conversion.
    """
    size = (getattr(mod, "TILE_SIZE", None) or getattr(mod, "TILE", None)
            or getattr(mod, "CELL_SIZE", None))
    if not size:
        root = Game.__module__.split(".", 1)[0]
        for name, m in list(sys.modules.items()):
            if name.startswith(root):
                size = (getattr(m, "TILE_SIZE", None)
                        or getattr(m, "TILE", None)
                        or getattr(m, "CELL_SIZE", None))
                if size:
                    break
    if not size:
        return _solve_tile_size(Game, kw)
    try:
        p = _new_game(Game, kw).player
        # Best proof available: the artifact's OWN accessor against its
        # OWN pixel position.
        known = _tuple_tile(p)
        if known is None:
            spawn = getattr(p, "spawn_tile", None)
            if spawn is None:
                col, row = (getattr(p, "start_col", None),
                            getattr(p, "start_row", None))
                spawn = (col, row) if col is not None else None
            known = tuple(spawn) if spawn is not None else None
        if known is None:
            return 0
        derived = (int(p.x // size), int(p.y // size))
        if tuple(known) != derived:
            print(f"  tile size {size} rejected: {known} != derived {derived}")
            return 0
        print(f"  tile size {size} verified against {known}")
        return size
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Wall query
# ---------------------------------------------------------------------------

def _takes_row_first(fn):
    """True when the query's first parameter is the ROW.

    Read from the signature, not guessed: artifacts have shipped
    is_wall(col,row), is_wall(tile_x,tile_y) AND is_walkable(row,col).
    """
    try:
        names = list(inspect.signature(fn).parameters)[:2]
    except (TypeError, ValueError):
        return False
    return len(names) == 2 and ("row" in names[0].lower()
                                or names[0].lower() in ("r", "y", "ty"))


def _wall_query(game):
    """Find the wall test by CAPABILITY, not by name.

    Returns (fn, takes_pixels, invert). Refuses rather than guesses.
    """
    for _, obj in vars(game).items():
        methods = [m for m in dir(obj)
                   if "wall" in m.lower() and not m.startswith("_")
                   and callable(getattr(obj, m, None))]
        if methods:
            for m in methods:            # pixel-native: no conversion
                if "pixel" in m.lower():
                    return getattr(obj, m), True, False
            for m in methods:
                low = m.lower()
                if "tile" in low or low in ("is_wall", "wall_at"):
                    return getattr(obj, m), False, False
            return getattr(obj, methods[0]), False, False

        # Some artifacts expose only the COMPLEMENT (is_walkable/is_open).
        # Same information, opposite sign; reading one as the other
        # inverts every frame, so polarity is explicit and proved below.
        inv = [m for m in dir(obj)
               if not m.startswith("_") and callable(getattr(obj, m, None))
               and any(w in m.lower() for w in
                       ("walkable", "is_open", "can_move", "passable"))]
        if inv:
            for m in inv:
                if "pixel" in m.lower():
                    return getattr(obj, m), True, True
            return getattr(obj, inv[0]), False, True
    raise SystemExit("no wall query found on the game — refusing to guess")


def entity_in_wall(entity, game):
    """True when *entity* occupies a wall tile."""
    raw, takes_pixels, invert = _wall_query(game)
    row_first = _takes_row_first(raw)

    def query(col, row):
        v = raw(row, col) if row_first else raw(col, row)
        return (not v) if invert else v

    def as_pixels(col, row):
        if not _TILE:
            raise SystemExit("tile position but only a pixel query and no "
                             "verified tile size — refusing to guess")
        return query(col * _TILE + _TILE / 2, row * _TILE + _TILE / 2)

    # 1. NAMED col/row fields — unambiguous, so they win over any tuple.
    t = _entity_tile(entity)
    if t is not None:
        return as_pixels(*t) if takes_pixels else query(*t)

    # 2. tuple accessor — ordering is a guess, so only when nothing named.
    t = _tuple_tile(entity)
    if t is not None:
        col, row = int(t[0]), int(t[1])
        return as_pixels(col, row) if takes_pixels else query(col, row)

    # 3. pixel position
    pix = _entity_pixels(entity)
    if pix is None:
        raise SystemExit("entity exposes no position — refusing to guess")
    px, py = pix
    if takes_pixels:
        return query(px, py)
    maze = getattr(game, "map", None)
    if maze is not None and hasattr(maze, "pixel_to_tile"):
        return query(*maze.pixel_to_tile(px, py))
    if _TILE:
        return query(_to_tile(px, _TILE, _TILE_MODE),
                     _to_tile(py, _TILE, _TILE_MODE))
    raise SystemExit("cannot determine entity tile — refusing to guess")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_polarity(Game, kw):
    """Every entity must start OUT of a wall.

    If the harness says the spawn is a wall, the sign or the coordinate
    space is wrong, and every frame after would be wrong the same way —
    a working game reported as catastrophically broken. Refuse instead.
    """
    g = _new_game(Game, kw)
    stuck = sorted({type(e).__name__ for e in [g.player] + list(g.ghosts)
                    if entity_in_wall(e, g)})
    if stuck:
        raise SystemExit(
            f"polarity check FAILED: {stuck} already in a wall at spawn, "
            f"before any update — the wall query is inverted or the "
            f"coordinate space is wrong; refusing to report a verdict")
    print("  polarity verified: no entity is in a wall at spawn")


def _run(Game, kw, label, dt_fn, frames, rng):
    g = _new_game(Game, kw)
    advance = g.step if hasattr(g, "step") else g.update
    bad = 0
    for _ in range(frames):
        advance(dt_fn(rng))
        for e in [g.player] + list(g.ghosts):
            if entity_in_wall(e, g):
                bad += 1
    print(f"{label:<32} wall-frames = {bad}")
    return bad


def main() -> int:
    global _TILE
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    sys.path.insert(0, ".")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    Game = _find_game()
    kw = ({"headless": True}
          if "headless" in inspect.signature(Game).parameters else {})
    _TILE = _tile_size(Game, kw, sys.modules[Game.__module__])
    _check_polarity(Game, kw)

    rng = random.Random(20260802)
    total = sum(_run(Game, kw, label, fn, frames, rng)
                for label, fn, frames in PROFILES)
    print()
    if total:
        print(f"VERDICT: FAIL - {total} wall-frames")
        return 1
    print("VERDICT: PASS - no entity entered a wall")
    return 0


if __name__ == "__main__":
    # A refusal must NOT share an exit code with a real failure. Every
    # SystemExit raised above is "I cannot verify this", and reporting
    # that as FAIL is the same class of error the whole file guards
    # against — a caller would record a working game as broken.
    try:
        sys.exit(main())
    except SystemExit as exc:
        if isinstance(exc.code, int):
            raise
        print(f"CANNOT VERIFY: {exc.code}")
        sys.exit(2)
    except Exception as exc:                       # noqa: BLE001
        # The game itself blew up (one artifact raised IndexError before
        # Game() even returned). That is a real defect, not a refusal.
        print(f"VERDICT: FAIL - game raised {type(exc).__name__}: {exc}")
        sys.exit(1)
