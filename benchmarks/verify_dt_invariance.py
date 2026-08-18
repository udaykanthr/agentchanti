"""Independent playability check for a generated tile-maze game.

Ground truth that does not trust the pipeline's own claim. Three things
are asserted, and each caught a real artifact that its own generated
tests passed and the pipeline reported as a success:

1. WALL SAFETY - drive the game at several timestep profiles and assert
   no entity ever occupies a wall tile. A game that only holds together
   at a fixed 1/60 dt fails here.
2. EVERY DIRECTION RETURNS - send each of the four directions under a
   wall-clock cap. One artifact computed the distance to the next tile
   centre as the CURRENT centre when moving left or up, so its movement
   loop never consumed its travel budget and spun forever: pressing Left
   or Up froze the game. It passed all five wall profiles above with
   zero wall-frames, so before this check the verdict was PASS.
3. THE PLAYER MAKES PROGRESS - drive it around and require the pellet
   count to fall (or the score to rise). Another artifact collected a
   pellet only where the player came to a STOP, because its "at tile
   centre" test was true only when stationary; a moving Pac-Man ate
   nothing and the maze could never be cleared.

2 and 3 are LIVENESS checks, and they are why this file is no longer
only about dt: every invariant it used to assert was negative ("nothing
bad happened"), which a game where nothing happens at all satisfies
perfectly.

    cd <generated-project> && python verify_dt_invariance.py

Exit code 0 = PASS, 1 = FAIL, 2 = could not verify. A run where some
checks pass and others cannot be derived exits 2, not 0 — a partial pass
is not a pass, and a caller must never record an unverified game as
verified.

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
import threading

# Wall-clock cap for any single driven phase. Generous: the slowest honest
# artifact measured ran 2,600 frames in 1.3s, so anything near this bound
# is not slow, it is stuck.
_WATCHDOG_SECONDS = 60

_DIR_VECTORS = {"LEFT": (-1, 0), "RIGHT": (1, 0),
                "UP": (0, -1), "DOWN": (0, 1)}

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


# Append-only, like _SENDER_NAMES and for the same reason: each of these
# was the only way some artifact leaves its start screen, and a missing
# name costs the whole run. `start_new_game` was absent, so a working game
# — score 0 -> 20, ghosts roaming — reported five inert dt profiles and
# refused on both halves.
_STARTER_NAMES = ("start_game", "start_playing", "start_new_game",
                  "new_game", "start", "begin_game", "begin", "new_round")

# Tokens an artifact may accept as "begin play" when it exposes no starter
# method at all. Tried through update(), and kept only if `state` moves.
_START_TOKENS = ("space", "start", "enter", "return", {"start": True},
                 {"space": True})


def _state_of(g):
    return str(getattr(g, "state", ""))


def _new_game(Game, kw):
    g = Game(**kw)
    # A game left on its start screen ignores input and often does not
    # advance entities at all, so every check downstream reads as "nothing
    # moves". One artifact named this `start_playing`, which was missing
    # here, and its whole liveness result was a false "cannot verify".
    for starter in _STARTER_NAMES:
        if hasattr(g, starter):
            getattr(g, starter)()
            return g

    # No starter method exists on some artifacts: the START -> PLAYING
    # transition is requested through the update input instead, e.g.
    # `update(0.0, "space")`. Leaving such a game on its start screen is
    # not a cosmetic miss — its entities never move, so the dt profiles
    # score zero wall-frames while proving nothing at all. Measured on one
    # artifact: 900 frames, not one entity displaced, reported clean.
    #
    # Nothing is assumed: a token is accepted only if `state` actually
    # changes, which is the artifact reporting the transition itself.
    before = _state_of(g)
    if "start" not in before.lower():
        return g
    for token in _START_TOKENS:
        try:
            _advance(g)(0.0, token)
        except Exception:
            continue
        if _state_of(g) != before:
            return g
    return g


# ---------------------------------------------------------------------------
# Coordinate space
# ---------------------------------------------------------------------------

def _to_tile(coord, scale, mode):
    return round(coord / scale) if mode == "round" else int(coord // scale)


def _tuple_tile(e):
    """(a, b) from a tuple accessor — ORDER UNKNOWN, see _entity_tile."""
    # "current_tile" belongs here: two artifacts in one session stored no
    # pixel position at all — an entity was (current_tile, destination_tile,
    # segment_progress) — so every pixel strategy below failed and the run
    # could only be refused. Ordering is still unverified for any tuple
    # accessor, which is what _check_polarity exists to catch.
    for attr in ("tile", "tile_pos", "get_tile_pos", "grid_pos",
                 "tile_position", "current_tile"):
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
        # Conventions seen: (dx, dy), ((dx, dy),), and NAMES ("RIGHT" /
        # Direction.RIGHT). Guarding only TypeError let a name-based API
        # raise ValueError("Unknown movement direction: 1") straight out of
        # this probe to the top-level handler, which reported a working
        # game as "VERDICT: FAIL - game raised ValueError".
        for setter in ("set_direction", "set_player_direction"):
            fn_set = getattr(g, setter, None) or getattr(p, setter, None)
            if not fn_set:
                continue
            for arg in ((1, 0), ((1, 0),), ("RIGHT",), ("right",)):
                try:
                    fn_set(*arg)
                except Exception:
                    continue
                break
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


# A "wall"-named method whose name also carries one of these reads as the
# COMPLEMENT: is_rect_wall_free(x, y, r) is true where there is NO wall.
# Taking the name at face value inverts every frame of the run.
_NEGATED = ("free", "clear", "open", "no_wall", "not_wall", "without")
_COMPLEMENT_WORDS = ("walkable", "is_open", "can_move", "passable")

# Names whose two-argument form takes a DIRECTION (or another tile) as the
# second argument rather than a second coordinate. Two artifacts in one
# session were refused or misread because of this: can_move(tile, direction)
# raised "Unknown movement direction: 1", and walkable_neighbor(tile,
# direction) matched on "walkable" and then answered a different question
# entirely, failing the polarity check on a game that was fine.
# See the arity guard in _wall_query.
_DIRECTIONAL_NAMES = ("can_move", "can_go", "can_pass", "can_turn",
                      "neighbor", "neighbour", "adjacent", "toward",
                      "direction", "segment")

# The unambiguous spellings of "is this coordinate a wall / walkable".
# Anything else is qualified (is_position_walkable, is_entity_..._walkable)
# and may be asking in a different coordinate space.
_CANONICAL_QUERIES = ("is_wall", "wall_at", "is_wall_tile", "is_walkable",
                      "walkable", "is_open", "passable", "is_passable")


def _positional_arity(fn):
    """Number of REQUIRED positional parameters, or None if unreadable.

    A coordinate query takes 2 (col,row) or 1 (a single (x,y) tuple).
    Anything else — is_rect_wall_free(x, y, radius) — is a different
    question wearing a matching name, and calling it either explodes or,
    worse, silently answers something else.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    return sum(1 for p in params
               if p.default is inspect.Parameter.empty
               and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD))


def _call_coord_fn(fn, x, y):
    """Call *fn* with whichever coordinate convention it accepts.

    Artifacts split about evenly between ``f(x, y)`` and ``f((x, y))``.
    Assuming one raised ``TypeError: pixel_to_tile() takes 2 positional
    arguments but 3 were given`` out of a probe, which the top-level
    handler then reported as ``VERDICT: FAIL - game raised TypeError`` —
    a working game recorded as broken.

    Returns the (col, row) pair, or None if neither convention works, so
    the caller can fall through to another strategy rather than die.
    """
    for args in ((x, y), ((x, y),)):
        try:
            result = fn(*args)
        except Exception:
            continue
        try:
            col, row = result
        except (TypeError, ValueError):
            continue
        return int(col), int(row)
    return None


def _wall_query(game):
    """Find the wall test by CAPABILITY, not by name.

    Returns (fn, takes_pixels, invert). Refuses rather than guesses.
    """
    for _, obj in vars(game).items():
        cands = []
        for m in dir(obj):
            if m.startswith("_"):
                continue
            fn = getattr(obj, m, None)
            if not callable(fn):
                continue
            low = m.lower()
            is_wall_named = "wall" in low
            is_complement = any(w in low for w in _COMPLEMENT_WORDS)
            if not (is_wall_named or is_complement):
                continue
            # Arity is checked BEFORE the name is trusted: a 3-arg
            # is_rect_wall_free used to outrank a perfectly good
            # is_walkable purely because it spelled "wall", and the
            # verifier then refused on a game it could drive.
            arity = _positional_arity(fn)
            if arity not in (1, 2):
                continue
            # A two-argument can_move/walkable_neighbor is (tile,
            # direction), not (col, row) — the second argument is a
            # DIRECTION. Calling it as a coordinate query passes a row
            # where a direction belongs. The one-argument form (a single
            # tile) is still a fine walkability test.
            if arity == 2 and any(w in low for w in _DIRECTIONAL_NAMES):
                continue
            invert = is_complement or (is_wall_named
                                       and any(n in low for n in _NEGATED))
            cands.append((m, fn, invert, arity))

        if not cands:
            continue

        def rank(c):
            name, _fn, invert, arity = c
            low = name.lower()
            return (
                0 if "pixel" in low else 1,       # pixel-native first
                0 if arity == 2 else 1,           # two coords beat a tuple
                0 if not invert else 1,           # direct wall test first
                0 if ("tile" in low or low in ("is_wall", "wall_at")) else 1,
                # Plainest name wins. is_walkable and is_position_walkable
                # tied on every term above, so the winner came down to
                # dir() ordering — and "position" usually means PIXELS, so
                # the alphabetical winner was answering in the wrong
                # coordinate space and failed polarity on a sound game.
                0 if low in _CANONICAL_QUERIES else 1,
                len(low),                         # deterministic tiebreak
                low,
            )

        name, fn, invert, _ = sorted(cands, key=rank)[0]
        return fn, "pixel" in name.lower(), invert
    raise SystemExit("no wall query found on the game — refusing to guess")


def entity_in_wall(entity, game):
    """True when *entity* occupies a wall tile."""
    raw, takes_pixels, invert = _wall_query(game)
    row_first = _takes_row_first(raw)

    def query(col, row):
        a, b = (row, col) if row_first else (col, row)
        try:
            v = raw(a, b)
        except TypeError:
            # Some generated maps take a single (x, y) tuple rather than
            # two positional args. Refusing to try it made this verifier
            # report VERDICT: FAIL on a game that drives perfectly well —
            # a vocabulary mismatch recorded as a defect, which is exactly
            # what the exit-2 "cannot verify" path exists to prevent.
            try:
                v = raw((a, b))
            except TypeError:
                raise SystemExit(
                    "wall query accepts neither (x, y) nor ((x, y)) — "
                    "refusing to guess")
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
        tile = _call_coord_fn(maze.pixel_to_tile, px, py)
        if tile is not None:
            return query(*tile)
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


def _entity_marks(g):
    """Movement marks for every entity, via the SAME resolver the player
    uses. An earlier version had its own narrower accessor list and could
    not read a `position` property, so it reported "NOTHING MOVED" for an
    artifact that demonstrably moves — turning a real result into a
    fabricated warning about itself."""
    return tuple(_entity_mark(e)
                 for e in [g.player] + list(g.ghosts))


def _run(Game, kw, label, dt_fn, frames, rng):
    """Return (wall-frames, did-anything-move).

    The second value exists because "no entity entered a wall" is a
    NEGATIVE invariant, and a game where nothing happens satisfies it
    perfectly. Measured on an artifact that starts through its update
    input and so never left its start screen here: 900 frames, not one
    entity displaced, zero wall-frames — reported clean while proving
    nothing. That is the same trap the liveness checks were added to
    catch, in the file that added them.
    """
    g = _new_game(Game, kw)
    advance = _advance(g)
    before = _entity_marks(g)
    bad, moved = 0, False
    for _ in range(frames):
        advance(dt_fn(rng))
        if not moved and _entity_marks(g) != before:
            moved = True
        for e in [g.player] + list(g.ghosts):
            if entity_in_wall(e, g):
                bad += 1
    print(f"{label:<32} wall-frames = {bad}"
          f"{'' if moved else '   (NOTHING MOVED)'}")
    return bad, moved


# ---------------------------------------------------------------------------
# Non-termination
# ---------------------------------------------------------------------------

def _guard(what, seconds, fn, *args):
    """Run *fn* under a wall-clock cap. A hang is a FAIL, not a refusal.

    Non-termination is the one defect this file could not see: every check
    above assumes the artifact eventually returns. Observed 2026-08-12 —
    a generated ``Player.update`` computed the distance to the next tile
    centre as the CURRENT centre when moving left or up, so the distance
    was 0, the "snap" branch consumed none of the travel budget, and
    ``while remaining > eps`` spun forever. Pressing Left or Up froze the
    game. There was no exception and no failing test: 8 generated tests
    passed and the pipeline reported success, because nothing drove the
    player left through ``Game.update`` from a tile centre.

    This is deliberately a FAIL and not a refusal. Everywhere else in this
    file an unknown is "I cannot verify"; a game that never returns is not
    unknown, it is broken, and a caller must not record it as unverified.

    A spinning thread cannot be killed in Python, so the worker is a
    daemon and the process leaves via ``os._exit`` rather than waiting on
    it at interpreter shutdown.
    """
    box: dict = {}

    def target():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:              # noqa: BLE001
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        print(f"\nVERDICT: FAIL - {what} did not return within {seconds}s "
              f"(non-termination)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ---------------------------------------------------------------------------
# Liveness: does anything actually happen?
# ---------------------------------------------------------------------------

def _local_modules(Game):
    """Modules belonging to the artifact, nearest first.

    Direction vocabularies live wherever the artifact put them —
    ``Direction`` in entities.py, ``DIRECTIONS`` in player.py, plain
    tuples in game.py have all been seen for the same prompt.
    """
    here = os.path.abspath(".")
    mods = [sys.modules[Game.__module__]]
    for mod in list(sys.modules.values()):
        path = getattr(mod, "__file__", None)
        if not path or mod in mods:
            continue
        try:
            if os.path.abspath(path).startswith(here):
                mods.append(mod)
        except Exception:
            continue
    return mods


def _direction_values(Game):
    """Candidate encodings of "go left", newest artifact first.

    Every one of these was the ONLY accepted form in some run: a bare
    vector, a ``Direction`` enum member, a ``DIRECTIONS["LEFT"]`` lookup,
    a lowercase string. Passing the wrong one is silent — one artifact's
    ``queue_direction`` ignored anything not in ``DIRECTIONS.values()``,
    so a string input simply did nothing and the player sat still. That
    reads exactly like a broken game, which is why nothing here is
    assumed: an encoding counts only once the player is seen to move.
    """
    encoders = [("vector", lambda n: _DIR_VECTORS[n])]
    for mod in _local_modules(Game):
        enum = getattr(mod, "Direction", None)
        if enum is not None and all(hasattr(enum, n) for n in _DIR_VECTORS):
            encoders.append((f"{mod.__name__}.Direction",
                             lambda n, e=enum: getattr(e, n)))
        table = getattr(mod, "DIRECTIONS", None)
        if isinstance(table, dict) and all(n in table for n in _DIR_VECTORS):
            encoders.append((f"{mod.__name__}.DIRECTIONS",
                             lambda n, t=table: t[n]))
    encoders.append(("name", lambda n: n))
    encoders.append(("lowercase name", lambda n: n.lower()))
    return encoders


# Every name here was the ONLY input entry point on some artifact. The list
# is append-only for a reason: `set_player_direction` was missing, so a
# perfectly playable game reported "no input method visibly moved the
# player" and lost its whole liveness result to a refusal.
_SENDER_NAMES = ("queue_direction", "request_direction", "set_direction",
                 "set_player_direction", "set_desired_direction",
                 "set_next_direction", "buffer_direction",
                 "change_direction", "move", "handle_input", "handle_key",
                 "turn")


def _senders(g):
    """(label, send) pairs to try, on the game and then on the player.

    `send` is normally a bound method. The one exception is the sentinel
    ``_UPDATE_INPUT``, meaning "this artifact has no direction method —
    hand the value to update() each frame instead"; `_drive` and
    `_check_progress` know how to honour it. It is tried LAST, so a real
    method always wins, and it only appears when update actually accepts
    a second positional argument.
    """
    out = []
    for holder_name, holder in (("game", g), ("player", getattr(g, "player", None))):
        if holder is None:
            continue
        for name in _SENDER_NAMES:
            fn = getattr(holder, name, None)
            if callable(fn):
                out.append((f"{holder_name}.{name}", fn))
    step = g.step if hasattr(g, "step") else getattr(g, "update", None)
    if step is not None and _update_takes_input(step):
        out.append((_UPDATE_INPUT, _UPDATE_INPUT))
    return out


_UPDATE_INPUT = "update(dt, input)"


def _update_takes_input(fn):
    """True when the step function accepts a second positional argument.

    Read from the signature rather than guessed; anything unreadable
    (builtins, *args) falls back to the single-argument form that every
    other artifact uses.
    """
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return False
    return len(params) >= 2


def _advance(g):
    """Return ``advance(dt, value=None)`` for this artifact.

    Some games have no direction method at all — input arrives as the
    second argument of update, e.g. `update(dt, "right")`, and the same
    channel carries the start request. Callers that have no input to send
    pass nothing and get the old single-argument behaviour.
    """
    # `advance` is checked FIRST and was missing entirely: every artifact
    # in the Pac-Man benchmark family exposes `Game.advance(dt)` because
    # its task prompt mandates that exact name, and this file refused all
    # of them with "could not verify" for want of one attribute lookup.
    # A verifier that cannot drive the artifact proves nothing about it.
    for _name in ("advance", "step", "update"):
        fn = getattr(g, _name, None)
        if callable(fn):
            break
    else:
        raise SystemExit("no advance/step/update method on the game object")
    takes_input = _update_takes_input(fn)

    def advance(dt, value=None):
        if value is not None and takes_input:
            return fn(dt, value)
        return fn(dt)

    return advance


def _player_mark(g):
    """Something that must change when the player moves.

    Sub-tile progress is folded in where the artifact exposes it. A
    tile-granular mark needs a WHOLE tile of travel to register, and a
    player that dies first respawns on its start tile — so a direction
    that moved fine reads as "did not move", differently on each run
    because ghost RNG is unseeded.
    """
    return _entity_mark(getattr(g, "player", None))


def _entity_mark(p):
    """Finest available position signal for one entity, or None."""
    if p is None:
        return None

    def _position_tuple(entity):
        pos = getattr(entity, "position", None)
        if pos is None:
            return None
        try:
            values = tuple(pos)
        except Exception:
            return None
        return values if len(values) == 2 else None

    # FINEST FIRST. Preferring the tile hid a real defect: an artifact whose
    # player advanced 2.4px on its first frame and then never moved again
    # never changed tile, so every direction read as "did not move", no
    # input API resolved, and both liveness checks were skipped as a
    # refusal — reported as "cannot verify" when the truthful answer is
    # that Pac-Man cannot move. With a pixel mark the API resolves, the
    # progress sweep runs, and the run fails as it should.
    mark = None
    for probe in (_entity_pixels, _position_tuple, _entity_tile, _tuple_tile):
        found = probe(p)
        if found is not None:
            mark = tuple(found)
            break
    if mark is None:
        return None
    for fine in ("progress", "segment_progress", "edge_progress", "offset"):
        value = getattr(p, fine, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return mark + (round(float(value), 4),)
    return mark


def _drive(g, send, value, frames=60, dt=0.02):
    """Send one direction and step; did the player move at ANY point?

    Sampled every frame, not compared start-to-end: a ghost catching the
    player respawns it on its start tile, so an end-state comparison
    reports "never moved" for a direction that moved fine and then died.
    Observed on a working artifact whose DOWN neighbour is walkable —
    and if that had happened on all four directions the verdict would
    have been a false "no direction moves the player".
    """
    before = _player_mark(g)
    if before is None:
        return False
    carried = None
    if send == _UPDATE_INPUT:
        carried = value          # delivered every frame, via update
    else:
        send(value)
    advance = _advance(g)
    for _ in range(frames):
        advance(dt, carried)
        if _player_mark(g) != before:
            return True
    return False


def _resolve_input_api(Game, kw):
    """Find a (label, send-factory, encoder) that visibly moves the player.

    Verified against the artifact's own behaviour rather than assumed —
    the rule this whole file runs on. Returns None when nothing works,
    which is a refusal for the liveness checks only; the wall checks above
    do not need input and keep their verdict.
    """
    for enc_label, encode in _direction_values(Game):
        probe = _new_game(Game, kw)
        for send_label, _ in _senders(probe):
            for name in _DIR_VECTORS:
                g = _new_game(Game, kw)
                send = dict(_senders(g)).get(send_label)
                if send is None:
                    continue
                try:
                    moved = _guard(
                        f"input probe {send_label}({enc_label} {name})",
                        _WATCHDOG_SECONDS, _drive, g, send, encode(name))
                except Exception:
                    break            # this sender rejects this encoding
                if moved:
                    return f"{send_label} <- {enc_label}", send_label, encode
    return None


def _check_directions(Game, kw, send_label, encode):
    """Every direction must RETURN. That is the assertion here.

    The freeze that motivated this affected LEFT and UP only, because
    their next-centre arithmetic was wrong while RIGHT and DOWN were
    fine, so the sweep has to cover all four; a single-direction probe
    would have passed it. A hang inside `_guard` exits FAIL by design.

    Which directions MOVED is reported but deliberately not a verdict.
    Some are walls at spawn, and even an open one can read as "did not
    move" when a ghost kills the player first — that flipped between
    identical runs of the same artifact. Liveness is carried instead by
    the progress check below, which sweeps 3,000 frames and accumulates
    across lives, and would fail anyway if the player could not move.
    """
    moved = []
    for name in _DIR_VECTORS:
        g = _new_game(Game, kw)
        send = dict(_senders(g)).get(send_label)
        if _guard(f"driving {name}", _WATCHDOG_SECONDS,
                  _drive, g, send, encode(name), 120):
            moved.append(name)
    print(f"{'all four directions return':<32} yes "
          f"(moved: {', '.join(moved) if moved else 'none observed'})")
    return True


def _progress_probe(g):
    """A counter that must change while the player eats.

    Pellets go down or the score goes up; either proves collection is
    wired to movement. Observed 2026-08-12: an artifact collected a pellet
    only where the player came to a STOP, because its "at tile centre"
    test was ``logical_tile == destination_tile`` — true only when
    stationary. A moving Pac-Man ate nothing and the maze could never be
    cleared. Seven generated tests passed; one of them proved a single
    pellet scores, from a standing start.
    """
    # Resolved from the live game on EVERY read, never captured. Restarting
    # rebuilds the map on some artifacts, so a reader holding the original
    # object keeps reporting a maze the game no longer plays on: progress
    # froze after the first death and read as +1 where the real figure was
    # 36. Under-reporting here is not cosmetic — zero is a FAIL.
    holder_attrs = [None, "map", "maze", "board", "level", "grid", "player"]

    def read(holder_attr, name):
        holder = g if holder_attr is None else getattr(g, holder_attr, None)
        if holder is None:
            return None
        value = getattr(holder, name, None)
        if value is None:
            return None
        try:
            value = value() if callable(value) else value
        except Exception:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        try:
            return len(value)
        except Exception:
            return None

    def label_of(holder_attr, name):
        return f"{holder_attr or 'game'}.{name}"

    for holder_attr in holder_attrs:
        for name in ("pellets_remaining", "remaining_pellet_count",
                     "pellet_count", "pellets_left", "remaining_pellets",
                     "remaining_collectibles", "collectibles_remaining",
                     "dots_remaining",
                     "pellets", "collectibles", "collectible_tiles",
                     "collectible_positions", "pellet_tiles"):
            if read(holder_attr, name) is not None:
                return (label_of(holder_attr, name),
                        lambda h=holder_attr, n=name: read(h, n), -1)
    for holder_attr in holder_attrs:
        if read(holder_attr, "score") is not None:
            return (label_of(holder_attr, "score"),
                    lambda h=holder_attr: read(h, "score"), +1)
    return None


def _check_progress(Game, kw, send_label, encode, rng):
    """Drive the player around and require the collection counter to move."""
    g = _new_game(Game, kw)
    probe = _progress_probe(g)
    if probe is None:
        print(f"{'collection progress':<32} cannot verify "
              f"(no pellet count or score found)")
        return None
    label, read, sign = probe
    send = dict(_senders(g)).get(send_label)

    def sweep():
        """Total progress ACROSS lives, not endpoint minus start.

        Reviving restores the pellets, so comparing the first and last
        readings measures only the final life — and a working game whose
        last life happens to eat nothing before dying would be reported
        as collecting nothing at all. Movement in the wrong direction is
        a reset: re-baseline on it rather than counting it.
        """
        advance = _advance(g)
        names = list(_DIR_VECTORS)
        gained, last = 0, read()
        carried = None
        for frame in range(3000):
            if frame % 20 == 0:
                value = encode(rng.choice(names))
                if send == _UPDATE_INPUT:
                    carried = value
                else:
                    try:
                        send(value)
                    except Exception:
                        pass
            advance(0.02, carried)
            now = read()
            if now is None:
                break
            step = (now - last) * sign
            if step > 0:
                gained += step
            last = now
            # A death that ends the run would cap progress at whatever the
            # first life managed; revive so the sweep keeps eating.
            state = str(getattr(g, "state", "")).lower()
            if "over" in state:
                for revive in ("restart", "reset", "start_game",
                               "start_playing", "start"):
                    if hasattr(g, revive):
                        getattr(g, revive)()
                        last = read()
                        break
        return gained

    gained = _guard("the collection sweep", _WATCHDOG_SECONDS, sweep)
    print(f"{'collection progress':<32} {label} +{gained} across the sweep")
    return gained > 0


def main() -> int:
    global _TILE
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    sys.path.insert(0, ".")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    # Everything up to the first _run is DERIVATION: working out this
    # artifact's vocabulary by probing it. An exception here means a guess
    # of ours was wrong, which says nothing about the game — but it used to
    # escape to the top-level handler and print "VERDICT: FAIL - game
    # raised ...". That happened twice in one benchmark session, on games
    # an independent drive then showed to be clean. Converted to a refusal
    # (exit 2); only the drive loop below can produce a FAIL.
    try:
        Game = _find_game()
        kw = ({"headless": True}
              if "headless" in inspect.signature(Game).parameters else {})
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"probing the artifact raised {type(exc).__name__}: {exc} — "
            f"could not derive its API, which is not evidence about the "
            f"game") from exc

    rng = random.Random(20260802)

    # The wall checks need the artifact's coordinate space; the liveness
    # checks below do not — they only need something observable to change.
    # So a coordinate refusal must no longer abort the program, or the
    # liveness checks are dead code on every artifact whose positions we
    # cannot map. Two of three artifacts in one session exposed only a
    # `position` tuple of unproven order, refused here, and would have
    # carried a freeze past this file untested.
    total, wall_note = None, None
    try:
        _TILE = _tile_size(Game, kw, sys.modules[Game.__module__])
        _check_polarity(Game, kw)
        # Guarded: a profile that never returns is the artifact spinning,
        # and exits FAIL from inside _guard rather than hanging here.
        results = _guard(
            "the dt profiles", _WATCHDOG_SECONDS * len(PROFILES),
            lambda: [_run(Game, kw, label, fn, frames, rng)
                     for label, fn, frames in PROFILES])
        if not any(moved for _, moved in results):
            # Zero wall-frames out of a game that never moved is not
            # evidence of wall safety, so it must not be recorded as any.
            raise SystemExit(
                "no entity moved in any dt profile, so zero wall-frames "
                "proves nothing about wall safety")
        # Assigned only once the result MEANS something. Setting it before
        # the check left total=0 through the refusal, so the summary went
        # on to claim "no entity entered a wall" about a game that never
        # moved — the precise false claim this guard exists to stop.
        total = sum(bad for bad, _ in results)
    except SystemExit as exc:
        total, wall_note = None, str(exc.code)
        print(f"{'wall checks':<32} cannot verify ({wall_note})")
    except Exception as exc:                       # noqa: BLE001
        total = None
        wall_note = f"probing raised {type(exc).__name__}: {exc}"
        print(f"{'wall checks':<32} cannot verify ({wall_note})")

    # Liveness. Deriving the input vocabulary can fail on an artifact that
    # exposes none we recognise; that is a refusal for THESE checks alone,
    # so the wall verdict above still stands.
    api = _resolve_input_api(Game, kw)
    if api is None:
        print(f"{'liveness':<32} cannot verify "
              f"(no input method visibly moved the player)")
        moved_any, progressed = None, None
    else:
        api_label, send_label, encode = api
        print(f"{'input api':<32} {api_label}")
        moved_any = _check_directions(Game, kw, send_label, encode)
        progressed = _check_progress(Game, kw, send_label, encode, rng)

    print()
    failures = []
    if total:
        failures.append(f"{total} wall-frames")
    if progressed is False:
        failures.append("nothing is ever collected while driving")
    if failures:
        print("VERDICT: FAIL - " + "; ".join(failures))
        return 1

    # A partial pass is not a pass. Whatever ran is reported as clean, but
    # the exit code stays 2 so a caller never records an unverified game as
    # verified — the same reason a refusal has never shared FAIL's code.
    proved = [name for name, ok in (("no entity entered a wall", total == 0),
                                    ("every direction returns", moved_any),
                                    ("the player makes progress", progressed))
              if ok]
    unproven = [name for name, ok in (("wall safety", total is not None),
                                      ("direction liveness", moved_any is not None),
                                      ("collection progress", progressed is not None))
                if not ok]
    if unproven:
        print("CLEAN SO FAR: " + ("; ".join(proved) if proved else "nothing"))
        raise SystemExit("could not verify " + ", ".join(unproven))
    print("VERDICT: PASS - " + ", ".join(proved))
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
