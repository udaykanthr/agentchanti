"""Validate verify_todo_cli against reference implementations.

A probe that has never passed a correct program and never failed a broken
one is not evidence. Three times this session the snake probe misjudged
aider before it was right, so the reference cases come first.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import verify_todo_cli as P  # noqa: E402

GOOD_PY = '''\
import json, sys
from pathlib import Path
DB = Path("todos.json")

def load():
    return json.loads(DB.read_text()) if DB.is_file() else []

def save(t):
    DB.write_text(json.dumps(t))

def main(argv):
    t = load()
    if not argv:
        print("usage: add|list|done|remove"); return 1
    cmd = argv[0]
    if cmd == "add":
        t.append({"text": " ".join(argv[1:]), "done": False}); save(t)
        print(f"Added: {t[-1]['text']}"); return 0
    if cmd == "list":
        if not t:
            print("No tasks"); return 0
        for i, x in enumerate(t, 1):
            print(f"{i}. [{'x' if x['done'] else ' '}] {x['text']}")
        return 0
    if cmd in ("done", "remove"):
        try:
            i = int(argv[1]) - 1
            item = t[i]
        except (IndexError, ValueError):
            print("No such task", file=sys.stderr); return 1
        if i < 0:
            print("No such task", file=sys.stderr); return 1
        if cmd == "done":
            item["done"] = True; save(t); print(f"Done: {item['text']}")
        else:
            t.pop(i); save(t); print(f"Removed: {item['text']}")
        return 0
    print("unknown command", file=sys.stderr); return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''

# Everything in memory: each process starts empty, so nothing survives.
NO_PERSIST_PY = GOOD_PY.replace(
    'return json.loads(DB.read_text()) if DB.is_file() else []', 'return []')
# Bad input reported as success - the exit-code half of the contract.
NO_EXITCODE_PY = GOOD_PY.replace(
    'print("No such task", file=sys.stderr); return 1',
    'print("No such task", file=sys.stderr); return 0')
# `done` prints a confirmation but never records anything.
NO_DONE_PY = GOOD_PY.replace('item["done"] = True; save(t);',
                             'save(t);')

CASES = [
    ("correct implementation", GOOD_PY, "PASS"),
    ("state never persists", NO_PERSIST_PY, "FAIL"),
    ("bad index exits 0", NO_EXITCODE_PY, "FAIL"),
    ("done never marks [x]", NO_DONE_PY, "FAIL"),
]

bad = 0
for name, src, expect in CASES:
    d = Path(tempfile.mkdtemp(prefix="todo-ref-"))
    (d / "main.py").write_text(src, encoding="utf-8")
    verdict, detail, _info = P.check_contract(d)
    ok = verdict == expect
    bad += not ok
    print(f"  [{'ok ' if ok else 'BAD'}] {name:<26} expected {expect}, "
          f"got {verdict} — {detail[:78]}")
    shutil.rmtree(d, ignore_errors=True)

# An empty project must be UNKNOWN, never FAIL.
d = Path(tempfile.mkdtemp(prefix="todo-ref-"))
verdict, detail, _ = P.check_contract(d)
ok = verdict == "UNKNOWN"
bad += not ok
print(f"  [{'ok ' if ok else 'BAD'}] {'no entry point':<26} expected UNKNOWN, "
      f"got {verdict}")
shutil.rmtree(d, ignore_errors=True)

print("\nPROBE VALIDATION", "PASSED" if not bad else f"FAILED ({bad})")
sys.exit(1 if bad else 0)
