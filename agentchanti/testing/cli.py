"""
`agentchanti test` subcommand CLI.

Commands
--------
agentchanti test record  --url <start_url> --out <trace.jsonl>
agentchanti test normalize --trace <trace.jsonl> --out <spec.yaml>
agentchanti test replay  --spec <spec.yaml> --report <report.json>

Heavy deps (playwright, mcp client) are imported lazily inside each handler
so merely running `agentchanti --help` never touches them.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_record(args: argparse.Namespace) -> int:
    from .recorder import Recorder
    rec = Recorder(mcp_server_url=args.mcp_server, output_path=args.out)
    rec.start(start_url=args.url)
    rec.stop()
    return 0


def _cmd_normalize(args: argparse.Namespace) -> int:
    from .normalizer import Normalizer
    # LLM client wiring will mirror what orchestrator/cli.py does — TODO.
    raise NotImplementedError("agentchanti test normalize is not wired yet")


def _cmd_replay(args: argparse.Namespace) -> int:
    from .replayer import Replayer
    from .validator import Validator
    from .reporter import Reporter  # noqa: F401 — used once replay loop is wired
    raise NotImplementedError("agentchanti test replay is not wired yet")


def test_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="agentchanti test",
        description="Agent-driven browser + API end-to-end testing.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="Record a browser session via MCP.")
    p_rec.add_argument("--url", required=True, help="Start URL for the session.")
    p_rec.add_argument("--out", required=True, help="Path to write the raw trace.")
    p_rec.add_argument("--mcp-server", default="http://localhost:8931",
                       help="Browser MCP server URL.")
    p_rec.set_defaults(func=_cmd_record)

    p_norm = sub.add_parser("normalize", help="Convert a raw trace into a semantic spec.")
    p_norm.add_argument("--trace", required=True, help="Raw trace path from `record`.")
    p_norm.add_argument("--out", required=True, help="Where to write the semantic spec (YAML).")
    p_norm.set_defaults(func=_cmd_normalize)

    p_rep = sub.add_parser("replay", help="Replay a spec against a live browser + validate.")
    p_rep.add_argument("--spec", required=True, help="Semantic spec path.")
    p_rep.add_argument("--report", required=True, help="Where to write the JSON report.")
    p_rep.add_argument("--mcp-server", default="http://localhost:8931",
                       help="Browser MCP server URL.")
    p_rep.set_defaults(func=_cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(test_main(sys.argv[1:]))
