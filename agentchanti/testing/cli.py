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
    """Open a browser, inject the JS event recorder, block until the user
    presses Ctrl+C, write the trace."""
    from .recorder import Recorder

    viewport = None
    if args.viewport:
        try:
            w, h = (int(x) for x in args.viewport.lower().split("x", 1))
            viewport = {"width": w, "height": h}
        except ValueError:
            print(f"\n  [ERROR] --viewport expects WxH, got {args.viewport!r}\n",
                  file=sys.stderr)
            return 2

    print(f"opening session at {args.url}")
    print(f"writing trace to {args.out}")
    print()
    print("  Browse the site as you would normally — clicks, typing,")
    print("  selections, and key presses are streamed into the trace.")
    print("  Sensitive inputs (type=password, [data-sensitive]) are redacted.")
    print()
    print("  Press Ctrl+C when you're done to finalize the trace.")
    print()

    with Recorder.from_url(args.mcp_server, args.out) as rec:
        rec.start(start_url=args.url, viewport=viewport)
        rec.subscribe_to_live_events()
        try:
            # Block until the user signals stop. Sleeping in short slices
            # so KeyboardInterrupt is responsive on Windows.
            import time as _t
            while True:
                _t.sleep(0.5)
        except KeyboardInterrupt:
            pass
        out_path = rec.stop()

    print(f"\nrecording complete: {out_path}")
    return 0


def _cmd_normalize(args: argparse.Namespace) -> int:
    from ..config import Config
    from ..llm import MissingAPIKeyError, build_llm_client
    from .normalizer import Normalizer, NormalizerError

    cfg = Config.load(args.config)
    try:
        llm = build_llm_client(cfg, provider=args.provider, model=args.model)
    except MissingAPIKeyError as e:
        print(f"\n  [ERROR] {e.provider.title()} provider requires an API key.\n"
              f"  Set {e.env_var} env var or add it to .agentchanti.yaml.\n",
              file=sys.stderr)
        return 1

    try:
        out = Normalizer(llm, spec_name=args.name).normalize(args.trace, args.out)
    except NormalizerError as e:
        print(f"\n  [ERROR] normalize failed: {e}\n", file=sys.stderr)
        return 2
    print(f"wrote spec → {out}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from ..config import Config
    from ..llm import MissingAPIKeyError, build_llm_client
    from .locator_cache import DEFAULT_CACHE_PATH, LocatorCache
    from .mcp_client import BrowserMCPClient
    from .replayer import Replayer
    from .reporter import Reporter
    from .spec import Spec
    from .validator import Validator

    try:
        spec = Spec.load(args.spec)
    except (OSError, ValueError) as e:
        print(f"\n  [ERROR] failed to load spec {args.spec}: {e}\n", file=sys.stderr)
        return 2

    cfg = Config.load(args.config)
    llm = None
    if not args.no_llm:
        try:
            llm = build_llm_client(cfg, provider=args.provider, model=args.model)
        except MissingAPIKeyError as e:
            print(
                f"\n  [WARN] {e.provider.title()} provider requires {e.env_var}; "
                f"self-healing and natural-language assertions will be skipped.\n",
                file=sys.stderr,
            )

    cache = LocatorCache(args.cache_path or DEFAULT_CACHE_PATH)

    try:
        with BrowserMCPClient(args.mcp_server) as mcp:
            run_result = Replayer(mcp, cache, llm_client=llm).replay(spec)
            assertion_results = Validator(llm_client=llm).validate(spec, run_result)
    except NotImplementedError as e:
        print(
            f"\n  [ERROR] browser MCP transport is not yet wired: {e}\n"
            f"  This command needs a live Playwright MCP server + the real\n"
            f"  BrowserMCPClient transport layer. The rest of the replay\n"
            f"  pipeline (Spec → Replayer → Validator → Reporter) is fully\n"
            f"  exercised by the test suite with a FakeMCP client.\n",
            file=sys.stderr,
        )
        return 2
    except Exception as e:
        print(f"\n  [ERROR] replay failed: {e}\n", file=sys.stderr)
        return 2

    reporter = Reporter()
    print(reporter.render_console(assertion_results))
    if args.report:
        out = reporter.render_json(assertion_results, args.report)
        print(f"wrote JSON report → {out}")

    failed = any(not r.passed and not r.skipped for r in assertion_results)
    return 1 if failed else 0


def test_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="agentchanti test",
        description="Agent-driven browser + API end-to-end testing.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="Record a browser session via MCP.")
    p_rec.add_argument("--url", required=True, help="Start URL for the session.")
    p_rec.add_argument("--out", required=True, help="Path to write the raw trace.")
    p_rec.add_argument("--mcp-server", default="http://localhost:8931/mcp",
                       help="Browser MCP server URL (default: http://localhost:8931/mcp).")
    p_rec.add_argument("--viewport", default=None,
                       help="Viewport in WxH form, e.g. 1280x720. "
                            "Saved into the trace so replay can match it.")
    p_rec.set_defaults(func=_cmd_record)

    p_norm = sub.add_parser("normalize", help="Convert a raw trace into a semantic spec.")
    p_norm.add_argument("--trace", required=True, help="Raw trace path from `record`.")
    p_norm.add_argument("--out", required=True, help="Where to write the semantic spec (YAML).")
    p_norm.add_argument("--name", default=None,
                        help="Optional human name for the spec (hint to the LLM).")
    p_norm.add_argument("--provider", default=None,
                        choices=["ollama", "lm_studio", "openai", "gemini", "anthropic"],
                        help="LLM provider override (default: from config).")
    p_norm.add_argument("--model", default=None, help="Model name override.")
    p_norm.add_argument("--config", default=None, help="Path to .agentchanti.yaml.")
    p_norm.set_defaults(func=_cmd_normalize)

    p_rep = sub.add_parser("replay", help="Replay a spec against a live browser + validate.")
    p_rep.add_argument("--spec", required=True, help="Semantic spec path.")
    p_rep.add_argument("--report", default=None,
                       help="Optional path to write a JSON report for CI.")
    p_rep.add_argument("--mcp-server", default="http://localhost:8931",
                       help="Browser MCP server URL.")
    p_rep.add_argument("--cache-path", default=None,
                       help="Locator cache path (default: .agentchanti/testing/locator-cache.json).")
    p_rep.add_argument("--provider", default=None,
                       choices=["ollama", "lm_studio", "openai", "gemini", "anthropic"],
                       help="LLM provider for self-healing + NL assertions.")
    p_rep.add_argument("--model", default=None, help="LLM model override.")
    p_rep.add_argument("--config", default=None, help="Path to .agentchanti.yaml.")
    p_rep.add_argument("--no-llm", action="store_true",
                       help="Disable self-healing + NL assertions (skip LLM calls).")
    p_rep.set_defaults(func=_cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(test_main(sys.argv[1:]))
