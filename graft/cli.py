"""Command line interface for graft."""

from __future__ import annotations

import argparse

from graft.__about__ import __version__
from graft.exceptions import GraftError
from graft.registry import build_connector, get_connector_class, list_connectors
from graft.specs import ENCODER_SPECS, LLM_SPECS
from graft.summary import connector_summary


def _cmd_list(args: argparse.Namespace) -> int:
    for name in list_connectors():
        print(name)
    return 0


def _cmd_specs(args: argparse.Namespace) -> int:
    print("encoders:")
    for name, enc in ENCODER_SPECS.items():
        print(f"  {name}: dim={enc.hidden_dim}, tokens={enc.num_tokens}")
    print("llms:")
    for name, lm in LLM_SPECS.items():
        print(f"  {name}: dim={lm.hidden_dim}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    try:
        cls = get_connector_class(args.name)
    except GraftError as exc:
        print(f"error: {exc}")
        return 1
    print(f"{args.name} -> {cls.__name__}")
    doc = (cls.__doc__ or "").strip()
    if doc:
        print()
        print(doc)
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    overrides: dict[str, object] = {}
    if args.num_query_tokens is not None:
        overrides["num_query_tokens"] = args.num_query_tokens
    if args.scale_factor is not None:
        overrides["extra"] = {"scale_factor": args.scale_factor}
    try:
        connector = build_connector(args.connector, encoder=args.encoder, llm=args.llm, **overrides)
        num_input_tokens = ENCODER_SPECS[args.encoder].num_tokens
    except GraftError as exc:
        print(f"error: {exc}")
        return 1
    print(connector_summary(connector, num_input_tokens=num_input_tokens))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graft", description="visual connectors for LLMs")
    parser.add_argument("--version", action="version", version=f"graft {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list registered connectors")
    p_list.set_defaults(func=_cmd_list)

    p_specs = sub.add_parser("specs", help="list known encoder/LLM specs")
    p_specs.set_defaults(func=_cmd_specs)

    p_desc = sub.add_parser("describe", help="describe a connector")
    p_desc.add_argument("name")
    p_desc.set_defaults(func=_cmd_describe)

    p_plan = sub.add_parser("plan", help="build a connector and print a summary")
    p_plan.add_argument("--connector", required=True)
    p_plan.add_argument("--encoder", required=True)
    p_plan.add_argument("--llm", required=True)
    p_plan.add_argument("--num-query-tokens", type=int, default=None, dest="num_query_tokens")
    p_plan.add_argument("--scale-factor", type=int, default=None, dest="scale_factor")
    p_plan.set_defaults(func=_cmd_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
