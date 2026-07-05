"""CLI for project bootstrapping and inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from chess_transformer.config import default_experiment_config
from chess_transformer.pipeline import bootstrap_project, render_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chess-transformer",
        description=(
            "Research scaffold for studying style asymmetry in self-play chess transformers."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="Print the default experiment plan.")
    subparsers.add_parser("show-config", help="Print the default experiment config JSON.")

    init_parser = subparsers.add_parser(
        "init", help="Create working directories and write the default config."
    )
    init_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root to initialize. Defaults to the current working directory.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = default_experiment_config()

    if args.command == "plan":
        print(render_plan(config))
        return

    if args.command == "show-config":
        print(config.to_json())
        return

    if args.command == "init":
        config_path = bootstrap_project(args.root, config)
        print(f"Initialized project scaffold at {args.root}")
        print(f"Wrote config to {config_path}")
        return

    parser.error(f"Unsupported command: {args.command}")
