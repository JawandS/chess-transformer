"""CLI for the minimal Lichess-to-transformer experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_chess_transformer.dataset import prepare_dataset
from simple_chess_transformer.train import TrainConfig, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chess-transformer",
        description="Prepare move sequences from games.csv and train a small model locally.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-dataset", help="Parse games.csv into move-token sequences."
    )
    prepare_parser.add_argument(
        "--input",
        type=Path,
        default=Path("simple_chess_transformer/games.csv"),
    )
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--max-games", type=int, default=5000)
    prepare_parser.add_argument("--min-plies", type=int, default=10)
    prepare_parser.add_argument("--max-plies", type=int, default=200)
    prepare_parser.add_argument("--min-move-frequency", type=int, default=2)

    train_parser = subparsers.add_parser(
        "train", help="Train a small causal transformer on the prepared dataset."
    )
    train_parser.add_argument("--dataset-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--embedding-dim", type=int, default=128)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--ff-dim", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-dataset":
        summary = prepare_dataset(
            input_path=args.input,
            output_dir=args.output_dir,
            max_games=args.max_games,
            min_plies=args.min_plies,
            max_plies=args.max_plies,
            min_move_frequency=args.min_move_frequency,
        )
        print(f"Prepared {summary.games_kept} games in {summary.output_dir}")
        print(f"Vocabulary size: {summary.vocab_size}")
        return

    if args.command == "train":
        config = TrainConfig(
            dataset_dir=str(args.dataset_dir),
            output_dir=str(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            max_batches_per_epoch=args.max_batches_per_epoch,
            device=args.device,
            seed=args.seed,
        )
        summary = train_model(config)
        print(f"Finished training on {summary['num_games']} games")
        print(f"Final perplexity: {summary['history'][-1]['perplexity']:.3f}")
        return

    parser.error(f"Unsupported command: {args.command}")
