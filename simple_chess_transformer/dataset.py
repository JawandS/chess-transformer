"""Dataset preparation for simple move-sequence modeling from `games.csv`."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")


@dataclass
class PreparedDatasetSummary:
    input_path: str
    output_dir: str
    max_games_requested: int
    games_kept: int
    vocab_size: int
    max_sequence_length: int


def iter_csv_move_sequences(
    input_path: Path,
    max_games: int | None = None,
    min_plies: int = 10,
    max_plies: int = 200,
    highest_rated: bool = False,
    min_average_rating: int | None = None,
) -> Iterator[list[str]]:
    import chess
    import csv

    parsed_games: list[tuple[float, list[str]]] = []
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            san_moves = (row.get("moves") or "").split()
            if len(san_moves) < min_plies:
                continue

            try:
                white_rating = int(row.get("white_rating") or 0)
                black_rating = int(row.get("black_rating") or 0)
            except ValueError:
                continue
            average_rating = (white_rating + black_rating) / 2
            if min_average_rating is not None and average_rating < min_average_rating:
                continue

            board = chess.Board()
            uci_moves: list[str] = []
            valid_game = True
            for san_move in san_moves[:max_plies]:
                try:
                    move = board.parse_san(san_move)
                except ValueError:
                    valid_game = False
                    break
                uci_moves.append(move.uci())
                board.push(move)

            if not valid_game:
                continue

            parsed_games.append((average_rating, uci_moves))

    if highest_rated:
        parsed_games.sort(key=lambda item: item[0], reverse=True)

    games_seen = 0
    for _, uci_moves in parsed_games:
        if max_games is not None and games_seen >= max_games:
            break
        games_seen += 1
        yield uci_moves


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    max_games: int = 5000,
    min_plies: int = 10,
    max_plies: int = 200,
    min_move_frequency: int = 2,
    highest_rated: bool = False,
    min_average_rating: int | None = None,
) -> PreparedDatasetSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences: list[list[str]] = []
    counter: Counter[str] = Counter()
    games_kept = 0
    max_sequence_length = 0

    for moves in iter_csv_move_sequences(
        input_path=input_path,
        max_games=max_games,
        min_plies=min_plies,
        max_plies=max_plies,
        highest_rated=highest_rated,
        min_average_rating=min_average_rating,
    ):
        games_kept += 1
        sequences.append(moves)
        counter.update(moves)
        max_sequence_length = max(max_sequence_length, len(moves) + 2)

    vocab = list(SPECIAL_TOKENS)
    vocab.extend(
        move for move, count in sorted(counter.items()) if count >= min_move_frequency
    )

    vocab_path = output_dir / "vocab.json"
    train_path = output_dir / "games.jsonl"
    meta_path = output_dir / "meta.json"

    vocab_path.write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")
    with train_path.open("w", encoding="utf-8") as handle:
        for moves in sequences:
            handle.write(json.dumps({"moves": moves}) + "\n")

    summary = PreparedDatasetSummary(
        input_path=str(input_path),
        output_dir=str(output_dir),
        max_games_requested=max_games,
        games_kept=games_kept,
        vocab_size=len(vocab),
        max_sequence_length=max_sequence_length,
    )
    meta_path.write_text(json.dumps(summary.__dict__, indent=2) + "\n", encoding="utf-8")
    return summary
