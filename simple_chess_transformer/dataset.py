"""Dataset preparation for simple move-sequence modeling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, TextIO


SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")


@dataclass
class PreparedDatasetSummary:
    input_path: str
    output_dir: str
    max_games_requested: int
    games_kept: int
    vocab_size: int
    max_sequence_length: int


def _open_text_input(path: Path) -> TextIO:
    if path.suffix == ".zst":
        import zstandard

        binary_handle = path.open("rb")
        dctx = zstandard.ZstdDecompressor()
        stream = dctx.stream_reader(binary_handle)
        return stream_text_reader(stream, binary_handle)
    return path.open("r", encoding="utf-8", errors="replace")


class stream_text_reader:
    """Wrap a decompression stream as a text handle with clean close semantics."""

    def __init__(self, stream, binary_handle) -> None:
        import io

        self._binary_handle = binary_handle
        self._stream = stream
        self._text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")

    def __enter__(self):
        return self._text

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._text.close()
        self._stream.close()
        self._binary_handle.close()

    def __getattr__(self, name: str):
        return getattr(self._text, name)


def iter_pgn_move_sequences(
    input_path: Path,
    max_games: int | None = None,
    min_plies: int = 10,
    max_plies: int = 200,
) -> Iterator[list[str]]:
    import chess.pgn

    games_seen = 0
    with _open_text_input(input_path) as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games_seen += 1
            if max_games is not None and games_seen > max_games:
                break

            moves = [move.uci() for move in game.mainline_moves()]
            if len(moves) < min_plies:
                continue
            yield moves[:max_plies]


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    max_games: int = 5000,
    min_plies: int = 10,
    max_plies: int = 200,
    min_move_frequency: int = 2,
) -> PreparedDatasetSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences: list[list[str]] = []
    counter: Counter[str] = Counter()
    games_kept = 0
    max_sequence_length = 0

    for moves in iter_pgn_move_sequences(
        input_path=input_path,
        max_games=max_games,
        min_plies=min_plies,
        max_plies=max_plies,
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
