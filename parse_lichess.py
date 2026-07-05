"""Stream Lichess PGNs from stdin and emit FEN-to-move training examples as CSV."""

from __future__ import annotations

import argparse
import csv
import sys

import chess.pgn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read PGN from stdin and write chess training examples as CSV."
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=1000,
        help="Maximum number of games to parse before stopping.",
    )
    parser.add_argument(
        "--min-plies",
        type=int,
        default=2,
        help="Skip games shorter than this many plies.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write the CSV header row.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    writer = csv.writer(sys.stdout)

    if not args.no_header:
        writer.writerow(
            [
                "white_elo",
                "black_elo",
                "time_control",
                "result",
                "fen_before",
                "uci_move",
            ]
        )

    games_kept = 0
    while games_kept < args.max_games:
        game = chess.pgn.read_game(sys.stdin)
        if game is None:
            break

        moves = list(game.mainline_moves())
        if len(moves) < args.min_plies:
            continue

        headers = game.headers
        white_elo = headers.get("WhiteElo", "")
        black_elo = headers.get("BlackElo", "")
        time_control = headers.get("TimeControl", "")
        result = headers.get("Result", "")

        board = game.board()
        for move in moves:
            writer.writerow(
                [
                    white_elo,
                    black_elo,
                    time_control,
                    result,
                    board.fen(),
                    move.uci(),
                ]
            )
            board.push(move)

        games_kept += 1


if __name__ == "__main__":
    main()
