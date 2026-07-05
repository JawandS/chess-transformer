"""Move parsing and formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import chess


class NotationStyle(str, Enum):
    AUTO = "auto"
    UCI = "uci"
    SAN = "san"
    LAN = "lan"


@dataclass(frozen=True)
class ParsedMove:
    move: chess.Move
    notation_used: NotationStyle


def available_notation_styles() -> list[str]:
    return [style.value for style in NotationStyle]


def parse_move_text(
    board: chess.Board,
    move_text: str,
    preferred_style: NotationStyle = NotationStyle.AUTO,
) -> ParsedMove:
    cleaned = move_text.strip()
    if not cleaned:
        raise ValueError("empty move text")

    styles = (
        [preferred_style]
        if preferred_style != NotationStyle.AUTO
        else [NotationStyle.UCI, NotationStyle.SAN, NotationStyle.LAN]
    )
    for style in styles:
        try:
            if style == NotationStyle.UCI:
                move = chess.Move.from_uci(cleaned)
            elif style == NotationStyle.SAN:
                move = board.parse_san(cleaned)
            elif style == NotationStyle.LAN:
                move = board.parse_san(cleaned)
            else:
                continue
        except ValueError:
            continue

        if move not in board.legal_moves:
            raise ValueError(f"illegal move: {cleaned}")
        return ParsedMove(move=move, notation_used=style)

    raise ValueError(f"could not parse move: {cleaned}")


def format_move(board_before: chess.Board, move: chess.Move, style: NotationStyle) -> str:
    if style == NotationStyle.UCI:
        return move.uci()
    if style == NotationStyle.SAN:
        return board_before.san(move)
    if style == NotationStyle.LAN:
        return board_before.lan(move)
    return move.uci()

