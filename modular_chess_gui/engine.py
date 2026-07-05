"""Core engine loop for human and model play."""

from __future__ import annotations

from dataclasses import dataclass, field
import sys

import chess

from modular_chess_gui.notation import NotationStyle, format_move, parse_move_text
from modular_chess_gui.players import BasePlayer, PlayerContext


@dataclass
class MoveRecord:
    ply: int
    color: str
    uci: str
    san: str
    lan: str
    fen_before: str
    fen_after: str
    actor: str


@dataclass
class EngineEvent:
    level: str
    message: str


@dataclass
class GameEngine:
    white_player: BasePlayer
    black_player: BasePlayer
    model_notation: NotationStyle = NotationStyle.UCI
    history_notation: NotationStyle = NotationStyle.SAN
    max_retries: int = 3
    cli_logging: bool = False
    board: chess.Board = field(default_factory=chess.Board)
    move_history: list[MoveRecord] = field(default_factory=list)
    events: list[EngineEvent] = field(default_factory=list)

    def reset(self) -> None:
        self.board.reset()
        self.move_history.clear()
        self.events.clear()

    def current_player(self) -> BasePlayer:
        return self.white_player if self.board.turn == chess.WHITE else self.black_player

    def side_name(self) -> str:
        return "White" if self.board.turn == chess.WHITE else "Black"

    def is_human_turn(self) -> bool:
        return self.current_player().is_human

    def status_text(self) -> str:
        if self.board.is_game_over():
            outcome = self.board.outcome()
            return f"Game over: {outcome.result()} ({outcome.termination.name})"
        return f"{self.side_name()} to move"

    def submit_human_move(self, move_text: str, notation_style: NotationStyle) -> bool:
        player = self.current_player()
        if not player.is_human:
            self._emit_event("error", "It is not a human turn.")
            return False
        return self._apply_move_text(move_text, notation_style, actor=player.name)

    def step_model_turn(self) -> bool:
        player = self.current_player()
        if player.is_human or self.board.is_game_over():
            return False

        error_message: str | None = None
        for retry_count in range(self.max_retries):
            context = PlayerContext(
                board_fen=self.board.fen(),
                move_history=tuple(record.uci for record in self.move_history),
                notation_style=self.model_notation,
                color=self.board.turn,
                retry_count=retry_count,
                error_message=error_message,
            )
            try:
                move_text = player.get_move_text(context)
            except Exception as exc:
                error_message = str(exc)
                self._emit_event(
                    "error",
                    (
                        f"{player.name} failed to produce a move "
                        f"on attempt {retry_count + 1}/{self.max_retries}: {error_message}"
                    ),
                )
                continue
            try:
                applied = self._apply_move_text(
                    move_text,
                    notation_style=self.model_notation,
                    actor=player.name,
                )
            except ValueError as exc:
                error_message = str(exc)
                self._emit_event(
                    "error",
                    (
                        f"{player.name} produced invalid move `{move_text}` "
                        f"on attempt {retry_count + 1}/{self.max_retries}: {error_message}"
                    ),
                )
                continue
            if applied:
                if retry_count > 0:
                    self._emit_event(
                        "info",
                        f"{player.name} succeeded after {retry_count + 1} attempts.",
                    )
                return True

        self._emit_event("error", f"{player.name} exceeded illegal move retry limit.")
        return False

    def _apply_move_text(
        self,
        move_text: str,
        notation_style: NotationStyle,
        actor: str,
    ) -> bool:
        parsed = parse_move_text(self.board, move_text, preferred_style=notation_style)
        board_before = self.board.copy(stack=False)
        move = parsed.move
        fen_before = board_before.fen()
        san = format_move(board_before, move, NotationStyle.SAN)
        lan = format_move(board_before, move, NotationStyle.LAN)
        self.board.push(move)
        self.move_history.append(
            MoveRecord(
                ply=len(self.move_history) + 1,
                color="white" if board_before.turn == chess.WHITE else "black",
                uci=move.uci(),
                san=san,
                lan=lan,
                fen_before=fen_before,
                fen_after=self.board.fen(),
                actor=actor,
            )
        )
        self._emit_event("info", f"{actor} played {san}")
        return True

    def _emit_event(self, level: str, message: str) -> None:
        event = EngineEvent(level, message)
        self.events.append(event)
        if self.cli_logging:
            print(f"[engine:{level}] {message}", file=sys.stderr)
