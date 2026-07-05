"""Player adapters for humans and simple model-like policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
import random

import chess

from modular_chess_gui.notation import NotationStyle


@dataclass(frozen=True)
class PlayerContext:
    board_fen: str
    move_history: tuple[str, ...]
    notation_style: NotationStyle
    color: chess.Color
    retry_count: int
    error_message: str | None


class MoveProvider(Protocol):
    def __call__(self, context: PlayerContext) -> str:
        """Return a move string in the requested notation style."""


class BasePlayer:
    name = "base"
    is_human = False

    def get_move_text(self, context: PlayerContext) -> str:
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    name = "human"
    is_human = True

    def get_move_text(self, context: PlayerContext) -> str:
        raise RuntimeError("Human moves are provided through the GUI.")


class RandomMovePlayer(BasePlayer):
    name = "random"

    def get_move_text(self, context: PlayerContext) -> str:
        board = chess.Board(context.board_fen)
        return random.choice(list(board.legal_moves)).uci()


class FirstLegalMovePlayer(BasePlayer):
    name = "first-legal"

    def get_move_text(self, context: PlayerContext) -> str:
        board = chess.Board(context.board_fen)
        return next(iter(board.legal_moves)).uci()


class CallableModelPlayer(BasePlayer):
    name = "callable-model"

    def __init__(self, provider: MoveProvider, name: str = "callable-model") -> None:
        self.provider = provider
        self.name = name

    def get_move_text(self, context: PlayerContext) -> str:
        return self.provider(context)


def player_registry() -> dict[str, Callable[[], BasePlayer]]:
    from modular_chess_gui.transformer_adapter import SimpleTransformerPlayer

    return {
        HumanPlayer.name: HumanPlayer,
        RandomMovePlayer.name: RandomMovePlayer,
        FirstLegalMovePlayer.name: FirstLegalMovePlayer,
        SimpleTransformerPlayer.name: SimpleTransformerPlayer,
    }
