"""Adapter for using the trained simple_chess_transformer checkpoint as a player."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from modular_chess_gui.notation import NotationStyle
from modular_chess_gui.players import BasePlayer, PlayerContext
from simple_chess_transformer.train import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    TrainConfig,
    build_causal_transformer,
)


@dataclass(frozen=True)
class TransformerCheckpoint:
    model: object
    vocab: list[str]
    token_to_id: dict[str, int]
    max_len: int
    bos_id: int
    eos_id: int
    pad_id: int
    unk_id: int
    device: str


def _lazy_import_torch():
    import torch
    import torch.nn as nn

    return torch, nn


def load_transformer_checkpoint(checkpoint_path: Path) -> TransformerCheckpoint:
    torch, nn = _lazy_import_torch()
    raw = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**raw["config"])
    vocab = raw["vocab"]
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    pad_id = token_to_id[PAD_TOKEN]
    bos_id = token_to_id[BOS_TOKEN]
    eos_id = token_to_id[EOS_TOKEN]
    unk_id = token_to_id[UNK_TOKEN]

    max_len = raw["model_state_dict"]["position_embedding.weight"].shape[0]
    model = build_causal_transformer(
        torch=torch,
        nn=nn,
        vocab_size=len(vocab),
        max_len=max_len,
        pad_id=pad_id,
        config=config,
    )
    model.load_state_dict(raw["model_state_dict"])
    model.eval()

    return TransformerCheckpoint(
        model=model,
        vocab=vocab,
        token_to_id=token_to_id,
        max_len=max_len,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        unk_id=unk_id,
        device="cpu",
    )


class SimpleTransformerPlayer(BasePlayer):
    name = "simple-transformer"

    def __init__(self, checkpoint_path: str = "runs/2m/model.pt") -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint = load_transformer_checkpoint(Path(checkpoint_path))

    def get_move_text(self, context: PlayerContext) -> str:
        torch, _ = _lazy_import_torch()
        move_ids = [self.checkpoint.bos_id]
        move_ids.extend(
            self.checkpoint.token_to_id.get(move, self.checkpoint.unk_id)
            for move in context.move_history[-(self.checkpoint.max_len - 2) :]
        )
        tensor = torch.tensor([move_ids], dtype=torch.long)
        with torch.no_grad():
            logits = self.checkpoint.model(tensor)
        next_token_logits = logits[0, -1]
        ranked_ids = torch.argsort(next_token_logits, descending=True).tolist()

        skip = context.retry_count
        for token_id in ranked_ids:
            token = self.checkpoint.vocab[token_id]
            if token in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN}:
                continue
            if skip > 0:
                skip -= 1
                continue
            if context.notation_style == NotationStyle.UCI:
                return token
            return token

        raise ValueError("transformer produced no candidate move")
