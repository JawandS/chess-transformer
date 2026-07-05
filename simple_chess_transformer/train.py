"""Small causal transformer for move-sequence prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    max_batches_per_epoch: int | None = None
    device: str = "auto"
    seed: int = 0


def _lazy_import_torch():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    return torch, nn, DataLoader, Dataset


def _load_dataset(dataset_dir: Path) -> tuple[list[list[str]], list[str]]:
    games_path = dataset_dir / "games.jsonl"
    vocab_path = dataset_dir / "vocab.json"
    games = []
    for line in games_path.read_text(encoding="utf-8").splitlines():
        games.append(json.loads(line)["moves"])
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    return games, vocab


def train_model(config: TrainConfig) -> dict:
    torch, nn, DataLoader, Dataset = _lazy_import_torch()

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games, vocab = _load_dataset(dataset_dir)
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    pad_id = token_to_id[PAD_TOKEN]
    bos_id = token_to_id[BOS_TOKEN]
    eos_id = token_to_id[EOS_TOKEN]
    unk_id = token_to_id[UNK_TOKEN]

    encoded_sequences = []
    for moves in games:
        seq = [bos_id]
        seq.extend(token_to_id.get(move, unk_id) for move in moves)
        seq.append(eos_id)
        encoded_sequences.append(seq)

    max_len = max(len(seq) for seq in encoded_sequences)

    class MoveDataset(Dataset):
        def __len__(self) -> int:
            return len(encoded_sequences)

        def __getitem__(self, index: int):
            seq = encoded_sequences[index]
            x = seq[:-1]
            y = seq[1:]
            return x, y

    def collate(batch):
        xs, ys = zip(*batch)
        max_batch_len = max(len(x) for x in xs)
        x_tensor = torch.full((len(xs), max_batch_len), pad_id, dtype=torch.long)
        y_tensor = torch.full((len(xs), max_batch_len), pad_id, dtype=torch.long)
        for row, (x, y) in enumerate(zip(xs, ys)):
            x_tensor[row, : len(x)] = torch.tensor(x, dtype=torch.long)
            y_tensor[row, : len(y)] = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

    class CausalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(len(vocab), config.embedding_dim)
            self.position_embedding = nn.Embedding(max_len, config.embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.num_layers,
            )
            self.norm = nn.LayerNorm(config.embedding_dim)
            self.head = nn.Linear(config.embedding_dim, len(vocab))

        def forward(self, tokens):
            batch_size, seq_len = tokens.shape
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
            x = self.token_embedding(tokens) + self.position_embedding(positions)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=tokens.device),
                diagonal=1,
            )
            padding_mask = tokens.eq(pad_id)
            x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
            x = self.norm(x)
            return self.head(x)

    device = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )

    model = CausalTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    data_loader = DataLoader(
        MoveDataset(),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    history = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        batches_seen = 0

        for x, y in data_loader:
            batches_seen += 1
            if config.max_batches_per_epoch and batches_seen > config.max_batches_per_epoch:
                break

            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            non_pad = y.ne(pad_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

        average_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(average_loss) if average_loss < 20 else float("inf")
        history.append(
            {
                "epoch": epoch + 1,
                "loss": average_loss,
                "perplexity": perplexity,
                "tokens": total_tokens,
            }
        )

    checkpoint_path = output_dir / "model.pt"
    summary_path = output_dir / "train_summary.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": asdict(config),
        },
        checkpoint_path,
    )
    summary = {
        "device": device,
        "num_games": len(games),
        "vocab_size": len(vocab),
        "max_sequence_length": max_len,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
