# Chess Transformer

This repository now targets one small local experiment:

- fetch a chunk of Lichess games
- stream PGN into `FEN -> UCI move` supervised examples
- train a small local model just to confirm the pipeline works

This is intentionally simpler than the archived research scaffold in `storage/bootstrap_v1/`.

## What This Version Does

The active code path is a minimal pipeline:

1. Download a Lichess PGN or PGN.ZST file
2. Stream a limited number of games into CSV examples
3. Train a first local model on those examples

The first supervision format is:

- input: `fen_before`
- target: `uci_move`

This version does not yet include a board-state model trainer, Stockfish evaluation, or self-play RL.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Example Workflow

Download a Lichess export or database chunk:

```bash
python3 main.py download-lichess \
  --url https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst \
  --output data/raw/lichess_sample.pgn.zst
```

Prepare a small CSV dataset by streaming the archive:

```bash
mkdir -p data/processed
zstdcat lichess_db_standard_rated_2026-06.pgn.zst \
  | python3 parse_lichess.py --max-games 5000 \
  > data/processed/examples.csv
```

That writes CSV rows with:

- `white_elo`
- `black_elo`
- `time_control`
- `result`
- `fen_before`
- `uci_move`

The earlier move-sequence CLI is still present in `simple_chess_transformer/` if you want a baseline trainer while the board-state trainer is still being built.

## Commands

- `download-lichess`: download a PGN or `.pgn.zst` file to disk
- `prepare-dataset`: parse PGNs into tokenized move sequences for the earlier baseline
- `train`: train a small causal transformer on the earlier baseline
- `parse_lichess.py`: primary streaming parser for `FEN -> UCI move` CSV generation

## Active Layout

```text
simple_chess_transformer/
  cli.py
  lichess.py
  dataset.py
  train.py
parse_lichess.py
storage/bootstrap_v1/
  ... archived larger scaffold ...
```
