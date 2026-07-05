# Chess Transformer

This repository now targets one small local experiment:

- use the local `simple_chess_transformer/games.csv`
- turn SAN move lists into `FEN -> UCI move` supervised examples
- train a small local model just to confirm the pipeline works

This is intentionally simpler than the archived research scaffold in `storage/bootstrap_v1/`.

## What This Version Does

The active code path is a minimal pipeline:

1. Read `simple_chess_transformer/games.csv`
2. Convert a limited number of games into CSV examples
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

Prepare a small CSV dataset from the extracted file:

```bash
mkdir -p data/processed
python3 parse_lichess.py \
  --input simple_chess_transformer/games.csv \
  --max-games 5000 \
  --output data/processed/examples.csv
```

That writes CSV rows with:

- `white_elo`
- `black_elo`
- `time_control`
- `result`
- `fen_before`
- `uci_move`

The earlier move-sequence trainer is still present in `simple_chess_transformer/`, but it now also reads `simple_chess_transformer/games.csv`.

Build its dataset like this:

```bash
python3 main.py prepare-dataset \
  --input simple_chess_transformer/games.csv \
  --output-dir data/processed/sample \
  --max-games 5000
```

## Chess GUI

There is also a separate modular chess app in `modular_chess_gui/`.

It includes:

- a simple game engine
- pluggable player adapters
- human vs model or model vs model play
- multiple move notation styles for model input and move history
- illegal-move retry handling for model players

Launch it with:

```bash
uv run chess-gui
```

Built-in player types:

- `human`
- `random`
- `first-legal`
- `simple-transformer`

`simple-transformer` loads the checkpoint path from the GUI setup panel, defaulting to `runs/2m/model.pt`.
If it proposes illegal moves, the engine retries and logs each failed attempt to the terminal.

To plug in a new model later, implement `BasePlayer.get_move_text(...)` or wrap a callable with `CallableModelPlayer` in [modular_chess_gui/players.py](/home/js/research/chess-transformer/modular_chess_gui/players.py:1).

## Commands

- `prepare-dataset`: parse `games.csv` into tokenized move sequences for the baseline trainer
- `train`: train a small causal transformer on the earlier baseline
- `parse_lichess.py`: primary parser for `FEN -> UCI move` CSV generation

## Active Layout

```text
simple_chess_transformer/
  cli.py
  dataset.py
  games.csv
  train.py
parse_lichess.py
storage/bootstrap_v1/
  ... archived larger scaffold ...
```
