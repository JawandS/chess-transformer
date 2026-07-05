# AGENTS Notes

## Progress

- Initial research scaffold from the full style-asymmetry spec was moved to `storage/bootstrap_v1/`.
- Reason: the next step is a much simpler first experiment, so the larger scaffold is archived rather than kept active at the repo root.
- Root files were reset to a minimal placeholder state to avoid coupling the simple experiment to the larger plan too early.
- Active root now contains a minimal pipeline for downloading Lichess PGNs, preparing move-sequence datasets, and training a small local transformer.
- Added `parse_lichess.py` as the primary streaming parser for the simpler board-state supervision path: stdin PGN to CSV rows of `fen_before -> uci_move`.
- Extracted `chess.zip` into `simple_chess_transformer/games.csv`, removed the zip, and simplified the active code to read `games.csv` directly.
