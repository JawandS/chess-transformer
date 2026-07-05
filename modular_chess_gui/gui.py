"""Simple Tkinter GUI for the modular chess engine."""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
import sys

import chess

from modular_chess_gui.engine import GameEngine
from modular_chess_gui.notation import NotationStyle, available_notation_styles
from modular_chess_gui.players import BasePlayer, player_registry
from modular_chess_gui.transformer_adapter import SimpleTransformerPlayer


PIECE_TO_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}

PIECE_TO_LABEL = {
    "P": "P",
    "N": "N",
    "B": "B",
    "R": "R",
    "Q": "Q",
    "K": "K",
    "p": "P",
    "n": "N",
    "b": "B",
    "r": "R",
    "q": "Q",
    "k": "K",
}

LIGHT_SQUARE = "#f3e7d0"
DARK_SQUARE = "#8c5e3c"
SELECTED_SQUARE = "#d7c45f"
TARGET_SQUARE = "#d08b5b"
BOARD_EDGE = "#1e1b18"
BOARD_SIZE = 720
SQUARE_SIZE = BOARD_SIZE // 8
WHITE_PIECE_FILL = "#fffaf0"
WHITE_PIECE_OUTLINE = "#6b5848"
BLACK_PIECE_FILL = "#1c1a18"
BLACK_PIECE_OUTLINE = "#efe5d7"
SYMBOL_FONT_CANDIDATES = [
    "Noto Chess",
    "FreeSerif",
    "DejaVu Sans",
    "Noto Sans Symbols 2",
    "Segoe UI Symbol",
    "Symbola",
]


class ChessGuiApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Chess GUI")
        self.root.configure(bg="#161311")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background="#161311")
        style.configure("Panel.TLabelframe", background="#221d19", foreground="#f4ede5")
        style.configure("Panel.TLabelframe.Label", background="#221d19", foreground="#f4ede5")
        style.configure("Panel.TLabel", background="#161311", foreground="#f4ede5")
        style.configure("Panel.TButton", padding=8)
        style.configure("Panel.TEntry", fieldbackground="#f6f2ec")

        registry = player_registry()
        self.player_options = list(registry.keys())
        self.player_factories = registry

        self.white_var = tk.StringVar(value="human")
        self.black_var = tk.StringVar(value="simple-transformer")
        self.model_notation_var = tk.StringVar(value=NotationStyle.UCI.value)
        self.history_notation_var = tk.StringVar(value=NotationStyle.SAN.value)
        self.transformer_checkpoint_var = tk.StringVar(value="runs/2m_v2_top/model.pt")
        self.human_input_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Configure players and start a game.")

        self.engine = self._build_engine()
        self.piece_text_map, self.piece_font_family, self.piece_font_size = self._select_piece_rendering()
        self.board_canvas: tk.Canvas | None = None
        self.history_widget: tk.Text | None = None
        self.log_widget: tk.Text | None = None

        self.selected_square: str | None = None
        self.drag_from_square: str | None = None
        self.drag_piece_symbol: str | None = None
        self.drag_piece_item: int | None = None
        self.highlight_targets: set[str] = set()

        self._build_layout()
        self._refresh_view()

    def _select_piece_rendering(self) -> tuple[dict[str, str], str, int]:
        available = set(tkfont.families(self.root))
        for family in SYMBOL_FONT_CANDIDATES:
            if family in available:
                print(f"[gui:info] using chess symbol font: {family}", file=sys.stderr)
                return PIECE_TO_UNICODE, family, 46
        keyword_matches = sorted(
            family
            for family in available
            if any(keyword in family.lower() for keyword in ("chess", "symbol", "symbola", "serif"))
        )
        for family in keyword_matches:
            print(f"[gui:info] using fallback symbol font candidate: {family}", file=sys.stderr)
            return PIECE_TO_UNICODE, family, 46
        print("[gui:info] no chess symbol font detected, falling back to letters", file=sys.stderr)
        return PIECE_TO_LABEL, "DejaVu Sans", 28

    def _build_engine(self) -> GameEngine:
        white = self._make_player(self.white_var.get())
        black = self._make_player(self.black_var.get())
        return GameEngine(
            white_player=white,
            black_player=black,
            model_notation=NotationStyle(self.model_notation_var.get()),
            history_notation=NotationStyle(self.history_notation_var.get()),
            cli_logging=True,
        )

    def _make_player(self, player_name: str) -> BasePlayer:
        if player_name == SimpleTransformerPlayer.name:
            return SimpleTransformerPlayer(self.transformer_checkpoint_var.get())
        return self.player_factories[player_name]()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        board_shell = ttk.Frame(self.root, style="App.TFrame", padding=18)
        board_shell.grid(row=0, column=0, sticky="nsew")
        side_shell = ttk.Frame(self.root, style="App.TFrame", padding=(0, 18, 18, 18))
        side_shell.grid(row=0, column=1, sticky="nsew")
        side_shell.columnconfigure(0, weight=1)
        side_shell.rowconfigure(4, weight=1)
        side_shell.rowconfigure(5, weight=1)

        title = ttk.Label(
            board_shell,
            text="Modular Chess",
            style="Panel.TLabel",
            font=("DejaVu Sans", 22, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        subtitle = ttk.Label(
            board_shell,
            textvariable=self.status_var,
            style="Panel.TLabel",
            font=("DejaVu Sans", 13),
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(0, 14))

        self.board_canvas = tk.Canvas(
            board_shell,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            bg=BOARD_EDGE,
            highlightthickness=0,
        )
        self.board_canvas.grid(row=2, column=0, sticky="nsew")
        self.board_canvas.bind("<Button-1>", self._on_board_press)
        self.board_canvas.bind("<B1-Motion>", self._on_board_drag)
        self.board_canvas.bind("<ButtonRelease-1>", self._on_board_release)

        setup_box = ttk.LabelFrame(side_shell, text="Game Setup", style="Panel.TLabelframe", padding=12)
        setup_box.grid(row=0, column=0, sticky="ew")
        setup_box.columnconfigure(1, weight=1)

        ttk.Label(setup_box, text="White", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(setup_box, self.white_var, self.white_var.get(), *self.player_options).grid(row=0, column=1, sticky="ew")
        ttk.Label(setup_box, text="Black", style="Panel.TLabel").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(setup_box, self.black_var, self.black_var.get(), *self.player_options).grid(row=1, column=1, sticky="ew")
        ttk.Label(setup_box, text="Model notation", style="Panel.TLabel").grid(row=2, column=0, sticky="w")
        ttk.OptionMenu(setup_box, self.model_notation_var, self.model_notation_var.get(), *available_notation_styles()).grid(row=2, column=1, sticky="ew")
        ttk.Label(setup_box, text="History notation", style="Panel.TLabel").grid(row=3, column=0, sticky="w")
        ttk.OptionMenu(setup_box, self.history_notation_var, self.history_notation_var.get(), *available_notation_styles()).grid(row=3, column=1, sticky="ew")
        ttk.Label(setup_box, text="Checkpoint", style="Panel.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Entry(setup_box, textvariable=self.transformer_checkpoint_var).grid(row=4, column=1, sticky="ew")

        actions = ttk.Frame(side_shell, style="App.TFrame")
        actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(actions, text="New Game", command=self.new_game, style="Panel.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(actions, text="Step", command=self.step_once, style="Panel.TButton").grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(actions, text="Auto Play", command=self.auto_play, style="Panel.TButton").grid(row=0, column=2, sticky="ew")

        human_box = ttk.LabelFrame(side_shell, text="Human Move", style="Panel.TLabelframe", padding=12)
        human_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        human_box.columnconfigure(0, weight=1)
        ttk.Label(
            human_box,
            text="Drag pieces on the board or type SAN/UCI here.",
            style="Panel.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Entry(human_box, textvariable=self.human_input_var).grid(row=1, column=0, sticky="ew")
        ttk.Button(human_box, text="Submit", command=self.submit_human_move, style="Panel.TButton").grid(row=1, column=1, padx=(8, 0))

        help_box = ttk.LabelFrame(side_shell, text="Notes", style="Panel.TLabelframe", padding=12)
        help_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            help_box,
            text="Illegal transformer moves are retried and logged to the terminal.",
            style="Panel.TLabel",
            wraplength=320,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        history_box = ttk.LabelFrame(side_shell, text="Move History", style="Panel.TLabelframe", padding=12)
        history_box.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        history_box.rowconfigure(0, weight=1)
        history_box.columnconfigure(0, weight=1)
        self.history_widget = tk.Text(
            history_box,
            height=16,
            width=34,
            font=("DejaVu Sans Mono", 11),
            bg="#faf6f0",
            fg="#1f1b18",
            relief="flat",
            state="disabled",
        )
        self.history_widget.grid(row=0, column=0, sticky="nsew")

        log_box = ttk.LabelFrame(side_shell, text="Engine Log", style="Panel.TLabelframe", padding=12)
        log_box.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        log_box.rowconfigure(0, weight=1)
        log_box.columnconfigure(0, weight=1)
        self.log_widget = tk.Text(
            log_box,
            height=10,
            width=34,
            font=("DejaVu Sans Mono", 10),
            bg="#faf6f0",
            fg="#1f1b18",
            relief="flat",
            state="disabled",
        )
        self.log_widget.grid(row=0, column=0, sticky="nsew")

    def new_game(self) -> None:
        self.engine = self._build_engine()
        self.engine.reset()
        self.human_input_var.set("")
        self._clear_selection()
        self._refresh_view()

    def submit_human_move(self) -> None:
        if not self.engine.is_human_turn():
            self.engine._emit_event("error", "It is not the human side's turn.")
            self._refresh_view()
            return
        try:
            self.engine.submit_human_move(
                self.human_input_var.get(),
                NotationStyle.AUTO,
            )
        except ValueError as exc:
            self.engine._emit_event("error", str(exc))
        self.human_input_var.set("")
        self._clear_selection()
        self._refresh_view()

    def step_once(self) -> None:
        if not self.engine.board.is_game_over() and not self.engine.is_human_turn():
            self.engine.step_model_turn()
        self._refresh_view()

    def auto_play(self) -> None:
        self.engine.model_notation = NotationStyle(self.model_notation_var.get())
        self.engine.history_notation = NotationStyle(self.history_notation_var.get())
        if self.engine.board.is_game_over() or self.engine.is_human_turn():
            self._refresh_view()
            return
        moved = self.engine.step_model_turn()
        self._refresh_view()
        if moved and not self.engine.board.is_game_over() and not self.engine.is_human_turn():
            self.root.after(120, self.auto_play)

    def _refresh_view(self) -> None:
        self.engine.model_notation = NotationStyle(self.model_notation_var.get())
        self.engine.history_notation = NotationStyle(self.history_notation_var.get())
        self.status_var.set(self.engine.status_text())
        self._draw_board()
        self._render_history()
        self._render_log()

    def _draw_board(self) -> None:
        if self.board_canvas is None:
            return
        self.board_canvas.delete("all")
        self.board_canvas.create_rectangle(
            0,
            0,
            BOARD_SIZE,
            BOARD_SIZE,
            fill=BOARD_EDGE,
            outline=BOARD_EDGE,
        )

        for rank_index in range(8):
            for file_index in range(8):
                square = chess.square(file_index, 7 - rank_index)
                square_name = chess.square_name(square)
                x1 = file_index * SQUARE_SIZE
                y1 = rank_index * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE

                if square_name == self.selected_square:
                    color = SELECTED_SQUARE
                elif square_name in self.highlight_targets:
                    color = TARGET_SQUARE
                else:
                    color = LIGHT_SQUARE if (rank_index + file_index) % 2 == 0 else DARK_SQUARE

                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

                if file_index == 0:
                    self.board_canvas.create_text(
                        x1 + 14,
                        y1 + 16,
                        text=str(8 - rank_index),
                        fill="#3a2f28",
                        font=("DejaVu Sans", 11, "bold"),
                    )
                if rank_index == 7:
                    self.board_canvas.create_text(
                        x2 - 14,
                        y2 - 14,
                        text=chr(ord("a") + file_index),
                        fill="#3a2f28",
                        font=("DejaVu Sans", 11, "bold"),
                    )

                piece = self.engine.board.piece_at(square)
                if piece is None:
                    continue
                if square_name == self.drag_from_square and self.drag_piece_item is not None:
                    continue
                self._draw_piece(
                    square_center_x=x1 + SQUARE_SIZE / 2,
                    square_center_y=y1 + SQUARE_SIZE / 2,
                    piece=piece,
                )

        if self.drag_piece_item is not None and self.drag_piece_symbol is not None:
            self.board_canvas.tag_raise(self.drag_piece_item)

    def _square_from_xy(self, x: int, y: int) -> str | None:
        if x < 0 or y < 0 or x >= BOARD_SIZE or y >= BOARD_SIZE:
            return None
        file_index = x // SQUARE_SIZE
        rank_index = y // SQUARE_SIZE
        square = chess.square(file_index, 7 - rank_index)
        return chess.square_name(square)

    def _on_board_press(self, event) -> None:
        if not self.engine.is_human_turn() or self.board_canvas is None:
            return
        square_name = self._square_from_xy(event.x, event.y)
        if square_name is None:
            return
        square = chess.parse_square(square_name)
        piece = self.engine.board.piece_at(square)
        if piece is None or piece.color != self.engine.board.turn:
            self._clear_selection()
            self._refresh_view()
            return

        self.selected_square = square_name
        self.drag_from_square = square_name
        self.drag_piece_symbol = piece.symbol()
        self.highlight_targets = {
            chess.square_name(move.to_square)
            for move in self.engine.board.legal_moves
            if move.from_square == square
        }
        self._refresh_view()
        self.drag_piece_item = self.board_canvas.create_text(
            event.x,
            event.y,
            text=self.piece_text_map[piece.symbol()],
            font=(self.piece_font_family, self.piece_font_size, "bold"),
            fill="#161311" if piece.color == chess.WHITE else "#fffaf0",
        )

    def _on_board_drag(self, event) -> None:
        if self.board_canvas is None or self.drag_piece_item is None:
            return
        self.board_canvas.coords(self.drag_piece_item, event.x, event.y)

    def _on_board_release(self, event) -> None:
        if self.drag_from_square is None:
            return
        target_square = self._square_from_xy(event.x, event.y)
        from_square = self.drag_from_square
        self._clear_drag_visual()

        if target_square is None:
            self._clear_selection()
            self._refresh_view()
            return
        if target_square == from_square:
            self._clear_selection()
            self._refresh_view()
            return

        move_text = from_square + target_square
        move = chess.Move.from_uci(move_text)
        if move not in self.engine.board.legal_moves:
            promotion_move = chess.Move.from_uci(move_text + "q")
            if promotion_move in self.engine.board.legal_moves:
                move_text = move_text + "q"

        try:
            self.engine.submit_human_move(move_text, NotationStyle.UCI)
        except ValueError as exc:
            self.engine._emit_event("error", str(exc))
        self._clear_selection()
        self._refresh_view()

    def _clear_drag_visual(self) -> None:
        if self.board_canvas is not None and self.drag_piece_item is not None:
            self.board_canvas.delete(self.drag_piece_item)
        self.drag_piece_item = None
        self.drag_piece_symbol = None
        self.drag_from_square = None

    def _clear_selection(self) -> None:
        self.selected_square = None
        self.highlight_targets.clear()
        self._clear_drag_visual()

    def _draw_piece(self, square_center_x: float, square_center_y: float, piece: chess.Piece) -> None:
        if self.board_canvas is None:
            return
        radius = SQUARE_SIZE * 0.33
        fill = WHITE_PIECE_FILL if piece.color == chess.WHITE else BLACK_PIECE_FILL
        outline = WHITE_PIECE_OUTLINE if piece.color == chess.WHITE else BLACK_PIECE_OUTLINE
        text_fill = "#161311" if piece.color == chess.WHITE else "#fffaf0"
        self.board_canvas.create_oval(
            square_center_x - radius,
            square_center_y - radius,
            square_center_x + radius,
            square_center_y + radius,
            fill=fill,
            outline=outline,
            width=3,
        )
        self.board_canvas.create_text(
            square_center_x,
            square_center_y,
            text=self.piece_text_map[piece.symbol()],
            font=(self.piece_font_family, self.piece_font_size, "bold"),
            fill=text_fill,
        )

    def _render_history(self) -> None:
        if self.history_widget is None:
            return
        lines = []
        style = self.engine.history_notation
        for record in self.engine.move_history:
            if style == NotationStyle.UCI:
                text = record.uci
            elif style == NotationStyle.LAN:
                text = record.lan
            else:
                text = record.san
            lines.append(f"{record.ply:>3}  {record.color:<5} {text}")
        self._set_text(self.history_widget, "\n".join(lines))

    def _render_log(self) -> None:
        if self.log_widget is None:
            return
        lines = [f"[{event.level}] {event.message}" for event in self.engine.events[-20:]]
        self._set_text(self.log_widget, "\n".join(lines))

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def run_gui() -> None:
    ChessGuiApp().run()
