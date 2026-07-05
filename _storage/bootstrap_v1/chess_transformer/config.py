"""Typed experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from chess_transformer.spec import (
    BASELINE_METRICS,
    COLOR_ASYMMETRY_METRICS,
    HUMAN_LIKENESS_METRICS,
    MODEL_SETUPS,
    POLICY_BEHAVIOR_METRICS,
    RL_DENSE_REWARDS,
    STRENGTH_METRICS,
)


@dataclass
class DatasetConfig:
    source: str = "lichess"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    rating_buckets: list[str] = field(
        default_factory=lambda: ["<1600", "1600-2000", "2000-2400", "2400+"]
    )
    max_games: int = 100_000


@dataclass
class ModelConfig:
    name: str
    description: str
    vocab_strategy: str
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    max_sequence_length: int = 256
    legal_move_masking: bool = True


@dataclass
class SupervisedConfig:
    objective: str = "predict_human_move"
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 3e-4
    metrics: list[str] = field(default_factory=lambda: list(BASELINE_METRICS))


@dataclass
class RLConfig:
    opponent_schedule: str = "self_play_only"
    reward_win: float = 1.0
    reward_draw: float = 0.0
    reward_loss: float = -1.0
    use_dense_rewards: bool = False
    dense_rewards: list[str] = field(default_factory=lambda: list(RL_DENSE_REWARDS))
    episodes_per_iteration: int = 512
    num_iterations: int = 20


@dataclass
class EvaluationConfig:
    strength_metrics: list[str] = field(default_factory=lambda: list(STRENGTH_METRICS))
    human_likeness_metrics: list[str] = field(
        default_factory=lambda: list(HUMAN_LIKENESS_METRICS)
    )
    color_asymmetry_metrics: list[str] = field(
        default_factory=lambda: list(COLOR_ASYMMETRY_METRICS)
    )
    policy_behavior_metrics: list[str] = field(
        default_factory=lambda: list(POLICY_BEHAVIOR_METRICS)
    )
    stockfish_depth: int = 8
    fixed_eval_positions: int = 256


@dataclass
class ExperimentConfig:
    project_name: str
    goal: str
    hypothesis: str
    model_setups: list[ModelConfig]
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)
    reinforcement_learning: RLConfig = field(default_factory=RLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        project_name="style-asymmetry-self-play-chess-transformers",
        goal=(
            "Measure whether self-play RL changes strength, human-likeness, "
            "and White/Black strategic asymmetry in chess transformers."
        ),
        hypothesis=(
            "White-specialized models may become more initiative-seeking while "
            "Black-specialized models may become more defensive or counterattacking."
        ),
        model_setups=[
            ModelConfig(
                name=MODEL_SETUPS[0],
                description="Single transformer using side-to-move from the board state.",
                vocab_strategy="board_plus_side_to_move",
            ),
            ModelConfig(
                name=MODEL_SETUPS[1],
                description="Single transformer with an explicit color token.",
                vocab_strategy="color_token_plus_board",
            ),
            ModelConfig(
                name=MODEL_SETUPS[2],
                description="Independent White and Black transformer policies.",
                vocab_strategy="board_only_per_color_model",
            ),
        ],
    )
