"""Project bootstrap and high-level stage planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chess_transformer.config import ExperimentConfig, default_experiment_config
from chess_transformer.spec import RESEARCH_QUESTIONS, StageName


@dataclass(frozen=True)
class StagePlan:
    name: str
    summary: str
    outputs: tuple[str, ...]


def default_stage_plan() -> tuple[StagePlan, ...]:
    stage_names = StageName()
    return (
        StagePlan(
            name=stage_names.supervised,
            summary="Train supervised human-move baselines for all three model setups.",
            outputs=(
                "processed training dataset",
                "baseline checkpoints",
                "baseline metric reports",
            ),
        ),
        StagePlan(
            name=stage_names.reinforcement_learning,
            summary="Fine-tune the supervised checkpoints through constrained self-play.",
            outputs=(
                "self-play games",
                "RL checkpoints",
                "training reward summaries",
            ),
        ),
        StagePlan(
            name=stage_names.evaluation,
            summary="Compare strength, human-likeness, and color asymmetry before and after RL.",
            outputs=(
                "evaluation tables",
                "style asymmetry comparisons",
                "supervised-vs-RL drift summaries",
            ),
        ),
        StagePlan(
            name=stage_names.representation_analysis,
            summary="Optionally probe hidden representations for tactical and color-sensitive features.",
            outputs=("probe datasets", "probe metrics", "representation notes"),
        ),
    )


def bootstrap_project(root: Path, config: ExperimentConfig | None = None) -> Path:
    experiment = config or default_experiment_config()
    for relative_path in (
        "artifacts",
        "artifacts/checkpoints",
        "artifacts/evaluations",
        "artifacts/reports",
        "configs",
        "data/raw",
        "data/processed",
    ):
        (root / relative_path).mkdir(parents=True, exist_ok=True)

    config_path = root / "configs" / "default_experiment.json"
    experiment.write_json(config_path)
    return config_path


def render_plan(config: ExperimentConfig | None = None) -> str:
    experiment = config or default_experiment_config()
    lines = [
        f"Project: {experiment.project_name}",
        "",
        "Goal:",
        f"  {experiment.goal}",
        "",
        "Hypothesis:",
        f"  {experiment.hypothesis}",
        "",
        "Model setups:",
    ]
    for model in experiment.model_setups:
        lines.append(f"  - {model.name}: {model.description}")

    lines.extend(["", "Stages:"])
    for stage in default_stage_plan():
        lines.append(f"  - {stage.name}: {stage.summary}")

    lines.extend(["", "Research questions:"])
    for question in RESEARCH_QUESTIONS:
        lines.append(f"  - {question}")

    return "\n".join(lines)
