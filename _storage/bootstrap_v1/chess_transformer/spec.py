"""Canonical experiment constants derived from the project spec."""

from __future__ import annotations

from dataclasses import dataclass


MODEL_SETUPS = (
    "shared_side_to_move",
    "shared_color_conditioned",
    "separate_color_models",
)

BASELINE_METRICS = (
    "human_move_matching_accuracy",
    "accuracy_by_color",
    "accuracy_by_rating_bucket",
    "legal_move_rate",
    "opening_diversity",
    "average_centipawn_loss",
    "blunder_rate",
    "aggression_score",
    "material_sacrifice_rate",
    "castling_timing",
    "king_safety",
    "piece_activity",
    "center_control",
)

RL_DENSE_REWARDS = (
    "stockfish_eval_improvement",
    "blunder_penalty",
    "legal_move_bonus",
    "checkmate_threat_bonus",
    "material_improvement",
    "king_safety_improvement",
)

STRENGTH_METRICS = (
    "elo_against_fixed_bots",
    "win_draw_loss_rate",
    "stockfish_low_depth_performance",
    "average_centipawn_loss",
    "blunder_mistake_inaccuracy_rate",
)

HUMAN_LIKENESS_METRICS = (
    "human_move_matching_accuracy",
    "move_matching_by_rating_bucket",
    "opening_similarity_to_human_games",
    "kl_divergence_from_supervised_policy",
    "style_drift_after_rl",
)

COLOR_ASYMMETRY_METRICS = (
    "aggression",
    "material_sacrifice_rate",
    "tactical_move_frequency",
    "defensive_move_frequency",
    "king_safety",
    "pawn_storm_frequency",
    "castling_timing",
    "center_control",
    "piece_activity",
    "opening_diversity",
    "willingness_to_enter_imbalanced_positions",
)

POLICY_BEHAVIOR_METRICS = (
    "policy_entropy",
    "top_k_move_diversity",
    "repeated_opening_lines",
    "exploitability_by_older_checkpoints",
    "robustness_against_unfamiliar_opponents",
)

REPRESENTATION_PROBES = (
    "board_state_reconstruction",
    "color_identity",
    "threat_detection",
    "material_balance",
    "king_safety",
    "tactical_motifs",
)

RESEARCH_QUESTIONS = (
    "Does White become more aggressive after self-play RL?",
    "Does Black become more defensive or counterattacking?",
    "Do both models collapse toward the same style?",
    "Does self-play reduce human-likeness?",
    "Do separate models overfit to each other?",
    "Does move diversity decrease?",
)


@dataclass(frozen=True)
class StageName:
    """Names of the top-level experimental stages."""

    supervised: str = "supervised_baseline"
    reinforcement_learning: str = "self_play_rl"
    evaluation: str = "evaluation"
    representation_analysis: str = "representation_analysis"
