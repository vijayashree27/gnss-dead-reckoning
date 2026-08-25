"""
INS Baseline vs AI Dead-Reckoning Comparison
==============================================
Glue script that ties together:
    - navigation.outage_simulator.simulate_gnss_outage()
    - navigation.ins_baseline.PureINSBaseline
    - navigation.evaluation.TrajectoryEvaluator / PlottingEngine

for a single reusable outage window, producing a side-by-side comparison
of the pure-INS baseline vs the AI Dead-Reckoning trajectory against the
same GPS ground truth.

CLI usage
---------
    python -m navigation.run_ins_vs_ai_comparison \
        --input data/ai_speed_predictions.csv \
        --duration 30 \
        --output-dir results/ins_vs_ai

    # Reproducible random outage window:
    python -m navigation.run_ins_vs_ai_comparison --duration 30 --seed 7

    # Explicit outage window:
    python -m navigation.run_ins_vs_ai_comparison --start-time 120 --duration 45

Programmatic usage
-------------------
    from navigation.run_ins_vs_ai_comparison import run_comparison
    result = run_comparison(input_csv="data/ai_speed_predictions.csv", duration_s=30)
    print(result.ins_metrics.calibrated_drift_percent_gps)
    print(result.ai_metrics.calibrated_drift_percent_gps)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .outage_simulator import simulate_gnss_outage, OutageWindow
    from .evaluation import TrajectoryEvaluator, EvaluationMetrics, PlottingEngine
except ImportError:  # allow running as a standalone script
    from outage_simulator import simulate_gnss_outage, OutageWindow
    from evaluation import TrajectoryEvaluator, EvaluationMetrics, PlottingEngine

DEFAULT_INPUT_CANDIDATES = [
    "data/ai_speed_predictions.csv",
    "data/ai_dead_reckoning_results.csv",
    "data/feature_dataset.csv",
]


@dataclass
class ComparisonResult:
    """Bundled output of an INS-baseline-vs-AI-DR comparison run."""
    window: OutageWindow
    ins_metrics: EvaluationMetrics
    ai_metrics: EvaluationMetrics
    ins_df: pd.DataFrame
    ai_df: pd.DataFrame
    output_dir: Path


def _resolve_default_input() -> str:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "No default input dataset found. Tried: "
        + ", ".join(DEFAULT_INPUT_CANDIDATES)
        + ". Pass --input explicitly."
    )


def _prepare_variant_csv(
    outage_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
) -> Path:
    """
    Writes a copy of `outage_df` with dr_x/dr_y and corrected_x/corrected_y
    both pointed at (x_col, y_col), so it can be evaluated in isolation by
    navigation.evaluation.TrajectoryEvaluator.
    """
    variant = outage_df.copy()
    variant["dr_x"] = variant[x_col]
    variant["dr_y"] = variant[y_col]
    variant["corrected_x"] = variant[x_col]
    variant["corrected_y"] = variant[y_col]
    variant.to_csv(out_path, index=False)
    return out_path


def _print_comparison_table(ins: EvaluationMetrics, ai: EvaluationMetrics) -> None:
    def pct_improve(ins_val: float, ai_val: float) -> str:
        if ins_val == 0:
            return "n/a"
        improvement = (ins_val - ai_val) / ins_val * 100.0
        return f"{improvement:+.1f}%"

    rows = [
        ("Final Position Error (m)", ins.calibrated_final_error_m, ai.calibrated_final_error_m),
        ("Mean Error / MAE (m)", ins.calibrated_mean_error_m, ai.calibrated_mean_error_m),
        ("Max Error (m)", ins.calibrated_max_error_m, ai.calibrated_max_error_m),
        ("RMSE (m)", ins.calibrated_rmse_m, ai.calibrated_rmse_m),
        ("CEP50 (m)", ins.calibrated_cep50_m, ai.calibrated_cep50_m),
        ("R95 (m)", ins.calibrated_r95_m, ai.calibrated_r95_m),
        ("Final Drift % (vs GPS dist)", ins.calibrated_drift_percent_gps, ai.calibrated_drift_percent_gps),
    ]

    print("=" * 78)
    print("      PURE INS BASELINE  vs  AI DEAD-RECKONING  (same GNSS outage)")
    print("=" * 78)
    print(f"{'Metric':<32}{'Pure INS':>14}{'AI DR':>14}{'AI Improvement':>18}")
    print("-" * 78)
    for label, ins_val, ai_val in rows:
        print(f"{label:<32}{ins_val:>14.3f}{ai_val:>14.3f}{pct_improve(ins_val, ai_val):>18}")
    print("=" * 78)
    print(f"Outage duration: {ins.outage_duration_sec:.2f} s | "
          f"GPS distance travelled: {ins.total_gps_distance_m:.1f} m | "
          f"Samples: {ins.total_samples}")
    print("=" * 78)


def _plot_trajectory_comparison(
    outage_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Custom 3-way trajectory plot: GPS ground truth vs Pure INS vs AI DR."""
    PlottingEngine.set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

    ax.plot(outage_df["gps_x"], outage_df["gps_y"], label="GPS Ground Truth",
            color="#2563EB", linewidth=2.8, zorder=3)
    ax.plot(outage_df["ins_x"], outage_df["ins_y"], label="Pure INS Baseline",
            color="#F59E0B", linewidth=2.0, linestyle="--", zorder=2)
    ax.plot(outage_df["dr_x"], outage_df["dr_y"], label="AI Dead-Reckoning",
            color="#10B981", linewidth=2.4, zorder=4)

    ax.scatter([outage_df["gps_x"].iloc[0]], [outage_df["gps_y"].iloc[0]],
               color="#0F172A", s=110, marker="o", zorder=6, label="Outage Start")
    ax.scatter([outage_df["gps_x"].iloc[-1]], [outage_df["gps_y"].iloc[-1]],
               color="#2563EB", s=100, marker="s", zorder=6, label="GPS End")
    ax.scatter([outage_df["ins_x"].iloc[-1]], [outage_df["ins_y"].iloc[-1]],
               color="#F59E0B", s=90, marker="^", zorder=6, label="Pure INS End")
    ax.scatter([outage_df["dr_x"].iloc[-1]], [outage_df["dr_y"].iloc[-1]],
               color="#10B981", s=90, marker="D", zorder=6, label="AI DR End")

    ax.set_title("Pure INS vs AI Dead-Reckoning (GNSS Outage Window)", fontweight="bold", pad=12)
    ax.set_xlabel("Local X Coordinate (meters)", fontweight="semibold")
    ax.set_ylabel("Local Y Coordinate (meters)", fontweight="semibold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", frameon=True, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_error_comparison(
    ins_df: pd.DataFrame,
    ai_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Custom position-error-over-time plot: Pure INS vs AI DR."""
    PlottingEngine.set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    t = ins_df["time_rel_sec"]
    ax.plot(t, ins_df["eval_calibrated_position_error_m"], label="Pure INS Baseline",
            color="#F59E0B", linewidth=2.2, linestyle="--")
    ax.plot(t, ai_df["eval_calibrated_position_error_m"], label="AI Dead-Reckoning",
            color="#10B981", linewidth=2.5)
    ax.axhline(0, color="#64748B", linewidth=0.8)

    ax.set_title("Position Error Over Time: Pure INS vs AI Dead-Reckoning", fontweight="bold", pad=12)
    ax.set_xlabel("Time in GNSS Outage (seconds)", fontweight="semibold")
    ax.set_ylabel("Euclidean Position Error (meters)", fontweight="semibold")
    ax.set_xlim(0, t.iloc[-1] * 1.02)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_comparison(
    input_csv: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "results/ins_vs_ai",
    start_idx: Optional[int] = None,
    start_time_s: Optional[float] = None,
    duration_s: float = 30.0,
    ai_speed_col: str = "predicted_speed",
    seed: Optional[int] = None,
    generate_plots: bool = True,
) -> ComparisonResult:
    """
    Simulates a single GNSS outage window and evaluates both a pure-INS
    baseline and the AI Dead-Reckoning trajectory against GPS ground
    truth, saving a comparison table (JSON) and comparison plots.
    """
    if input_csv is None:
        input_csv = _resolve_default_input()
    input_csv = Path(input_csv)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    outage_df, full_df, window = simulate_gnss_outage(
        df,
        start_idx=start_idx,
        start_time_s=start_time_s,
        duration_s=duration_s,
        ai_speed_col=ai_speed_col,
        seed=seed,
    )

    ins_csv = out_dir / "ins_baseline_input.csv"
    ai_csv = out_dir / "ai_dr_input.csv"
    _prepare_variant_csv(outage_df, "ins_x", "ins_y", ins_csv)
    _prepare_variant_csv(outage_df, "dr_x", "dr_y", ai_csv)

    ins_df, ins_metrics = TrajectoryEvaluator(ins_csv).evaluate()
    ai_df, ai_metrics = TrajectoryEvaluator(ai_csv).evaluate()

    _print_comparison_table(ins_metrics, ai_metrics)

    comparison_summary = {
        "window": asdict(window),
        "input_csv": str(input_csv),
        "pure_ins": asdict(ins_metrics),
        "ai_dead_reckoning": asdict(ai_metrics),
    }
    with open(out_dir / "ins_vs_ai_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    if generate_plots:
        _plot_trajectory_comparison(outage_df, out_dir / "ins_vs_ai_trajectory.png")
        _plot_error_comparison(ins_df, ai_df, out_dir / "ins_vs_ai_position_error.png")

    print(f"\nSaved comparison summary to: {out_dir / 'ins_vs_ai_comparison.json'}")
    if generate_plots:
        print(f"Saved plots to: {out_dir}/ins_vs_ai_trajectory.png, "
              f"{out_dir}/ins_vs_ai_position_error.png")

    return ComparisonResult(
        window=window,
        ins_metrics=ins_metrics,
        ai_metrics=ai_metrics,
        ins_df=ins_df,
        ai_df=ai_df,
        output_dir=out_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare a Pure INS baseline against AI Dead-Reckoning over a simulated GNSS outage."
    )
    parser.add_argument("--input", "-i", default=None,
                         help="Path to input CSV (feature_dataset.csv / ai_speed_predictions.csv). "
                              "Defaults to the first available file among the project's known data files.")
    parser.add_argument("--output-dir", "-o", default="results/ins_vs_ai",
                         help="Directory to save comparison CSVs, JSON summary, and plots.")
    parser.add_argument("--start-idx", type=int, default=None, help="Explicit outage start row index.")
    parser.add_argument("--start-time", type=float, default=None,
                         help="Outage start time in seconds relative to trip start.")
    parser.add_argument("--duration", type=float, default=30.0, help="Outage duration in seconds.")
    parser.add_argument("--ai-speed-col", default="predicted_speed",
                         help="Column holding the AI-predicted speed (falls back to vehicle_speed if absent).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for picking the outage window when no start is given.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")

    args = parser.parse_args()

    run_comparison(
        input_csv=args.input,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        start_time_s=args.start_time,
        duration_s=args.duration,
        ai_speed_col=args.ai_speed_col,
        seed=args.seed,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()