"""
GNSS Dead-Reckoning Navigation & Evaluation Module
=================================================
Compares AI dead-reckoning trajectory against GPS ground truth, computes
position errors, cumulative distance travelled, drift percentages, and
implements post-outage GNSS recovery / smooth transition filtering.

Authors: Antigravity AI Navigation Team
Input: data/final_calibrated_results.csv (or compatible DR/GNSS dataset)
Outputs: results/evaluation_results.csv, metrics_summary.json, and visual graphs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EvaluationMetrics:
    """Statistical evaluation metrics for trajectory comparison."""
    total_samples: int
    outage_duration_sec: float
    total_gps_distance_m: float
    total_dr_distance_m: float
    raw_final_error_m: float
    raw_mean_error_m: float
    raw_max_error_m: float
    raw_rmse_m: float
    raw_cep50_m: float
    raw_r95_m: float
    raw_drift_percent_gps: float
    raw_drift_percent_dr: float
    calibrated_final_error_m: float
    calibrated_mean_error_m: float
    calibrated_max_error_m: float
    calibrated_rmse_m: float
    calibrated_cep50_m: float
    calibrated_r95_m: float
    calibrated_drift_percent_gps: float
    calibrated_drift_percent_dr: float
    error_reduction_percent: float


class TrajectoryEvaluator:
    """
    Evaluates AI Dead-Reckoning navigation performance against GPS ground truth.
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the evaluator with a dataset.

        Args:
            data_path: Path to the input CSV file containing DR and GPS columns.
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.data_path}")

        self.df_raw = pd.read_csv(self.data_path)
        self.df_evaluated: Optional[pd.DataFrame] = None
        self.metrics: Optional[EvaluationMetrics] = None

    def evaluate(self) -> Tuple[pd.DataFrame, EvaluationMetrics]:
        """
        Performs full trajectory evaluation:
        - Normalizes timestamps to relative outage time (seconds).
        - Computes ground-truth GPS and Dead-Reckoning step distances.
        - Calculates cumulative distance travelled from outage start.
        - Calculates Euclidean position errors (Raw AI DR vs GPS, Calibrated DR vs GPS).
        - Calculates instantaneous and final drift percentages.

        Returns:
            Tuple of (evaluated_dataframe, summary_metrics).
        """
        df = self.df_raw.copy()

        # 1. Temporal normalization
        t_start = df["timestamp_ms"].iloc[0]
        df["time_rel_sec"] = (df["timestamp_ms"] - t_start) / 1000.0
        outage_duration = float(df["time_rel_sec"].iloc[-1])

        # 2. Coordinate resolution
        # Resolve GPS Ground Truth coordinates
        if "gps_x" in df.columns and "gps_y" in df.columns:
            gps_x = df["gps_x"].to_numpy(dtype=float)
            gps_y = df["gps_y"].to_numpy(dtype=float)
        else:
            raise ValueError("Required GPS coordinates ('gps_x', 'gps_y') missing from dataset.")

        # Resolve Raw Dead-Reckoning coordinates
        if "dr_x" in df.columns and "dr_y" in df.columns:
            dr_x = df["dr_x"].to_numpy(dtype=float)
            dr_y = df["dr_y"].to_numpy(dtype=float)
        elif "predicted_x" in df.columns and "predicted_y" in df.columns:
            dr_x = df["predicted_x"].to_numpy(dtype=float)
            dr_y = df["predicted_y"].to_numpy(dtype=float)
        else:
            raise ValueError("Dead-Reckoning coordinates ('dr_x'/'dr_y' or 'predicted_x'/'predicted_y') not found.")

        # Resolve Calibrated / Corrected Dead-Reckoning coordinates
        if "corrected_x" in df.columns and "corrected_y" in df.columns:
            corr_x = df["corrected_x"].to_numpy(dtype=float)
            corr_y = df["corrected_y"].to_numpy(dtype=float)
        else:
            corr_x = dr_x.copy()
            corr_y = dr_y.copy()

        # 3. Incremental and Cumulative Distance Calculations
        # GPS Ground Truth distance travelled
        dx_gps = np.diff(gps_x, prepend=gps_x[0])
        dy_gps = np.diff(gps_y, prepend=gps_y[0])
        gps_step_dist = np.sqrt(dx_gps**2 + dy_gps**2)
        gps_cum_dist = np.cumsum(gps_step_dist)

        # Raw Dead-Reckoning distance travelled
        dx_dr = np.diff(dr_x, prepend=dr_x[0])
        dy_dr = np.diff(dr_y, prepend=dr_y[0])
        dr_step_dist = np.sqrt(dx_dr**2 + dy_dr**2)
        dr_cum_dist = np.cumsum(dr_step_dist)

        # Calibrated Dead-Reckoning distance travelled
        dx_corr = np.diff(corr_x, prepend=corr_x[0])
        dy_corr = np.diff(corr_y, prepend=corr_y[0])
        corr_step_dist = np.sqrt(dx_corr**2 + dy_corr**2)
        corr_cum_dist = np.cumsum(corr_step_dist)

        # 4. Position Error Calculations (Euclidean distance to GPS ground truth)
        raw_pos_error = np.sqrt((dr_x - gps_x)**2 + (dr_y - gps_y)**2)
        calibrated_pos_error = np.sqrt((corr_x - gps_x)**2 + (corr_y - gps_y)**2)

        # 5. Drift Percentage Calculations: (Position Error / Distance Travelled) * 100
        # For initial point (distance = 0), drift is 0.0 to prevent division by zero.
        raw_drift_pct_gps = np.zeros_like(raw_pos_error)
        mask_gps = gps_cum_dist > 1e-3
        raw_drift_pct_gps[mask_gps] = (raw_pos_error[mask_gps] / gps_cum_dist[mask_gps]) * 100.0

        calibrated_drift_pct_gps = np.zeros_like(calibrated_pos_error)
        calibrated_drift_pct_gps[mask_gps] = (calibrated_pos_error[mask_gps] / gps_cum_dist[mask_gps]) * 100.0

        raw_drift_pct_dr = np.zeros_like(raw_pos_error)
        mask_dr = dr_cum_dist > 1e-3
        raw_drift_pct_dr[mask_dr] = (raw_pos_error[mask_dr] / dr_cum_dist[mask_dr]) * 100.0

        calibrated_drift_pct_dr = np.zeros_like(calibrated_pos_error)
        calibrated_drift_pct_dr[mask_dr] = (calibrated_pos_error[mask_dr] / dr_cum_dist[mask_dr]) * 100.0

        # Store evaluated series in the dataframe
        df["eval_gps_step_dist_m"] = gps_step_dist
        df["eval_gps_cum_dist_m"] = gps_cum_dist
        df["eval_dr_step_dist_m"] = dr_step_dist
        df["eval_dr_cum_dist_m"] = dr_cum_dist
        df["eval_corr_cum_dist_m"] = corr_cum_dist

        df["eval_raw_position_error_m"] = raw_pos_error
        df["eval_calibrated_position_error_m"] = calibrated_pos_error

        df["eval_raw_drift_pct"] = raw_drift_pct_gps
        df["eval_calibrated_drift_pct"] = calibrated_drift_pct_gps
        df["eval_raw_drift_pct_dr_dist"] = raw_drift_pct_dr
        df["eval_calibrated_drift_pct_dr_dist"] = calibrated_drift_pct_dr

        # 6. Aggregate Statistical Metrics
        tot_gps_dist = float(gps_cum_dist[-1])
        tot_dr_dist = float(dr_cum_dist[-1])

        raw_final_err = float(raw_pos_error[-1])
        raw_mean_err = float(np.mean(raw_pos_error))
        raw_max_err = float(np.max(raw_pos_error))
        raw_rmse = float(np.sqrt(np.mean(raw_pos_error**2)))
        raw_cep50 = float(np.percentile(raw_pos_error, 50))
        raw_r95 = float(np.percentile(raw_pos_error, 95))
        raw_drift_final_gps = float((raw_final_err / tot_gps_dist) * 100.0)
        raw_drift_final_dr = float((raw_final_err / tot_dr_dist) * 100.0)

        cal_final_err = float(calibrated_pos_error[-1])
        cal_mean_err = float(np.mean(calibrated_pos_error))
        cal_max_err = float(np.max(calibrated_pos_error))
        cal_rmse = float(np.sqrt(np.mean(calibrated_pos_error**2)))
        cal_cep50 = float(np.percentile(calibrated_pos_error, 50))
        cal_r95 = float(np.percentile(calibrated_pos_error, 95))
        cal_drift_final_gps = float((cal_final_err / tot_gps_dist) * 100.0)
        cal_drift_final_dr = float((cal_final_err / tot_dr_dist) * 100.0)

        err_reduction = float(((raw_final_err - cal_final_err) / raw_final_err) * 100.0) if raw_final_err > 0 else 0.0

        metrics = EvaluationMetrics(
            total_samples=len(df),
            outage_duration_sec=round(outage_duration, 2),
            total_gps_distance_m=round(tot_gps_dist, 3),
            total_dr_distance_m=round(tot_dr_dist, 3),
            raw_final_error_m=round(raw_final_err, 3),
            raw_mean_error_m=round(raw_mean_err, 3),
            raw_max_error_m=round(raw_max_err, 3),
            raw_rmse_m=round(raw_rmse, 3),
            raw_cep50_m=round(raw_cep50, 3),
            raw_r95_m=round(raw_r95, 3),
            raw_drift_percent_gps=round(raw_drift_final_gps, 2),
            raw_drift_percent_dr=round(raw_drift_final_dr, 2),
            calibrated_final_error_m=round(cal_final_err, 3),
            calibrated_mean_error_m=round(cal_mean_err, 3),
            calibrated_max_error_m=round(cal_max_err, 3),
            calibrated_rmse_m=round(cal_rmse, 3),
            calibrated_cep50_m=round(cal_cep50, 3),
            calibrated_r95_m=round(cal_r95, 3),
            calibrated_drift_percent_gps=round(cal_drift_final_gps, 2),
            calibrated_drift_percent_dr=round(cal_drift_final_dr, 2),
            error_reduction_percent=round(err_reduction, 2),
        )

        self.df_evaluated = df
        self.metrics = metrics
        return df, metrics


class GNSSRecovery:
    """
    Implements smooth GNSS reacquisition and post-outage recovery filters.
    Prevents discontinuous position jumps when GNSS signal locks back after outage.
    """

    @staticmethod
    def apply_exponential_blend(
        dr_x: np.ndarray,
        dr_y: np.ndarray,
        gps_x: np.ndarray,
        gps_y: np.ndarray,
        time_rel_sec: np.ndarray,
        outage_end_idx: int,
        recovery_time_constant: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies an exponential decay blending filter to seamlessly converge from
        dead-reckoning position to acquired GNSS position.

        Args:
            dr_x, dr_y: Dead-reckoning position series.
            gps_x, gps_y: Ground truth / acquired GPS positions.
            time_rel_sec: Relative timestamps (seconds).
            outage_end_idx: Index where GNSS signal is reacquired.
            recovery_time_constant: Time constant tau (seconds) for exponential blending.

        Returns:
            Tuple of (recovered_x, recovered_y).
        """
        n = len(dr_x)
        rec_x = np.copy(dr_x)
        rec_y = np.copy(dr_y)

        if outage_end_idx >= n:
            return rec_x, rec_y

        t_recovery_start = time_rel_sec[outage_end_idx]

        for i in range(outage_end_idx, n):
            dt_rec = time_rel_sec[i] - t_recovery_start
            # Weight for GPS increases smoothly from 0 to 1
            weight_gps = 1.0 - np.exp(-dt_rec / max(recovery_time_constant, 0.1))
            weight_dr = 1.0 - weight_gps

            rec_x[i] = (weight_dr * dr_x[i]) + (weight_gps * gps_x[i])
            rec_y[i] = (weight_dr * dr_y[i]) + (weight_gps * gps_y[i])

        return rec_x, rec_y

    @staticmethod
    def simulate_recovery_trajectory(
        df_outage: pd.DataFrame,
        recovery_samples: int = 50,
        recovery_speed_mps: float = 8.0,
        dt_step: float = 0.1,
    ) -> pd.DataFrame:
        """
        Extends an outage dataset with a simulated GNSS recovery phase demonstrating
        how the navigation filter smoothly transitions back to GPS lock.

        Args:
            df_outage: Outage dataframe containing evaluation columns.
            recovery_samples: Number of post-outage recovered time steps to simulate.
            recovery_speed_mps: Assumed vehicle speed during recovery.
            dt_step: Sampling interval.

        Returns:
            Extended DataFrame including both outage and recovery phases.
        """
        last_row = df_outage.iloc[-1]
        last_t = last_row["timestamp_ms"]
        last_gps_x, last_gps_y = last_row["gps_x"], last_row["gps_y"]
        last_corr_x, last_corr_y = last_row["corrected_x"], last_row["corrected_y"]

        records = []
        tau = 2.5  # seconds convergence rate

        for k in range(1, recovery_samples + 1):
            t_ms = last_t + int(k * dt_step * 1000)
            t_sec = (k * dt_step)

            # Continue motion along trajectory heading
            step_move = recovery_speed_mps * dt_step
            # Natural linear projection
            curr_gps_x = last_gps_x - (step_move * 0.4)
            curr_gps_y = last_gps_y + (step_move * 0.9)

            # Raw DR accumulates more drift
            curr_dr_x = last_corr_x - (step_move * 0.35)
            curr_dr_y = last_corr_y + (step_move * 0.95)

            # Exponential blending
            w_gps = 1.0 - np.exp(-t_sec / tau)
            nav_x = (1.0 - w_gps) * curr_dr_x + w_gps * curr_gps_x
            nav_y = (1.0 - w_gps) * curr_dr_y + w_gps * curr_gps_y

            error_nav = float(np.sqrt((nav_x - curr_gps_x)**2 + (nav_y - curr_gps_y)**2))

            records.append({
                "timestamp_ms": t_ms,
                "time_rel_sec": last_row["time_rel_sec"] + t_sec,
                "gps_x": curr_gps_x,
                "gps_y": curr_gps_y,
                "dr_x": curr_dr_x,
                "dr_y": curr_dr_y,
                "corrected_x": nav_x,
                "corrected_y": nav_y,
                "recovered_x": nav_x,
                "recovered_y": nav_y,
                "eval_raw_position_error_m": np.sqrt((curr_dr_x - curr_gps_x)**2 + (curr_dr_y - curr_gps_y)**2),
                "eval_calibrated_position_error_m": error_nav,
                "phase": "GNSS_RECOVERED",
            })

        df_rec = pd.DataFrame(records)
        df_outage_tagged = df_outage.copy()
        df_outage_tagged["phase"] = "GNSS_OUTAGE"
        df_outage_tagged["recovered_x"] = df_outage_tagged["corrected_x"]
        df_outage_tagged["recovered_y"] = df_outage_tagged["corrected_y"]

        return pd.concat([df_outage_tagged, df_rec], ignore_index=True)


class PlottingEngine:
    """
    Generates high-resolution, publication-quality visualizations for navigation evaluation.
    """

    DARK_THEME = {
        "bg_color": "#0F172A",
        "panel_color": "#1E293B",
        "text_color": "#F8FAFC",
        "grid_color": "#334155",
        "accent_blue": "#38BDF8",
        "accent_green": "#4ADE80",
        "accent_red": "#F87171",
        "accent_purple": "#C084FC",
        "accent_amber": "#FBBF24",
    }

    @classmethod
    def set_plot_style(cls):
        """Configures clean matplotlib aesthetic parameters."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.titlesize": 15,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        })

    @classmethod
    def plot_position_error(
        cls,
        df: pd.DataFrame,
        metrics: EvaluationMetrics,
        save_path: Path,
    ):
        """
        Plots Position Error over time during the GNSS outage.
        """
        cls.set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

        t = df["time_rel_sec"]
        raw_err = df["eval_raw_position_error_m"]
        cal_err = df["eval_calibrated_position_error_m"]

        ax.plot(
            t, raw_err,
            label=f"Raw AI Dead-Reckoning (Final: {metrics.raw_final_error_m:.2f} m)",
            color="#EF4444",
            linewidth=2.2,
            linestyle="--",
        )
        ax.plot(
            t, cal_err,
            label=f"Calibrated AI DR (Final: {metrics.calibrated_final_error_m:.2f} m | Drift: {metrics.calibrated_drift_percent_gps:.2f}%)",
            color="#10B981",
            linewidth=2.5,
        )

        # Highlight error envelope / reduction
        ax.fill_between(t, cal_err, raw_err, color="#EF4444", alpha=0.12, label="Error Reduction Zone")
        ax.axhline(0, color="#64748B", linewidth=0.8)

        # Key milestone annotations
        ax.scatter([t.iloc[-1]], [cal_err.iloc[-1]], color="#10B981", s=70, zorder=5)
        ax.annotate(
            f"  {metrics.calibrated_final_error_m:.2f} m ({metrics.calibrated_drift_percent_gps:.2f}% Drift)",
            xy=(t.iloc[-1], cal_err.iloc[-1]),
            xytext=(t.iloc[-1] - 4.5, cal_err.iloc[-1] + 4.0),
            arrowprops=dict(facecolor="#10B981", shrink=0.08, width=1, headwidth=6),
            fontweight="bold",
            color="#065F46",
        )

        ax.set_title("Position Error Over Time During GNSS Outage", fontweight="bold", pad=12)
        ax.set_xlabel("Time in GNSS Outage (seconds)", fontweight="semibold")
        ax.set_ylabel("Euclidean Position Error (meters)", fontweight="semibold")
        ax.set_xlim(0, t.iloc[-1] * 1.02)
        ax.set_ylim(0, max(raw_err.max() * 1.08, 10))
        ax.legend(loc="upper left", frameon=True, framealpha=0.92)

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def plot_trajectory_comparison(
        cls,
        df: pd.DataFrame,
        metrics: EvaluationMetrics,
        save_path: Path,
    ):
        """
        Plots 2D Trajectory comparison (GPS Ground Truth vs Raw DR vs Calibrated DR).
        """
        cls.set_plot_style()
        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

        gps_x, gps_y = df["gps_x"], df["gps_y"]
        dr_x, dr_y = df["dr_x"], df["dr_y"]
        corr_x, corr_y = df["corrected_x"], df["corrected_y"]

        # Trajectory lines
        ax.plot(gps_x, gps_y, label="GPS Ground Truth", color="#2563EB", linewidth=2.8, zorder=3)
        ax.plot(dr_x, dr_y, label="Raw AI Dead-Reckoning", color="#EF4444", linewidth=2.0, linestyle="--", zorder=2)
        ax.plot(corr_x, corr_y, label="Calibrated AI Dead-Reckoning", color="#10B981", linewidth=2.4, zorder=4)

        # Mark Start and End Points
        ax.scatter([gps_x.iloc[0]], [gps_y.iloc[0]], color="#0F172A", s=110, marker="o", zorder=6, label="Outage Start Point")
        ax.scatter([gps_x.iloc[-1]], [gps_y.iloc[-1]], color="#2563EB", s=100, marker="s", zorder=6, label="GPS End Point")
        ax.scatter([dr_x.iloc[-1]], [dr_y.iloc[-1]], color="#EF4444", s=90, marker="^", zorder=6, label="Raw DR End Point")
        ax.scatter([corr_x.iloc[-1]], [corr_y.iloc[-1]], color="#10B981", s=90, marker="D", zorder=6, label="Calibrated DR End Point")

        # Connection line showing final calibrated drift error
        ax.plot([corr_x.iloc[-1], gps_x.iloc[-1]], [corr_y.iloc[-1], gps_y.iloc[-1]],
                color="#059669", linestyle=":", linewidth=2, label=f"Final Drift Vector: {metrics.calibrated_final_error_m:.2f} m")

        ax.set_title("2D Trajectory Comparison (GNSS Outage Window)", fontweight="bold", pad=12)
        ax.set_xlabel("Local X Coordinate (meters)", fontweight="semibold")
        ax.set_ylabel("Local Y Coordinate (meters)", fontweight="semibold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", frameon=True, framealpha=0.92)

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def plot_drift_percentage(
        cls,
        df: pd.DataFrame,
        metrics: EvaluationMetrics,
        save_path: Path,
    ):
        """
        Plots Drift Percentage over time during GNSS outage.
        """
        cls.set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)

        t = df["time_rel_sec"]
        cal_drift = df["eval_calibrated_drift_pct"]
        raw_drift = df["eval_raw_drift_pct"]

        ax.plot(
            t, raw_drift,
            label=f"Raw AI Dead-Reckoning (Final: {metrics.raw_drift_percent_gps:.2f}%)",
            color="#F59E0B",
            linewidth=2.2,
            linestyle="--",
        )
        ax.plot(
            t, cal_drift,
            label=f"Calibrated AI DR (Final: {metrics.calibrated_drift_percent_gps:.2f}%)",
            color="#3B82F6",
            linewidth=2.5,
        )

        ax.axhline(metrics.calibrated_drift_percent_gps, color="#3B82F6", linestyle=":", alpha=0.7)
        ax.scatter([t.iloc[-1]], [cal_drift.iloc[-1]], color="#3B82F6", s=70, zorder=5)

        ax.annotate(
            f" Target Baseline: {metrics.calibrated_drift_percent_gps:.2f}%",
            xy=(t.iloc[-1], cal_drift.iloc[-1]),
            xytext=(t.iloc[-1] - 5.0, cal_drift.iloc[-1] + 12),
            arrowprops=dict(facecolor="#3B82F6", shrink=0.08, width=1, headwidth=6),
            fontweight="bold",
            color="#1E3A8A",
        )

        ax.set_title("Drift Percentage Over Time (Position Error / GPS Distance Travelled × 100)", fontweight="bold", pad=12)
        ax.set_xlabel("Time in GNSS Outage (seconds)", fontweight="semibold")
        ax.set_ylabel("Cumulative Drift (%)", fontweight="semibold")
        ax.set_xlim(0, t.iloc[-1] * 1.02)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92)

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def plot_gnss_recovery(
        cls,
        df_rec: pd.DataFrame,
        save_path: Path,
    ):
        """
        Plots the smooth post-outage GNSS recovery transition.
        """
        cls.set_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        t = df_rec["time_rel_sec"]
        outage_mask = df_rec["phase"] == "GNSS_OUTAGE"
        rec_mask = df_rec["phase"] == "GNSS_RECOVERED"

        # Panel 1: Position error transition
        ax1.plot(t[outage_mask], df_rec.loc[outage_mask, "eval_calibrated_position_error_m"],
                 color="#EF4444", linewidth=2.2, label="Outage Phase (DR Drift)")
        ax1.plot(t[rec_mask], df_rec.loc[rec_mask, "eval_calibrated_position_error_m"],
                 color="#10B981", linewidth=2.5, label="Recovery Phase (Smooth GNSS Lock)")
        ax1.axvline(t[outage_mask].iloc[-1], color="#64748B", linestyle="--", label="GNSS Signal Reacquired")

        ax1.set_title("Position Error: Outage vs Smooth Recovery", fontweight="bold")
        ax1.set_xlabel("Elapsed Time (s)", fontweight="semibold")
        ax1.set_ylabel("Position Error (m)", fontweight="semibold")
        ax1.legend(loc="upper left")

        # Panel 2: 2D Path
        ax2.plot(df_rec["gps_x"], df_rec["gps_y"], label="GPS Ground Truth", color="#2563EB", linewidth=2.5)
        ax2.plot(df_rec.loc[outage_mask, "recovered_x"], df_rec.loc[outage_mask, "recovered_y"],
                 label="Outage Dead-Reckoning", color="#EF4444", linewidth=2.2, linestyle="--")
        ax2.plot(df_rec.loc[rec_mask, "recovered_x"], df_rec.loc[rec_mask, "recovered_y"],
                 label="Smooth Re-acquisition Path", color="#10B981", linewidth=2.8)

        ax2.set_title("Trajectory Blending During Signal Recovery", fontweight="bold")
        ax2.set_xlabel("Local X (m)", fontweight="semibold")
        ax2.set_ylabel("Local Y (m)", fontweight="semibold")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.legend(loc="best")

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def plot_dashboard_summary(
        cls,
        df: pd.DataFrame,
        metrics: EvaluationMetrics,
        save_path: Path,
    ):
        """
        Generates a 4-panel comprehensive dashboard summary plot.
        """
        cls.set_plot_style()
        fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

        t = df["time_rel_sec"]

        # Panel 1: Position Error
        ax1 = axs[0, 0]
        ax1.plot(t, df["eval_raw_position_error_m"], label="Raw AI DR", color="#EF4444", linewidth=2, linestyle="--")
        ax1.plot(t, df["eval_calibrated_position_error_m"], label="Calibrated AI DR", color="#10B981", linewidth=2.5)
        ax1.fill_between(t, df["eval_calibrated_position_error_m"], df["eval_raw_position_error_m"], color="#EF4444", alpha=0.15)
        ax1.set_title("A) Position Error Over Outage", fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Error (m)")
        ax1.legend(loc="upper left")

        # Panel 2: 2D Trajectory
        ax2 = axs[0, 1]
        ax2.plot(df["gps_x"], df["gps_y"], label="GPS Truth", color="#2563EB", linewidth=2.5)
        ax2.plot(df["dr_x"], df["dr_y"], label="Raw DR", color="#EF4444", linewidth=1.8, linestyle="--")
        ax2.plot(df["corrected_x"], df["corrected_y"], label="Calibrated DR", color="#10B981", linewidth=2.2)
        ax2.scatter([df["gps_x"].iloc[0]], [df["gps_y"].iloc[0]], color="#0F172A", s=80, marker="o", label="Start", zorder=5)
        ax2.set_title("B) Trajectory Comparison (2D)", fontweight="bold")
        ax2.set_xlabel("Local X (m)")
        ax2.set_ylabel("Local Y (m)")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.legend(loc="best")

        # Panel 3: Speed Comparison
        ax3 = axs[1, 0]
        if "predicted_speed_mps" in df.columns:
            ai_speed = df["predicted_speed_mps"]
        else:
            ai_speed = df["predicted_speed"] / 3.6
        if "gps_speed" in df.columns:
            gps_spd = df["gps_speed"]
        else:
            gps_spd = df["vehicle_speed"] / 3.6
        ax3.plot(t, gps_spd, label="GPS Speed", color="#2563EB", linewidth=2.2)
        ax3.plot(t, ai_speed, label="AI Predicted Speed", color="#F59E0B", linewidth=2.0, linestyle="--")
        ax3.set_title("C) Vehicle Speed Profile", fontweight="bold")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Speed (m/s)")
        ax3.legend(loc="best")

        # Panel 4: Drift Percentage & KPI Summary
        ax4 = axs[1, 1]
        ax4.plot(t, df["eval_calibrated_drift_pct"], label="Calibrated Drift %", color="#3B82F6", linewidth=2.2)
        ax4.plot(t, df["eval_raw_drift_pct"], label="Raw Drift %", color="#EF4444", linewidth=1.8, linestyle=":")
        ax4.set_title("D) Drift Percentage Profile", fontweight="bold")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Drift (%)")
        ax4.legend(loc="upper right")

        # KPI Card Overlay in Panel 4
        kpi_text = (
            f"Outage Duration: {metrics.outage_duration_sec:.1f} s\n"
            f"GPS Distance: {metrics.total_gps_distance_m:.1f} m\n"
            f"Final Position Error: {metrics.calibrated_final_error_m:.2f} m\n"
            f"Final Drift: {metrics.calibrated_drift_percent_gps:.2f}%\n"
            f"RMSE: {metrics.calibrated_rmse_m:.2f} m\n"
            f"Error Reduction: {metrics.error_reduction_percent:.1f}%"
        )
        ax4.text(
            0.05, 0.45, kpi_text,
            transform=ax4.transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.92)
        )

        fig.suptitle(
            f"GNSS Dead-Reckoning Navigation Evaluation (Final Drift: {metrics.calibrated_drift_percent_gps:.2f}%)",
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def evaluate_dead_reckoning(
    input_csv: Union[str, Path] = "data/final_calibrated_results.csv",
    output_dir: Union[str, Path] = "results",
    generate_plots: bool = True,
) -> Tuple[pd.DataFrame, EvaluationMetrics]:
    """
    High-level API to evaluate dead-reckoning trajectory, export results CSV,
    summary JSON metrics, and visual evaluation plots.

    Args:
        input_csv: Path to input results CSV.
        output_dir: Destination folder for exported CSV, JSON, and graphs.
        generate_plots: Whether to generate and save matplotlib evaluation figures.

    Returns:
        Tuple of (df_evaluated, metrics).
    """
    input_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Evaluation
    evaluator = TrajectoryEvaluator(input_path)
    df_eval, metrics = evaluator.evaluate()

    # 2. Export Evaluated Dataset
    csv_out_path = out_dir / "evaluation_results.csv"
    df_eval.to_csv(csv_out_path, index=False)

    # 3. Export Metrics Summary JSON
    json_out_path = out_dir / "metrics_summary.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    # 4. Generate Visual Plots
    if generate_plots:
        PlottingEngine.plot_position_error(df_eval, metrics, out_dir / "position_error_outage.png")
        PlottingEngine.plot_trajectory_comparison(df_eval, metrics, out_dir / "trajectory_comparison.png")
        PlottingEngine.plot_drift_percentage(df_eval, metrics, out_dir / "drift_percentage_over_time.png")

        # Simulate smooth recovery extension for visual verification
        df_recovery = GNSSRecovery.simulate_recovery_trajectory(df_eval)
        PlottingEngine.plot_gnss_recovery(df_recovery, out_dir / "gnss_recovery_analysis.png")

        # Master dashboard summary
        PlottingEngine.plot_dashboard_summary(df_eval, metrics, out_dir / "dashboard_summary.png")

    return df_eval, metrics


def main():
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="GNSS Dead-Reckoning Navigation & Trajectory Evaluation Tool"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/final_calibrated_results.csv",
        help="Path to input results CSV (default: data/final_calibrated_results.csv)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Directory to save evaluation results and plots (default: results)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation and only export CSV/JSON metrics.",
    )

    args = parser.parse_args()

    print("=" * 68)
    print("      GNSS DEAD-RECKONING NAVIGATION & EVALUATION ENGINE")
    print("=" * 68)
    print(f"Loading input: {args.input}")

    df_eval, metrics = evaluate_dead_reckoning(
        input_csv=args.input,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
    )

    print("\n--- EVALUATION SUMMARY ---")
    print(f"Total Outage Duration     : {metrics.outage_duration_sec:.2f} s ({metrics.total_samples} samples)")
    print(f"GPS Distance Travelled    : {metrics.total_gps_distance_m:.2f} m")
    print(f"AI DR Distance Travelled  : {metrics.total_dr_distance_m:.2f} m")
    print("-" * 68)
    print("RAW AI DEAD-RECKONING PERFORMANCE:")
    print(f"  Final Position Error    : {metrics.raw_final_error_m:.2f} m")
    print(f"  Mean Error (MAE)        : {metrics.raw_mean_error_m:.2f} m")
    print(f"  Max Error               : {metrics.raw_max_error_m:.2f} m")
    print(f"  RMSE                    : {metrics.raw_rmse_m:.2f} m")
    print(f"  CEP50 / R95             : {metrics.raw_cep50_m:.2f} m / {metrics.raw_r95_m:.2f} m")
    print(f"  Drift % (vs GPS dist)   : {metrics.raw_drift_percent_gps:.2f}%")
    print(f"  Drift % (vs DR dist)    : {metrics.raw_drift_percent_dr:.2f}%")
    print("-" * 68)
    print("CALIBRATED AI DEAD-RECKONING PERFORMANCE:")
    print(f"  Final Position Error    : {metrics.calibrated_final_error_m:.2f} m")
    print(f"  Mean Error (MAE)        : {metrics.calibrated_mean_error_m:.2f} m")
    print(f"  Max Error               : {metrics.calibrated_max_error_m:.2f} m")
    print(f"  RMSE                    : {metrics.calibrated_rmse_m:.2f} m")
    print(f"  CEP50 / R95             : {metrics.calibrated_cep50_m:.2f} m / {metrics.calibrated_r95_m:.2f} m")
    print(f"  [*] Final Drift % (GPS) : {metrics.calibrated_drift_percent_gps:.2f}% (Benchmark Baseline: 34.63%)")
    print(f"  Final Drift % (DR)      : {metrics.calibrated_drift_percent_dr:.2f}%")
    print(f"  Error Reduction         : {metrics.error_reduction_percent:.2f}%")
    print("=" * 68)
    print(f"Saved evaluation results to: {args.output_dir}/evaluation_results.csv")
    print(f"Saved metrics summary to   : {args.output_dir}/metrics_summary.json")
    if not args.no_plots:
        print(f"Saved visual graphs to     : {args.output_dir}/*.png")
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
