"""
Reusable GNSS Outage Simulator
===============================
Turns a continuous IMU + GPS trip (e.g. `feature_dataset.csv` /
`ai_speed_predictions.csv`) into a clean, parametrized GNSS-outage
scenario that can be re-run for any start point / duration, producing
output directly compatible with `navigation.evaluation.TrajectoryEvaluator`.

Typical usage
-------------
    df = pd.read_csv("data/ai_speed_predictions.csv")
    outage_df, full_df, window = simulate_gnss_outage(
        df,
        start_time_s=120.0,
        duration_s=30.0,
    )
    # outage_df is ready to feed straight into TrajectoryEvaluator:
    outage_df.to_csv("data/current_outage.csv", index=False)
    df_eval, metrics = evaluate_dead_reckoning(input_csv="data/current_outage.csv")

Design goals
------------
1. Reusable: any start index/time and any duration can be requested,
   including a *random* valid window (for repeated experiments / demos).
2. Clean: GPS is genuinely hidden for the outage window -- downstream
   navigation only sees IMU (+ optionally the AI speed model) for that
   window, anchored at the last known GNSS fix.
3. Comparable: produces BOTH a pure-INS trajectory (`ins_x`/`ins_y`,
   via navigation.ins_baseline) and an AI Dead-Reckoning trajectory
   (`dr_x`/`dr_y`, using the AI-predicted speed if available, otherwise
   falling back to vehicle_speed) over the exact same outage window, so
   the two can be evaluated and plotted side by side.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .ins_baseline import PureINSBaseline
except ImportError:  # allow running as a standalone script
    from ins_baseline import PureINSBaseline

EARTH_RADIUS_M = 6371000.0


@dataclass
class OutageWindow:
    """Resolved indices/timing describing a simulated outage."""
    start_idx: int
    end_idx: int  # exclusive
    anchor_idx: int  # last known-good GNSS fix (start_idx - 1, clipped to 0)
    start_time_s: float
    duration_s: float
    n_samples: int


def latlon_to_local_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: Optional[float] = None,
    lon0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Equirectangular projection of lat/lon (degrees) to local East/North
    metres, anchored at (lat0, lon0). If lat0/lon0 are None, the first
    element of `lat`/`lon` is used as the anchor.

    Returns (x, y, lat0, lon0).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if lat0 is None:
        lat0 = float(lat[0])
    if lon0 is None:
        lon0 = float(lon[0])

    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * EARTH_RADIUS_M
    y = (lat_rad - lat0_rad) * EARTH_RADIUS_M

    return x, y, lat0, lon0


def _resolve_window(
    n_rows: int,
    time_rel_s: np.ndarray,
    start_idx: Optional[int],
    start_time_s: Optional[float],
    duration_s: float,
    seed: Optional[int],
) -> OutageWindow:
    """Resolve a valid (start_idx, end_idx) pair from the given hints."""
    total_duration = float(time_rel_s[-1])

    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if duration_s >= total_duration:
        raise ValueError(
            f"Requested outage duration ({duration_s}s) exceeds trip "
            f"duration ({total_duration:.1f}s)."
        )

    if start_idx is not None:
        resolved_start_idx = int(start_idx)
    elif start_time_s is not None:
        resolved_start_idx = int(np.searchsorted(time_rel_s, start_time_s))
    else:
        # Random valid start: leave at least 1 sample before (for the
        # anchor fix) and `duration_s` seconds of room afterwards.
        rng = random.Random(seed)
        latest_valid_start_time = total_duration - duration_s
        if latest_valid_start_time <= 0:
            raise ValueError("Trip too short for requested duration_s.")
        random_start_time = rng.uniform(0.0, latest_valid_start_time)
        resolved_start_idx = int(np.searchsorted(time_rel_s, random_start_time))

    resolved_start_idx = max(1, min(resolved_start_idx, n_rows - 2))
    start_time = float(time_rel_s[resolved_start_idx])
    end_time = start_time + duration_s
    resolved_end_idx = int(np.searchsorted(time_rel_s, end_time))
    resolved_end_idx = max(resolved_start_idx + 1, min(resolved_end_idx, n_rows))

    anchor_idx = resolved_start_idx - 1

    return OutageWindow(
        start_idx=resolved_start_idx,
        end_idx=resolved_end_idx,
        anchor_idx=anchor_idx,
        start_time_s=start_time,
        duration_s=float(time_rel_s[resolved_end_idx - 1] - start_time),
        n_samples=resolved_end_idx - resolved_start_idx,
    )


def simulate_gnss_outage(
    df: pd.DataFrame,
    start_idx: Optional[int] = None,
    start_time_s: Optional[float] = None,
    duration_s: float = 30.0,
    ai_speed_col: str = "predicted_speed",
    vehicle_speed_col: str = "vehicle_speed",
    vehicle_heading_col: str = "vehicle_heading",
    timestamp_col: str = "timestamp_ms",
    gps_lat_col: str = "gps_lat",
    gps_lon_col: str = "gps_lon",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, OutageWindow]:
    """
    Simulate a single, reusable GNSS outage window over a continuous trip.

    GNSS (gps_x/gps_y) is only used as ground truth for evaluation -- it is
    never fed into the dead-reckoning propagation for rows inside the
    outage window. Both a pure-INS baseline and an AI Dead-Reckoning
    trajectory are produced for the same window so they can be compared
    directly.

    Args:
        df: Full trip dataframe (chronological order), containing IMU
            columns (ax, gyro_yaw, dt), GPS lat/lon, vehicle_speed,
            vehicle_heading, and optionally an AI-predicted speed column.
        start_idx: Explicit outage start row index. Takes priority over
            start_time_s.
        start_time_s: Outage start time in seconds relative to the start
            of `df`. Ignored if start_idx is given.
        duration_s: Outage duration in seconds. If neither start_idx nor
            start_time_s is given, a random valid start is chosen so that
            a window of this duration fits within the trip.
        ai_speed_col: Column with AI-predicted vehicle speed (same units
            as vehicle_speed, typically km/h). If absent, the AI
            Dead-Reckoning trajectory falls back to vehicle_speed (so the
            function still works on datasets that only have the pure
            feature set).
        seed: RNG seed used only when picking a random outage window.

    Returns:
        (outage_df, full_annotated_df, window)
            outage_df: rows within [start_idx, end_idx), containing
                timestamp_ms, gps_x, gps_y, dr_x, dr_y (AI DR), ins_x,
                ins_y (pure INS), corrected_x/corrected_y (alias of
                dr_x/dr_y so it drops straight into
                navigation.evaluation.TrajectoryEvaluator).
            full_annotated_df: the full trip with a `phase` column
                ("GNSS_AVAILABLE" / "GNSS_OUTAGE") and gps_x/gps_y added,
                useful for a "GNSS available -> lost -> recovered"
                dashboard demo.
            window: the resolved OutageWindow (indices/timing) actually
                used, for logging/reproducibility.
    """
    required = [timestamp_col, gps_lat_col, gps_lon_col, "ax", "gyro_yaw", "dt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    df = df.reset_index(drop=True).copy()
    n_rows = len(df)

    t0 = df[timestamp_col].iloc[0]
    time_rel_s = (df[timestamp_col].to_numpy(dtype=float) - float(t0)) / 1000.0

    window = _resolve_window(
        n_rows=n_rows,
        time_rel_s=time_rel_s,
        start_idx=start_idx,
        start_time_s=start_time_s,
        duration_s=duration_s,
        seed=seed,
    )

    # 1. Ground-truth GPS -> local metres, anchored at trip start.
    gps_x_full, gps_y_full, lat0, lon0 = latlon_to_local_xy(
        df[gps_lat_col].to_numpy(dtype=float),
        df[gps_lon_col].to_numpy(dtype=float),
    )
    df["gps_x"] = gps_x_full
    df["gps_y"] = gps_y_full

    # 2. Phase / GNSS-availability annotation over the FULL trip.
    df["phase"] = "GNSS_AVAILABLE"
    df["gnss_available"] = True
    df.loc[window.start_idx: window.end_idx - 1, "phase"] = "GNSS_OUTAGE"
    df.loc[window.start_idx: window.end_idx - 1, "gnss_available"] = False

    # 3. Anchor position/speed/heading at the last known-good GNSS fix.
    anchor_x = float(gps_x_full[window.anchor_idx])
    anchor_y = float(gps_y_full[window.anchor_idx])
    anchor_speed_mps = float(df[vehicle_speed_col].iloc[window.anchor_idx]) / 3.6
    anchor_heading_rad = float(np.deg2rad(df[vehicle_heading_col].iloc[window.anchor_idx]))

    outage_slice = df.iloc[window.start_idx: window.end_idx].reset_index(drop=True)

    # 4. Pure INS baseline (accel + gyro integration only).
    ins = PureINSBaseline()
    ins_df, ins_result = ins.propagate(
        outage_slice,
        initial_heading_rad=anchor_heading_rad,
        initial_speed_mps=anchor_speed_mps,
        initial_x=anchor_x,
        initial_y=anchor_y,
    )

    # 5. AI Dead-Reckoning trajectory: AI-predicted speed (falls back to
    #    vehicle_speed if no AI predictions are present) + vehicle heading,
    #    exactly matching the existing AI DR convention used elsewhere in
    #    this project (heading comes from the vehicle sensor; only speed
    #    estimation is replaced by the AI model).
    speed_col = ai_speed_col if ai_speed_col in outage_slice.columns else vehicle_speed_col
    speed_mps = outage_slice[speed_col].to_numpy(dtype=float) / 3.6
    heading_rad = np.deg2rad(outage_slice[vehicle_heading_col].to_numpy(dtype=float))
    dt = outage_slice["dt"].to_numpy(dtype=float)

    m = len(outage_slice)
    dr_x = np.empty(m)
    dr_y = np.empty(m)
    dr_x[0] = anchor_x
    dr_y[0] = anchor_y
    for i in range(1, m):
        distance = speed_mps[i] * dt[i]
        dr_x[i] = dr_x[i - 1] + distance * np.sin(heading_rad[i])
        dr_y[i] = dr_y[i - 1] + distance * np.cos(heading_rad[i])

    outage_df = ins_df.copy()
    outage_df["dr_x"] = dr_x
    outage_df["dr_y"] = dr_y
    # `corrected_x/y` alias so this drops straight into
    # navigation.evaluation.TrajectoryEvaluator without modification.
    outage_df["corrected_x"] = dr_x
    outage_df["corrected_y"] = dr_y
    outage_df["dr_source_speed_col"] = speed_col

    return outage_df, df, window
