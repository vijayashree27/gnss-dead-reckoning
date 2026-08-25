"""
Pure Inertial Navigation System (INS) Baseline
===============================================
Implements a "pure INS" dead-reckoning baseline that relies ONLY on
onboard IMU signals (accelerometer + gyroscope) integrated over time,
with a single one-time heading alignment at t=0. This is the classical
inertial-navigation baseline that AI-assisted dead reckoning is meant
to outperform.

Unlike the AI Dead-Reckoning pipeline (which uses vehicle_speed /
vehicle_heading or an AI-predicted speed as input), this module never
reads vehicle_speed or vehicle_heading during propagation. It only
uses:
    - ax          : forward-axis accelerometer reading (m/s^2)
    - gyro_yaw    : yaw rate gyroscope reading (rad/s)
    - dt          : sample interval (s)

A single heading value (e.g. vehicle_heading at the first sample, or
0.0) may be supplied purely as an initial alignment constant -- this
mirrors how a real INS needs *some* initial heading reference (usually
from a magnetometer or GNSS fix at the moment the outage begins) but
does not use any external heading/speed measurements thereafter.

Because a pure INS has no absolute position correction, it is expected
to drift significantly -- that is the point: it is the baseline that
the AI Dead-Reckoning system must beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class INSBaselineResult:
    """Summary metrics for a pure-INS propagation run."""
    samples: int
    duration_sec: float
    accel_bias_mps2: float
    gyro_bias_radps: float
    final_speed_mps: float
    total_distance_m: float


class PureINSBaseline:
    """
    Classical strapdown-style INS baseline using only accelerometer +
    gyroscope integration (no wheel speed, no vehicle heading sensor,
    no AI model).
    """

    def __init__(
        self,
        forward_accel_col: str = "ax",
        gyro_yaw_col: str = "gyro_yaw",
        dt_col: str = "dt",
        bias_calib_samples: int = 20,
    ):
        """
        Args:
            forward_accel_col: Column holding forward-axis accelerometer
                readings (m/s^2). The IO-VNBD phone IMU is assumed to be
                roughly forward/lateral aligned, so `ax` is treated as the
                forward acceleration axis.
            gyro_yaw_col: Column holding yaw-rate gyroscope readings (rad/s).
            dt_col: Column holding the sample interval (s).
            bias_calib_samples: Number of leading samples used to estimate
                constant accelerometer/gyro bias (simple static calibration,
                standard practice for INS -- assumes the vehicle is
                approximately stationary or at constant heading/speed for
                this short window). Set to 0 to disable bias removal.
        """
        self.forward_accel_col = forward_accel_col
        self.gyro_yaw_col = gyro_yaw_col
        self.dt_col = dt_col
        self.bias_calib_samples = max(0, int(bias_calib_samples))

    def propagate(
        self,
        df: pd.DataFrame,
        initial_heading_rad: float = 0.0,
        initial_speed_mps: float = 0.0,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
    ) -> Tuple[pd.DataFrame, INSBaselineResult]:
        """
        Propagate a pure-INS trajectory over `df`.

        Args:
            df: DataFrame containing at least [forward_accel_col,
                gyro_yaw_col, dt_col], in chronological order.
            initial_heading_rad: One-time heading alignment (radians,
                compass convention: 0 = North, +90 = East). This is the
                ONLY external reference the INS baseline is allowed --
                equivalent to knowing which way the vehicle was pointed
                the instant GNSS was lost.
            initial_speed_mps: Starting forward speed (m/s). Typically the
                last known GNSS/vehicle speed at outage start.
            initial_x, initial_y: Starting local-frame position (m),
                typically the last known GNSS fix.

        Returns:
            (df_out, result): df_out is a copy of df with added columns
                `ins_heading_rad`, `ins_speed_mps`, `ins_x`, `ins_y`.
                result is a summary INSBaselineResult.
        """
        for col in (self.forward_accel_col, self.gyro_yaw_col, self.dt_col):
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataframe.")

        df = df.reset_index(drop=True).copy()
        n = len(df)
        if n == 0:
            raise ValueError("Cannot propagate INS over an empty dataframe.")

        accel = df[self.forward_accel_col].to_numpy(dtype=float)
        gyro = df[self.gyro_yaw_col].to_numpy(dtype=float)
        dt = df[self.dt_col].to_numpy(dtype=float)

        # --- Static bias calibration -----------------------------------
        # Standard INS practice: estimate constant sensor bias from a
        # short leading window and remove it before integration. This is
        # NOT a position correction -- it never looks at GNSS/vehicle
        # speed and is applied identically regardless of outage location.
        n_calib = min(self.bias_calib_samples, n)
        if n_calib > 0:
            accel_bias = float(np.mean(accel[:n_calib]))
            gyro_bias = float(np.mean(gyro[:n_calib]))
        else:
            accel_bias = 0.0
            gyro_bias = 0.0

        accel_dbiased = accel - accel_bias
        gyro_dbiased = gyro - gyro_bias

        # --- Integrate heading (yaw-rate integration) -------------------
        heading = np.empty(n)
        heading[0] = initial_heading_rad
        for i in range(1, n):
            heading[i] = heading[i - 1] + gyro_dbiased[i] * dt[i]

        # --- Integrate forward speed (accelerometer integration) --------
        speed = np.empty(n)
        speed[0] = initial_speed_mps
        for i in range(1, n):
            speed[i] = speed[i - 1] + accel_dbiased[i] * dt[i]

        # --- Integrate position (compass convention: 0=N, 90=E) ---------
        x = np.empty(n)
        y = np.empty(n)
        x[0] = initial_x
        y[0] = initial_y
        for i in range(1, n):
            distance = speed[i] * dt[i]
            x[i] = x[i - 1] + distance * np.sin(heading[i])
            y[i] = y[i - 1] + distance * np.cos(heading[i])

        df["ins_heading_rad"] = heading
        df["ins_speed_mps"] = speed
        df["ins_x"] = x
        df["ins_y"] = y

        total_distance = float(np.sum(np.abs(speed[1:] * dt[1:])))
        result = INSBaselineResult(
            samples=n,
            duration_sec=float(np.sum(dt[1:])),
            accel_bias_mps2=accel_bias,
            gyro_bias_radps=gyro_bias,
            final_speed_mps=float(speed[-1]),
            total_distance_m=total_distance,
        )

        return df, result


def compute_pure_ins(
    df: pd.DataFrame,
    forward_accel_col: str = "ax",
    gyro_yaw_col: str = "gyro_yaw",
    dt_col: str = "dt",
    bias_calib_samples: int = 20,
    initial_heading_rad: float = 0.0,
    initial_speed_mps: float = 0.0,
    initial_x: float = 0.0,
    initial_y: float = 0.0,
) -> Tuple[pd.DataFrame, INSBaselineResult]:
    """Convenience functional wrapper around PureINSBaseline.propagate()."""
    ins = PureINSBaseline(
        forward_accel_col=forward_accel_col,
        gyro_yaw_col=gyro_yaw_col,
        dt_col=dt_col,
        bias_calib_samples=bias_calib_samples,
    )
    return ins.propagate(
        df,
        initial_heading_rad=initial_heading_rad,
        initial_speed_mps=initial_speed_mps,
        initial_x=initial_x,
        initial_y=initial_y,
    )
