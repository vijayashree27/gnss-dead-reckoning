
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Intelligent Dead Reckoning",
    page_icon="🛰️",
    layout="wide"
)

# --------------------------------------------------
# DATA
# --------------------------------------------------

FILE = "data/final_calibrated_results.csv"

df = pd.read_csv(FILE)

# --------------------------------------------------
# REAL RESULTS FROM OUR EXPERIMENT
# --------------------------------------------------

DISTANCE = 64.16
FINAL_ERROR = 22.22
MAX_ERROR = 28.87
DRIFT = 34.63

ACTUAL_SPEED = 43.55
AI_SPEED = 37.03

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛰️ Intelligent Dead Reckoning")
st.caption("GNSS-Denied Vehicle Navigation using AI + IMU")

# --------------------------------------------------
# OUTAGE CONTROL
# --------------------------------------------------

if "outage" not in st.session_state:
    st.session_state.outage = False

if st.button(
    "🔴 Simulate GNSS Outage",
    use_container_width=True
):
    st.session_state.outage = True

# --------------------------------------------------
# STATUS
# --------------------------------------------------

if st.session_state.outage:

    gnss_status = "🔴 GNSS LOST"
    navigation_mode = "AI DEAD RECKONING"
    speed = AI_SPEED

else:

    gnss_status = "🟢 GNSS AVAILABLE"
    navigation_mode = "GNSS + INS"
    speed = ACTUAL_SPEED

# --------------------------------------------------
# METRIC CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "GNSS Status",
        gnss_status
    )

with col2:
    st.metric(
        "Navigation Mode",
        navigation_mode
    )

with col3:
    st.metric(
        "Vehicle Speed",
        f"{speed:.2f} km/h"
    )

with col4:
    if st.session_state.outage:
        st.metric(
            "Drift",
            f"{DRIFT:.2f}%"
        )
    else:
        st.metric(
            "Drift",
            "—"
        )

st.divider()

# --------------------------------------------------
# TRAJECTORY
# --------------------------------------------------

st.subheader("Vehicle Trajectory")

# Try to automatically find useful columns
columns = df.columns.tolist()

# possible X/Y columns
x_candidates = [
    "predicted_x",
    "ai_x",
    "x",
    "east_west",
    "east",
    "position_x"
]

y_candidates = [
    "predicted_y",
    "ai_y",
    "y",
    "north_south",
    "north",
    "position_y"
]

x_col = next((c for c in x_candidates if c in columns), None)
y_col = next((c for c in y_candidates if c in columns), None)

# Ground truth candidates
gt_x_candidates = [
    "ground_truth_x",
    "gt_x",
    "gps_x",
    "true_x"
]

gt_y_candidates = [
    "ground_truth_y",
    "gt_y",
    "gps_y",
    "true_y"
]

gt_x = next((c for c in gt_x_candidates if c in columns), None)
gt_y = next((c for c in gt_y_candidates if c in columns), None)

fig = go.Figure()

# AI trajectory
if x_col and y_col:

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name="AI Dead Reckoning"
        )
    )

# Ground truth
if gt_x and gt_y:

    fig.add_trace(
        go.Scatter(
            x=df[gt_x],
            y=df[gt_y],
            mode="lines",
            name="Ground Truth"
        )
    )

fig.update_layout(
    xaxis_title="East-West Position (m)",
    yaxis_title="North-South Position (m)",
    height=500,
    template="plotly_white"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# OUTAGE RESULTS
# --------------------------------------------------

if st.session_state.outage:

    st.divider()

    st.subheader("🚨 GNSS Outage Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Distance Travelled",
            f"{DISTANCE:.2f} m"
        )

    with c2:
        st.metric(
            "Final Position Error",
            f"{FINAL_ERROR:.2f} m"
        )

    with c3:
        st.metric(
            "Drift",
            f"{DRIFT:.2f}%"
        )

    st.info(
        "GNSS unavailable. Vehicle navigation is continuing "
        "using AI-based dead reckoning."
    )

else:

    st.success(
        "GNSS available. Press 'Simulate GNSS Outage' "
        "to switch to AI dead reckoning."
    )

# --------------------------------------------------
# PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("AI Model Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Actual Mean Speed",
        "43.55 km/h"
    )

with c2:
    st.metric(
        "AI Predicted Mean Speed",
        "37.03 km/h"
    )

with c3:
    st.metric(
        "Drift After Calibration",
        "34.63%"
    )

st.caption(
    "Baseline drift: 105.72% → Calibrated drift: 34.63%"
)