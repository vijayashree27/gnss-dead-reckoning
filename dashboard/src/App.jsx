import "./App.css";
import { useEffect, useMemo, useState } from "react";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import {
  Navigation,
  Satellite,
  SatelliteDish,
  Activity,
  AlertTriangle,
  RotateCcw,
  Play,
  MapPin,
  Gauge,
} from "lucide-react";

// Demo trajectory data from the navigation pipeline.
// Kept as the dashboard's current benchmark/demo dataset.
const trajectoryData = [
  { t: 0, truth: 0, pureIns: 0, aiIns: 0, corrected: 0 },
  { t: 10, truth: 10, pureIns: 9.8, aiIns: 9.9, corrected: 10 },
  { t: 20, truth: 20, pureIns: 19.4, aiIns: 19.8, corrected: 20 },
  { t: 30, truth: 30, pureIns: 28.7, aiIns: 29.7, corrected: 30 },
  { t: 40, truth: 40, pureIns: 37.5, aiIns: 39.5, corrected: 40 },
  { t: 50, truth: 50, pureIns: 46.0, aiIns: 49.3, corrected: 50 },
  { t: 60, truth: 60, pureIns: 54.0, aiIns: 59.1, corrected: 60 },
  { t: 70, truth: 70, pureIns: 61.5, aiIns: 68.9, corrected: 70 },
  { t: 80, truth: 80, pureIns: 68.5, aiIns: 78.7, corrected: 80 },
  { t: 90, truth: 90, pureIns: 75.0, aiIns: 88.5, corrected: 90 },
  { t: 100, truth: 100, pureIns: 81.0, aiIns: 98.2, corrected: 100 },
];

// Interpolate the benchmark data to one point per second.
// This makes the simulation animate smoothly instead of jumping every 10 seconds.
function createInterpolatedTrajectory(data) {
  const result = [];

  for (let second = 0; second <= 100; second += 1) {
    const exactPoint = data.find((point) => point.t === second);

    if (exactPoint) {
      result.push(exactPoint);
      continue;
    }

    const previousPoint =
      [...data].reverse().find((point) => point.t < second) || data[0];

    const nextPoint =
      data.find((point) => point.t > second) || data[data.length - 1];

    const range = nextPoint.t - previousPoint.t;
    const ratio =
      range === 0 ? 0 : (second - previousPoint.t) / range;

    result.push({
      t: second,
      truth:
        previousPoint.truth +
        (nextPoint.truth - previousPoint.truth) * ratio,
      pureIns:
        previousPoint.pureIns +
        (nextPoint.pureIns - previousPoint.pureIns) * ratio,
      aiIns:
        previousPoint.aiIns +
        (nextPoint.aiIns - previousPoint.aiIns) * ratio,
      corrected:
        previousPoint.corrected +
        (nextPoint.corrected - previousPoint.corrected) * ratio,
    });
  }

  return result;
}

// One-second resolution makes the live dashboard smoother.
const simulationData = createInterpolatedTrajectory(trajectoryData);

function App() {
  // Fixed duplicate state declarations from the previous version.
  const [gnssLost, setGnssLost] = useState(false);

  // Controls which navigation solution the user wants to demonstrate.
  const [selectedMode, setSelectedMode] = useState("Pure INS");

  // Controls whether the demo timer is currently running.
  const [isRunning, setIsRunning] = useState(false);

  // Start at 100 so the complete benchmark is visible before simulation starts.
  const [simulationTime, setSimulationTime] = useState(100);

  // The previous simulationRunning state was redundant and has been removed.
  // isRunning is now the single source of truth for simulation status.

  // Navigation mode is now valid JavaScript and reflects GNSS availability.
  const navigationMode = gnssLost
    ? "AI DEAD RECKONING"
    : selectedMode === "Pure INS"
      ? "GNSS + INS"
      : selectedMode;

  // Run the simulation clock.
  // The interval is cleaned up automatically when the component unmounts
  // or when the simulation stops.
  useEffect(() => {
    if (!isRunning) return undefined;

    const timer = setInterval(() => {
      setSimulationTime((previousTime) => {
        if (previousTime >= 100) {
          setIsRunning(false);
          return 100;
        }

        return previousTime + 1;
      });
    }, 100);

    return () => clearInterval(timer);
  }, [isRunning]);

  // Only show trajectory data that has been reached by the simulation.
  const visibleTrajectoryData = useMemo(
    () => simulationData.filter((point) => point.t <= simulationTime),
    [simulationTime],
  );

  // Find the latest point currently displayed on the chart.
  const currentPoint =
    visibleTrajectoryData[visibleTrajectoryData.length - 1] ||
    simulationData[0];

  // Select the trajectory corresponding to the chosen operating mode.
  const selectedTrajectoryKey =
    selectedMode === "Pure INS"
      ? "pureIns"
      : selectedMode === "AI + INS"
        ? "aiIns"
        : "corrected";

  // Calculate the current trajectory error against ground truth.
  const calculatedError = Math.abs(
    currentPoint[selectedTrajectoryKey] - currentPoint.truth,
  );

  // Calculate the maximum error observed so far.
  const calculatedMaximumError = visibleTrajectoryData.reduce(
    (maximum, point) => {
      const error = Math.abs(
        point[selectedTrajectoryKey] - point.truth,
      );

      return Math.max(maximum, error);
    },
    0,
  );

  // Use the existing hackathon demo metrics when GNSS is lost.
  // The values gradually increase during the simulation to communicate drift.
  const progress = simulationTime / 100;

  const positionError = gnssLost
    ? 18.4 * progress
    : 0;

  const finalError = gnssLost
    ? 34.7 * progress
    : 0;

  const maximumError = gnssLost
    ? 51.2 * progress
    : 0;

  const drift = gnssLost
    ? 1.9 * progress
    : 0;

  // Use the calculated trajectory error when GNSS is available,
  // otherwise show the GNSS-denied demo metric.
  const displayedPositionError = gnssLost
    ? positionError
    : calculatedError;

  // Start or restart the simulation.
  const handleStartSimulation = () => {
    if (simulationTime >= 100) {
      setSimulationTime(0);
    }

    setIsRunning(true);
  };

  // Toggle GNSS outage while keeping the simulation state intact.
  const handleGnssToggle = () => {
    setGnssLost((previousValue) => !previousValue);
  };

  // Reset every interactive part of the dashboard.
  const handleReset = () => {
    setGnssLost(false);
    setIsRunning(false);
    setSimulationTime(100);
    setSelectedMode("Pure INS");
  };

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div>
          <div className="eyebrow">
            INTELLIGENT VEHICLE NAVIGATION
          </div>

          <h1>GNSS Dead Reckoning Dashboard</h1>

          <p>
            AI-assisted positioning during GNSS-denied operation
          </p>
        </div>

        {/* GNSS connection indicator changes between available/lost states. */}
        <div
          className={`connection ${
            gnssLost ? "lost" : "active"
          }`}
        >
          {gnssLost ? (
            <SatelliteDish size={18} />
          ) : (
            <Satellite size={18} />
          )}

          {gnssLost
            ? "GNSS SIGNAL LOST"
            : "GNSS AVAILABLE"}
        </div>
      </header>

      {/* Status Cards */}
      <section className="status-grid">
        <StatusCard
          icon={<Satellite size={22} />}
          label="GNSS STATUS"
          value={gnssLost ? "LOST" : "AVAILABLE"}
          status={gnssLost ? "danger" : "success"}
        />

        <StatusCard
          icon={<Navigation size={22} />}
          label="NAVIGATION MODE"
          value={navigationMode}
          status={gnssLost ? "warning" : "success"}
        />

        <StatusCard
          icon={<Gauge size={22} />}
          label="VEHICLE SPEED"
          value="42.3 km/h"
        />

        <StatusCard
          icon={<MapPin size={22} />}
          label="POSITION ERROR"
          value={`${displayedPositionError.toFixed(1)} m`}
        />

        <StatusCard
          icon={<Activity size={22} />}
          label="DRIFT"
          value={`${drift.toFixed(1)}%`}
        />
      </section>

      {/* Main Visualization */}
      <section className="panel trajectory-panel">
        <div className="panel-header">
          <div>
            <span className="panel-kicker">
              LIVE NAVIGATION
            </span>

            <h2>Vehicle Trajectory</h2>
          </div>

          <div className="legend">
            <LegendItem
              label="Ground Truth"
              className="truth"
            />

            <LegendItem
              label="Pure INS"
              className="pure"
            />

            <LegendItem
              label="AI + INS"
              className="ai"
            />

            <LegendItem
              label="Corrected"
              className="corrected"
            />
          </div>
        </div>

        {/* Recharts now receives the live simulation subset. */}
        <div className="trajectory-placeholder">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={visibleTrajectoryData}
              margin={{
                top: 20,
                right: 25,
                left: 10,
                bottom: 20,
              }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#1b334e"
              />

              <XAxis
                dataKey="t"
                stroke="#7187a3"
                tick={{
                  fill: "#7187a3",
                  fontSize: 11,
                }}
                label={{
                  value: "Time (s)",
                  position: "insideBottom",
                  offset: -10,
                  fill: "#7187a3",
                }}
              />

              <YAxis
                stroke="#7187a3"
                tick={{
                  fill: "#7187a3",
                  fontSize: 11,
                }}
                label={{
                  value: "Position (m)",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#7187a3",
                }}
              />

              <Tooltip
                contentStyle={{
                  background: "#0b1a2d",
                  border: "1px solid #29435f",
                  borderRadius: "8px",
                  color: "#ffffff",
                }}
              />

              {/* Ground truth remains the reference trajectory. */}
              <Line
                type="monotone"
                dataKey="truth"
                name="Ground Truth"
                stroke="#eef4fb"
                strokeWidth={3}
                dot={false}
                isAnimationActive={false}
              />

              {/* Pure INS shows accumulated inertial drift. */}
              <Line
                type="monotone"
                dataKey="pureIns"
                name="Pure INS"
                stroke="#ff6978"
                strokeWidth={3}
                dot={false}
                isAnimationActive={false}
              />

              {/* AI + INS demonstrates the improved dead-reckoning estimate. */}
              <Line
                type="monotone"
                dataKey="aiIns"
                name="AI + INS"
                stroke="#55d58c"
                strokeWidth={3}
                dot={false}
                isAnimationActive={false}
              />

              {/* Corrected trajectory represents map/fusion correction. */}
              <Line
                type="monotone"
                dataKey="corrected"
                name="Corrected"
                stroke="#5ca9ff"
                strokeWidth={3}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Controls */}
      <section className="panel controls-panel">
        <div className="controls-left">
          <button
            className="btn primary"
            onClick={handleStartSimulation}
            disabled={isRunning}
            aria-label="Start vehicle navigation simulation"
          >
            <Play size={17} />

            {isRunning
              ? "Simulation Running..."
              : "Start Simulation"}
          </button>

          <button
            className={`btn ${
              gnssLost ? "danger" : "warning"
            }`}
            onClick={handleGnssToggle}
            aria-label={
              gnssLost
                ? "Restore GNSS signal"
                : "Simulate GNSS outage"
            }
          >
            <AlertTriangle size={17} />

            {gnssLost
              ? "Restore GNSS"
              : "Simulate GNSS Outage"}
          </button>

          <button
            className="btn secondary"
            onClick={handleReset}
            aria-label="Reset navigation dashboard"
          >
            <RotateCcw size={17} />
            Reset
          </button>
        </div>

        {/* Controlled radio buttons fix duplicate/uncontrolled mode state. */}
        <div className="mode-selector">
          <span>MODE</span>

          <label>
            <input
              type="radio"
              name="mode"
              value="Pure INS"
              checked={selectedMode === "Pure INS"}
              onChange={() =>
                setSelectedMode("Pure INS")
              }
            />

            Pure INS
          </label>

          <label>
            <input
              type="radio"
              name="mode"
              value="AI + INS"
              checked={selectedMode === "AI + INS"}
              onChange={() =>
                setSelectedMode("AI + INS")
              }
            />

            AI + INS
          </label>

          <label>
            <input
              type="radio"
              name="mode"
              value="AI + INS + Correction"
              checked={
                selectedMode ===
                "AI + INS + Correction"
              }
              onChange={() =>
                setSelectedMode(
                  "AI + INS + Correction",
                )
              }
            />

            AI + INS + Correction
          </label>
        </div>
      </section>

      {/* Bottom Metrics */}
      <section className="metrics-grid">
        <MetricCard
          label="DISTANCE TRAVELLED"
          value={(1.82 * progress).toFixed(2)}
          unit="km"
        />

        <MetricCard
          label="FINAL POSITION ERROR"
          value={finalError.toFixed(1)}
          unit="m"
        />

        <MetricCard
          label="MAXIMUM ERROR"
          value={maximumError.toFixed(1)}
          unit="m"
        />

        <MetricCard
          label="DRIFT"
          value={drift.toFixed(1)}
          unit="%"
        />
      </section>

      {/* Footer */}
      <footer>
        GNSS-RESILIENT VEHICLE NAVIGATION • AI + INS • SIH 2026
      </footer>
    </div>
  );
}

function StatusCard({
  icon,
  label,
  value,
  status,
}) {
  return (
    <div className="status-card">
      <div
        className={`status-icon ${
          status || ""
        }`}
      >
        {icon}
      </div>

      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function LegendItem({
  label,
  className,
}) {
  return (
    <div className="legend-item">
      <i
        className={`legend-line ${className}`}
        aria-hidden="true"
      />

      {label}
    </div>
  );
}

function MetricCard({
  label,
  value,
  unit,
}) {
  return (
    <div className="metric-card">
      <span>{label}</span>

      <div>
        <strong>{value}</strong>
        <small>{unit}</small>
      </div>
    </div>
  );
}

export default App;