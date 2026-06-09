"""
End-to-End Agent Invocation Test
=================================
Generates synthetic smartwatch biometric readings, feeds them through the
full stack, and — when the danger gate fires — invokes all 10 LangGraph agents.

Pipeline:
  Random watch readings
    -> StandardScaler normalization (fitted on WESAD dataset)
    -> DangerLSTM  (distress probability * 100 = danger score 0-100)
    -> ContextualGate  (5 gates: smoother, threshold, multi-signal, duration, dismiss)
    -> [if gate fires] LangGraph 10-Agent Pipeline
         Agent 1  DangerScore    (LLM + rule fallback)
         Agent 2  SOSBroadcast   (LLM + rule fallback)
         Agent 3  StationFinder  (SQLite, nearest Hyderabad police station)
         Agent 4  PathPlanner    (3-waypoint GPS flight plan)
         Agent 5  Dispatch       (reserves drone, calls mock server)
         Agent 6  Tracking       (polls drone telemetry, detects arrival)
         Agent 7  MalfunctionMonitor (battery + fault checks)
         Agent 8  Handoff        (backup drone, LLM context briefing)
         Agent 9  SceneIntelligence  (LLM threat classification)
         Agent 10 ManagementNotify   (LLM supervisor update)

Two scenarios run back-to-back:
  EXERCISE  — elevated movement + breathing, low EDA  -> gate must NOT fire
  DISTRESS  — EDA spike, HRV drop, hyperventilation   -> gate MUST fire -> agents invoked

Run:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe simulation/e2e_agent_invoke_test.py

Requirements:
  - conda activate LLM_GPU
  - data/processed/all_subjects.npz  (run biometric_ml/run_preprocessing.py if missing)
  - biometric_ml/models/best_danger_model.pt  (run biometric_ml/lstm_classifier.py if missing)
  - .env with OPENROUTER_API_KEY  (LLM agents use rule-based fallback if key is missing)
  - Mock drone server is started automatically inside this script (port 8098)
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.contextual_gate import ContextualGate, check_multi_signal
from agents.database import ensure_schema, reset_all_drones
from agents.drone_mock import app as drone_app
from agents.agent_graph import run_sos

# ─── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR   = ROOT / "biometric_ml" / "models"
PROCESSED    = ROOT / "data" / "processed" / "all_subjects.npz"
SEQ_LEN      = 8       # 8 consecutive 30s windows = 40s temporal context
N_FEATURES   = 16
WINDOW_SEC   = 5       # each window represents 5 seconds of sensor data
MOCK_PORT    = 8098    # dedicated port for this test (avoids conflicts)

FEATURE_NAMES = [
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_lf_hf", "hrv_mean_rr", "hrv_std_rr",
    "eda_mean", "eda_std", "eda_phasic_peaks", "eda_phasic_amplitude", "eda_slope",
    "eda_tonic_mean", "acc_mean_mag", "acc_std_mag", "acc_spectral_entropy",
    "acc_breathing_rate",
]

# Victim GPS (Hyderabad — near Madhapur PS for fast arrival in test)
VICTIM_LAT = 17.4495
VICTIM_LON = 78.3878

RNG = np.random.default_rng(seed=42)


# ─── LSTM model (local copy so no circular import from lstm_classifier.py) ───

class DangerLSTM(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, n_layers,
                            dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# ─── Prototype feature vectors (raw WESAD feature space) ─────────────────────
#
# BASELINE  — resting, walking home (safe medians from WESAD)
# ESCALATING— noticing a threat (values midway to distress)
# DISTRESS  — acute fear: EDA matches WESAD distress median, HRV shows sympathetic
# EXERCISE  — running: elevated ACC + breathing, EDA stays below distress threshold

BASELINE = np.array([
    0.30, 0.23, 0.71, 0.46, 0.83, 0.23,    # HRV
    0.63, 0.012, 3.0, 0.010, -0.0002, 0.63, # EDA
    63.4, 1.36, 0.92, 15.0,                 # ACC
], dtype=np.float32)

ESCALATING = np.array([
    0.18, 0.15, 0.35, 1.80, 0.74, 0.15,    # HRV: rmssd dropping, lf_hf rising
    1.20, 0.030, 4.0, 0.025, 0.0001, 1.20, # EDA: conductance rising
    65.0, 2.0,  0.89, 18.0,                # ACC: slight movement increase
], dtype=np.float32)

DISTRESS = np.array([
    0.12, 0.09, 0.12, 3.20, 0.68, 0.09,    # HRV: low rmssd (<0.20), high lf_hf (>2.0)
    2.10, 0.052, 6.0, 0.040, 0.0002, 2.10, # EDA: matches WESAD distress median, tonic=2.10
    58.0, 3.10, 0.86, 22.0,                # ACC: frozen body + hyperventilation (>20)
], dtype=np.float32)

EXERCISE = np.array([
    0.24, 0.19, 0.45, 1.00, 0.62, 0.19,    # HRV: moderate (HR ~97 bpm)
    0.82, 0.022, 2.0, 0.013, 0.00003, 0.82,# EDA: mild thermosweat, tonic=0.82 (<0.90 threshold)
    95.0, 4.80, 0.72, 25.0,                # ACC: high movement, breathing elevated
], dtype=np.float32)


def _noisy(vec: np.ndarray, scale: float = 0.03) -> np.ndarray:
    """Add small realistic sensor noise proportional to signal magnitude."""
    return (vec + RNG.standard_normal(len(vec)) * scale * np.abs(vec)).astype(np.float32)


def _build_exercise_windows(n: int = 40) -> np.ndarray:
    windows = []
    for i in range(n):
        if i < 3:
            w = _noisy(BASELINE)
        elif i < 8:
            a = (i - 3) / 5.0
            w = _noisy((1 - a) * BASELINE + a * EXERCISE)
        else:
            w = _noisy(EXERCISE)
        windows.append(w)
    return np.array(windows, dtype=np.float32)


def _build_distress_windows(n: int = 40) -> np.ndarray:
    windows = []
    for i in range(n):
        if i < 8:
            w = _noisy(BASELINE)
        elif i < 16:
            a = (i - 8) / 8.0
            w = _noisy((1 - a) * BASELINE + a * ESCALATING)
        else:
            a = min(1.0, (i - 16) / 6.0)
            w = _noisy((1 - a) * ESCALATING + a * DISTRESS)
        windows.append(w)
    return np.array(windows, dtype=np.float32)


# ─── Mock drone server ────────────────────────────────────────────────────────

def _start_mock_server(port: int) -> uvicorn.Server:
    config = uvicorn.Config(drone_app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # wait until /health returns 200
    import httpx
    for _ in range(30):
        try:
            httpx.get(f"http://localhost:{port}/health", timeout=1.0).raise_for_status()
            return server
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"Mock drone server did not start on port {port}")


# ─── Logging helpers ──────────────────────────────────────────────────────────

def _hline(char: str = "-", width: int = 68) -> None:
    print(f"  {char * width}")

def _section(title: str) -> None:
    print()
    _hline("=")
    print(f"  {title}")
    _hline("=")

def _step(step: int, total: int, label: str) -> None:
    print(f"\n  -- STEP {step}/{total}: {label}")


# ─── Main test ────────────────────────────────────────────────────────────────

def run_e2e_test() -> None:
    _section("Guardian Drone - End-to-End Agent Invocation Test")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Root    : {ROOT}")

    # ── STEP 1: Start mock drone server ───────────────────────────────────────
    _step(1, 6, "Starting mock drone server")
    server_url = f"http://localhost:{MOCK_PORT}"
    server = _start_mock_server(MOCK_PORT)
    print(f"  Mock drone server started at {server_url}")
    print(f"  Routes: /drone/{{id}}/launch | /telemetry | /inject_fault | /health")

    # ── STEP 2: Load WESAD scaler ─────────────────────────────────────────────
    _step(2, 6, "Fitting normalization scaler on WESAD dataset")
    if not PROCESSED.exists():
        print(f"  ERROR: {PROCESSED} not found.")
        print("  Run: C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe biometric_ml/run_preprocessing.py")
        return

    d = np.load(PROCESSED, allow_pickle=True)
    X_all = d["X"].astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_all)
    print(f"  Scaler fitted on {len(X_all):,} windows from {len(np.unique(d['subject_ids']))} WESAD subjects.")
    print(f"  Features: {', '.join(str(n) for n in d['feature_names'])}")

    # ── STEP 3: Load LSTM model ────────────────────────────────────────────────
    _step(3, 6, "Loading DangerLSTM model")
    model_path = MODELS_DIR / "best_danger_model.pt"
    if not model_path.exists():
        print(f"  ERROR: {model_path} not found.")
        print("  Run: C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe -m biometric_ml.lstm_classifier")
        return

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = DangerLSTM()
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded: {model_path.name}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}  |  Input: (batch, {SEQ_LEN}, {N_FEATURES})")

    # ── STEP 4: Print prototype feature values ─────────────────────────────────
    _step(4, 6, "Generating synthetic watch readings")

    print(f"\n  Prototype feature vectors (raw sensor values):")
    print(f"  {'Feature':<22}  {'BASELINE':>9}  {'EXERCISE':>9}  {'DISTRESS':>9}")
    _hline()
    for i, name in enumerate(FEATURE_NAMES):
        b_mark = ""
        e_mark = ""
        d_mark = ""
        # Annotate values that affect the gate
        if   i == 0:  d_mark = " <- rmssd<0.20 (HRV gate)"
        elif i == 3:  d_mark = " <- lf_hf>2.0 (HRV gate)"
        elif i == 11: d_mark = " <- tonic>0.90 (GSR gate)"
        elif i == 8:  d_mark = " <- peaks>2 (GSR gate)"
        elif i == 15:
            e_mark = " <- >20 (Breathing gate)"
            d_mark = " <- >20 (Breathing gate)"
        print(f"  [{i:2d}] {name:<20}  {BASELINE[i]:>9.4f}  "
              f"{EXERCISE[i]:>9.4f}{e_mark}  "
              f"{DISTRESS[i]:>9.4f}{d_mark}")

    print(f"\n  Multi-signal gate check for each prototype:")
    for label, vec in [("Baseline",  BASELINE),
                       ("Exercise",  EXERCISE),
                       ("Distress",  DISTRESS)]:
        agree = check_multi_signal(vec)
        status = "AGREE (>=2/3 signals in distress range)" if agree \
                 else "NOT AGREE (<2/3) — gate will BLOCK even if score is high"
        print(f"    {label:<12}: {status}")

    print(f"\n  Building windows:")
    ex_windows   = _build_exercise_windows(n=60)
    dist_windows = _build_distress_windows(n=60)
    print(f"    Exercise scenario : {len(ex_windows)} windows x {WINDOW_SEC}s = "
          f"{len(ex_windows)*WINDOW_SEC}s of data")
    print(f"    Distress scenario : {len(dist_windows)} windows x {WINDOW_SEC}s = "
          f"{len(dist_windows)*WINDOW_SEC}s of data")

    # ── STEP 5: Run LSTM + Gate simulation ────────────────────────────────────
    _step(5, 6, "Running LSTM + ContextualGate (streaming window-by-window)")

    def stream_windows(windows: np.ndarray, label: str) -> tuple[bool, list[float], list[dict]]:
        """
        Feed windows through LSTM + gate one at a time.
        Returns (sos_fired, all_scores, all_raw_features_at_sos).
        """
        seq_buf = deque(maxlen=SEQ_LEN)
        gate    = ContextualGate()
        all_scores: list[float] = []
        sos_fired_at: float | None = None
        sos_raw_features: np.ndarray | None = None

        print(f"\n  {'-'*68}")
        print(f"  Scenario: {label}")
        print(f"  {'-'*68}")
        print(f"  {'Time':>6}  {'RawScore':>9}  {'Smoothed':>9}  {'MultiSig':>8}  "
              f"{'GateStatus':<26}")
        print(f"  {'-'*68}")

        with torch.no_grad():
            for i, raw_w in enumerate(windows):
                t = float(i * WINDOW_SEC)

                # Normalize using WESAD scaler
                norm_w = scaler.transform(raw_w.reshape(1, -1))[0].astype(np.float32)
                seq_buf.append(norm_w)

                if len(seq_buf) < SEQ_LEN:
                    continue   # need at least 8 windows before first prediction

                # LSTM inference
                seq    = np.array(seq_buf, dtype=np.float32)[np.newaxis]  # (1, 8, 16)
                logits = model(torch.from_numpy(seq))
                prob   = float(torch.softmax(logits, dim=-1)[0, 1].item())
                score  = prob * 100.0
                all_scores.append(score)

                # Multi-signal check on raw (un-normalized) features
                agree = check_multi_signal(raw_w)

                # ContextualGate decision
                dec = gate.process(raw_score=score, multi_signal_agree=agree, now=t)

                sos_marker = "  <<< SOS FIRED" if dec.should_sos else ""
                print(f"  {t:>6.0f}s  {score:>9.1f}  {dec.smoothed_score:>9.1f}  "
                      f"{'YES' if agree else 'NO':>8}  "
                      f"{dec.blocked_by:<26}{sos_marker}")

                if dec.should_sos and sos_fired_at is None:
                    sos_fired_at      = t
                    sos_raw_features  = raw_w.copy()

        print(f"  {'-'*68}")
        if sos_fired_at is not None:
            print(f"  GATE RESULT: SOS fired at t={sos_fired_at:.0f}s  "
                  f"(avg score {sum(all_scores)/len(all_scores):.1f})")
        else:
            print(f"  GATE RESULT: Gate held - NO SOS  "
                  f"(avg score {sum(all_scores)/len(all_scores):.1f})")

        return (sos_fired_at is not None), all_scores, (
            sos_raw_features.tolist() if sos_raw_features is not None else []
        )

    # Run Exercise scenario
    ex_fired, ex_scores, _  = stream_windows(ex_windows,   "EXERCISE  (running)")

    # Run Distress scenario
    dist_fired, dist_scores, dist_features = stream_windows(dist_windows, "DISTRESS  (threat / attack)")

    # ── Summary of gate results ────────────────────────────────────────────────
    print(f"\n  GATE SUMMARY:")
    print(f"    Exercise -> SOS? {'YES (false positive!)' if ex_fired else 'NO  - correct'}")
    print(f"    Distress -> SOS? {'YES - correct'         if dist_fired else 'NO  (missed!)'}")

    gate_pass = (not ex_fired) and dist_fired

    # ── STEP 6: Invoke LangGraph agents on distress event ─────────────────────
    _step(6, 6, "Invoking 10 LangGraph Agents on distress SOS event")

    if not dist_fired:
        print("  Distress gate did not fire — adjusting danger score for agent demo.")
        # Force a high score to demonstrate agent invocation
        dist_features = DISTRESS.tolist()
        agent_score   = 88.0
    else:
        agent_score = dist_scores[-1] if dist_scores else 88.0

    print(f"\n  Input to agent pipeline:")
    print(f"    danger_score   : {agent_score:.1f} / 100")
    print(f"    victim_lat     : {VICTIM_LAT}")
    print(f"    victim_lon     : {VICTIM_LON}")
    print(f"    raw_features   : {len(dist_features)} features (distress biometrics)")
    print(f"    drone_server   : {server_url}")
    print()
    print("  Invoking agents (LLM calls via OpenRouter — ~20-30s expected) ...")
    print()

    # Reset DB drone availability before agent run
    ensure_schema()
    reset_all_drones()

    t_agents_start = time.perf_counter()

    final_state = run_sos(
        danger_score     = agent_score,
        victim_lat       = VICTIM_LAT,
        victim_lon       = VICTIM_LON,
        max_iterations   = 5,
        drone_server_url = server_url,
        raw_features     = dist_features,
    )

    agent_elapsed = time.perf_counter() - t_agents_start

    # ── Final report ──────────────────────────────────────────────────────────
    _section("Final Results")

    print(f"  GATE TEST:")
    print(f"    Exercise scenario   SOS fired? {'YES (BUG)' if ex_fired else 'NO  [correct]'}")
    print(f"    Distress scenario   SOS fired? {'YES [correct]' if dist_fired else 'NO  (MISS)'}")
    print(f"    Gate test overall:  {'PASS' if gate_pass else 'PARTIAL'}")

    print(f"\n  AGENT PIPELINE (duration: {agent_elapsed:.1f}s):")
    print(f"    Agent 1  go_decision      : {final_state.get('go_decision')}")

    if final_state.get('sos_alert'):
        alert = final_state['sos_alert']
        print(f"    Agent 2  SOS priority     : {alert.priority.upper()}")
        print(f"    Agent 2  SOS message      : {alert.message}")

    if final_state.get('nearest_station'):
        st = final_state['nearest_station']
        print(f"    Agent 3  Nearest station  : {st.name} ({st.distance_km:.2f} km)")

    if final_state.get('flight_path'):
        fp = final_state['flight_path']
        km = final_state.get('path_distance_km', 0)
        print(f"    Agent 4  Flight path      : {len(fp)} waypoints, {km:.3f} km")

    if final_state.get('dispatch_result'):
        dr = final_state['dispatch_result']
        print(f"    Agent 5  Drone dispatched : {dr.drone_id}  ETA={dr.eta_seconds:.0f}s")

    if final_state.get('drone_telemetry'):
        tl = final_state['drone_telemetry']
        print(f"    Agent 6  Last telemetry   : pos=({tl.lat:.4f},{tl.lon:.4f}) "
              f"bat={tl.battery_pct:.0f}% status={tl.status}")

    print(f"    Agent 7  Malfunction flag : {final_state.get('malfunction_flag', False)}")

    if final_state.get('scene_classification'):
        sc = final_state['scene_classification']
        print(f"    Agent 9  Scene type       : {sc.scene_type} | "
              f"threat={sc.threat_level}/5")
        print(f"    Agent 9  Action           : {sc.recommended_action}")

    if final_state.get('management_update'):
        mu = final_state['management_update']
        print(f"    Agent 10 Event type       : {mu.event_type}")
        print(f"    Agent 10 Status msg       : {mu.status_message}")

    print(f"\n    Final mission_status : {final_state.get('mission_status', 'unknown')}")
    print(f"    Notifications sent   : {final_state.get('notifications_sent', 0)}")

    if final_state.get('errors'):
        print(f"\n    Non-fatal errors during pipeline:")
        for e in final_state['errors']:
            print(f"      - {e}")

    overall_pass = (
        gate_pass
        and final_state.get('go_decision') is True
        and final_state.get('dispatch_result') is not None
        and (final_state.get('dispatch_result').success if final_state.get('dispatch_result') else False)
    )

    _hline("=")
    status_str = "ALL SYSTEMS GO" if overall_pass else "COMPLETED WITH WARNINGS"
    print(f"  E2E TEST: {status_str}")
    _hline("=")

    # ── Stop mock server ──────────────────────────────────────────────────────
    server.should_exit = True

    # ── HOW TO RUN MANUALLY ───────────────────────────────────────────────────
    _section("How to run each component manually")

    print("""
  1. ACTIVATE ENVIRONMENT
     conda activate LLM_GPU
     cd D:/My_Project_Drone

  2. RUN MOCK DRONE SERVER  (terminal 1 - keep it running)
     python agents/drone_mock.py
     Starts on http://localhost:8001
     Test with: curl http://localhost:8001/health

  3. RUN THIS END-TO-END TEST  (terminal 2)
     python simulation/e2e_agent_invoke_test.py
     Starts its own server on port 8098 automatically

  4. RUN JUST THE LSTM + GATE SCENARIO TEST  (no agents)
     python simulation/scenario_test.py
     Shows score traces + gate decisions for exercise vs distress

  5. RUN JUST THE AGENT PIPELINE  (if gate already fired)
     python agents/agent_graph.py 85.0
     Invokes all 10 agents with score=85, victim at Madhapur PS

  6. RUN UNIT TESTS (no network needed for Phase 1)
     python -m pytest tests/test_preprocessing.py tests/test_contextual_gate.py -v

  7. RUN AGENT INTEGRATION TESTS (needs internet for OpenRouter)
     python -m pytest tests/test_agents.py -v -s
     5 tests, ~105s (3 LLM round-trips per full-pipeline test)

  8. VIEW ALL PLOTS
     python show_plots.py

  ENVIRONMENT VARIABLES  (in .env file — never commit):
     OPENROUTER_API_KEY=sk-or-...  (LLM agents fall back to rules if missing)
    """)


if __name__ == "__main__":
    run_e2e_test()
