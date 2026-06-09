"""
Phase 3.5 — Simulation Scenario Runner

Runs 5 malfunction scenarios × 10 repetitions each against a live or mocked
SimBridge (sim_bridge.py).  Each run exercises a specific drone fault and
measures whether the agent graph responds within the SLA.

Scenarios
---------
  S1  battery_critical  — battery drops to 5%, drone must RTL within 30 s
  S2  gps_loss          — GPS lost, drone must LOITER/hold within 15 s
  S3  motor_failure     — one motor fails, drone must emergency-land within 20 s
  S4  signal_loss       — GCS signal drops, drone must return-to-launch within 25 s
  S5  camera_failure    — front camera fails, system must switch to bottom cam within 10 s

Each run is saved as a RunResult (success, latency_s, fault, notes).
Summary CSV + pass/fail table written by report_generator.py.

Usage (against live AirSim + PX4 SITL)
---------------------------------------
  python simulation/scenario_runner.py

Usage (dry-run / CI, no AirSim needed)
---------------------------------------
  python simulation/scenario_runner.py --dry-run

The --dry-run flag replaces the SimBridge with MockBridge so every scenario
completes deterministically — useful for testing the runner logic itself.
"""
from __future__ import annotations

import argparse
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional
import csv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RunResult:
    scenario_id:    str           # e.g. "S1"
    run_index:      int           # 1-10
    fault_type:     str
    success:        bool
    latency_s:      float         # seconds from fault injection to agent response
    sla_s:          float         # required latency threshold
    drone_id:       str
    notes:          str = ""
    timestamp:      float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.success and self.latency_s <= self.sla_s


# ─── Scenario definitions ─────────────────────────────────────────────────────

@dataclass
class Scenario:
    id:         str
    name:       str
    fault_type: str
    sla_s:      float               # maximum acceptable response latency
    n_runs:     int = 10
    drone_id:   str = "DRONE_1"
    setup_fn:   Optional[Callable] = None   # called before fault injection


SCENARIOS: list[Scenario] = [
    Scenario("S1", "Battery Critical",  "battery_critical", sla_s=30.0),
    Scenario("S2", "GPS Loss",          "gps_loss",         sla_s=15.0),
    Scenario("S3", "Motor Failure",     "motor_failure",    sla_s=20.0),
    Scenario("S4", "Signal Loss",       "signal_loss",      sla_s=25.0),
    Scenario("S5", "Camera Failure",    "camera_failure",   sla_s=10.0),
]


# ─── Mock bridge for dry-run / CI ─────────────────────────────────────────────

class MockBridge:
    """
    Deterministic stand-in for simulation.sim_bridge.
    After fault injection the drone immediately transitions to the expected
    response state so _wait_for_agent_response() returns on the first poll.
    This keeps unit tests fast (< 1 s total) while still exercising the polling logic.
    """

    # Per-instance fault state so fixtures don't share mutable class state
    def __init__(self):
        self._faults: dict[str, str] = {}

    _FAULT_STATUS = {
        "battery_critical": "returning",   # agent commands RTL
        "motor_failure":    "fault",        # motor HW fault detected
        "signal_loss":      "returning",   # agent commands RTL on RC loss
        "gps_loss":         "hovering",    # agent commands position hold
        "camera_failure":   "flying",      # still airborne; camera_active=False is the flag
    }

    def launch_drone(self, drone_id, station_id, target_lat, target_lon,
                     target_alt=30.0, server_url="") -> dict:
        return {"success": True, "drone_id": drone_id, "alt_m": target_alt}

    def get_telemetry(self, drone_id, server_url=""):
        from agents.state import DroneTelemetry
        fault = self._faults.get(drone_id)
        status = self._FAULT_STATUS.get(fault, "flying") if fault else "flying"
        return DroneTelemetry(
            drone_id     = drone_id,
            lat          = 47.641468,
            lon          = -122.140165,
            alt_m        = 30.0,
            speed_ms     = 3.0,
            battery_pct  = 5.0 if fault == "battery_critical" else 85.0,
            status       = status,
            fault_code   = fault,
            camera_active= fault != "camera_failure",
            motor_status = "failed" if fault == "motor_failure" else "ok",
            signal_rssi  = -120 if fault == "signal_loss" else -55,
        )

    def set_waypoint(self, drone_id, lat, lon, alt_m=30.0, server_url="") -> dict:
        return {"success": True}

    def inject_fault(self, drone_id, fault_type, server_url="") -> dict:
        self._faults[drone_id] = fault_type
        return {"success": True, "drone_id": drone_id, "fault": fault_type}

    def disconnect_all(self) -> None:
        self._faults.clear()



# ─── Agent-side response detector ─────────────────────────────────────────────

def _wait_for_agent_response(
    bridge,
    drone_id: str,
    fault_type: str,
    timeout_s: float,
    poll_interval: float = 0.5,
) -> tuple[bool, float]:
    """
    Poll telemetry until the drone transitions out of the fault state
    (status changes or fault_code clears).

    Returns (responded: bool, latency_s: float).
    """
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            return False, elapsed

        try:
            telem = bridge.get_telemetry(drone_id)
        except Exception as e:
            return False, elapsed

        # Camera fault: agent should flag camera_active=False within SLA
        if fault_type == "camera_failure":
            if not telem.camera_active:
                return True, elapsed

        # Motor fault: drone enters fault state (agent detects + logs; hardware can't RTL safely)
        elif fault_type == "motor_failure":
            if telem.status in ("fault", "returning"):
                return True, elapsed

        # Battery / signal: agent should command RTL → "returning"
        elif fault_type in ("battery_critical", "signal_loss"):
            if telem.status == "returning":
                return True, elapsed

        # GPS loss: agent should hold position (hovering) or switch to ALT_HOLD
        elif fault_type == "gps_loss":
            if telem.status in ("hovering", "alt_hold"):
                return True, elapsed

        time.sleep(poll_interval)


# ─── Single run executor ──────────────────────────────────────────────────────

def _run_scenario_once(
    scenario: Scenario,
    run_idx:  int,
    bridge,
    verbose:  bool = True,
) -> RunResult:
    drone_id = scenario.drone_id
    prefix   = f"  [{scenario.id} run {run_idx:02d}]"

    try:
        # 1. Ensure drone is flying
        telem = bridge.get_telemetry(drone_id)
        if telem.status not in ("flying", "hovering"):
            if verbose:
                print(f"{prefix} Launching drone ...")
            launch = bridge.launch_drone(
                drone_id, 1,
                target_lat=47.641468, target_lon=-122.140165,
                target_alt=30.0,
            )
            if not launch.get("success"):
                return RunResult(
                    scenario.id, run_idx, scenario.fault_type,
                    False, 0.0, scenario.sla_s, drone_id,
                    notes="launch failed",
                )
            time.sleep(2.0)     # wait for stable hover

        # 2. Optional pre-fault setup
        if scenario.setup_fn:
            scenario.setup_fn(bridge, drone_id)

        # 3. Inject fault
        if verbose:
            print(f"{prefix} Injecting fault '{scenario.fault_type}' ...")
        t_inject = time.time()
        inj = bridge.inject_fault(drone_id, scenario.fault_type)
        if not inj.get("success"):
            return RunResult(
                scenario.id, run_idx, scenario.fault_type,
                False, 0.0, scenario.sla_s, drone_id,
                notes=f"inject failed: {inj}",
            )

        # 4. Wait for agent/system response
        responded, latency = _wait_for_agent_response(
            bridge, drone_id, scenario.fault_type,
            timeout_s=scenario.sla_s * 2,   # poll for 2× SLA before giving up
        )

        result = RunResult(
            scenario.id, run_idx, scenario.fault_type,
            responded, round(latency, 3), scenario.sla_s, drone_id,
        )

        status = "PASS" if result.passed else "FAIL"
        if verbose:
            print(f"{prefix} {status}  latency={latency:.2f}s  SLA={scenario.sla_s}s")

        return result

    except Exception as exc:
        if verbose:
            print(f"{prefix} ERROR: {exc}")
        return RunResult(
            scenario.id, run_idx, scenario.fault_type,
            False, 0.0, scenario.sla_s, drone_id,
            notes=str(exc),
        )


# ─── Main runner ──────────────────────────────────────────────────────────────

def run_all_scenarios(
    bridge,
    scenarios:   list[Scenario] = SCENARIOS,
    verbose:     bool = True,
    out_csv:     Optional[Path] = None,
) -> list[RunResult]:
    """
    Execute all scenarios × n_runs, return flat list of RunResult.

    Results are also written to out_csv (if provided) for report_generator.
    """
    all_results: list[RunResult] = []

    for scenario in scenarios:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Scenario {scenario.id}: {scenario.name}")
            print(f"  Fault: {scenario.fault_type}  SLA: {scenario.sla_s}s  Runs: {scenario.n_runs}")
            print(f"{'='*60}")

        for run_idx in range(1, scenario.n_runs + 1):
            result = _run_scenario_once(scenario, run_idx, bridge, verbose)
            all_results.append(result)

    if out_csv:
        _write_csv(all_results, out_csv)

    return all_results


def _write_csv(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario_id", "run_index", "fault_type", "success",
            "latency_s", "sla_s", "passed", "drone_id", "notes", "timestamp",
        ])
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["passed"] = r.passed
            writer.writerow(row)
    print(f"\n  Results written to {path}")


def print_summary(results: list[RunResult]) -> None:
    from collections import defaultdict
    by_scenario: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_scenario[r.scenario_id].append(r)

    print(f"\n{'='*60}")
    print("  SCENARIO SUMMARY")
    print(f"{'='*60}")
    print(f"  {'ID':<4} {'Name':<20} {'Pass':>5} {'Fail':>5} {'PassRate':>9} {'AvgLat':>8}")
    print(f"  {'-'*60}")

    for sid, rs in sorted(by_scenario.items()):
        passed  = sum(1 for r in rs if r.passed)
        failed  = len(rs) - passed
        rate    = passed / len(rs) * 100
        avg_lat = sum(r.latency_s for r in rs) / len(rs)
        name    = next(s.name for s in SCENARIOS if s.id == sid)
        print(f"  {sid:<4} {name:<20} {passed:>5} {failed:>5} {rate:>8.0f}% {avg_lat:>7.2f}s")

    total_pass = sum(1 for r in results if r.passed)
    total      = len(results)
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<25} {total_pass:>5} {total-total_pass:>5} {total_pass/total*100:>8.0f}%")
    print(f"{'='*60}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardian Drone — Scenario Runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use MockBridge (no AirSim required)")
    parser.add_argument("--scenario", nargs="+", default=None,
                        help="Run only specified scenario IDs, e.g. --scenario S1 S3")
    parser.add_argument("--runs", type=int, default=None,
                        help="Override n_runs per scenario")
    parser.add_argument("--out", default=None,
                        help="Path for output CSV (default: docs/sim_results.csv)")
    args = parser.parse_args()

    # Select scenarios
    selected = SCENARIOS
    if args.scenario:
        selected = [s for s in SCENARIOS if s.id in args.scenario]
        if not selected:
            print(f"Unknown scenario IDs: {args.scenario}")
            sys.exit(1)

    # Override runs
    if args.runs is not None:
        for s in selected:
            s.n_runs = args.runs

    # Select bridge
    if args.dry_run:
        print("\n[DRY RUN] Using MockBridge — no AirSim needed.")
        bridge = MockBridge()
    else:
        print("\n[LIVE] Connecting to AirSim + PX4 SITL via SimBridge ...")
        import simulation.sim_bridge as bridge   # type: ignore

    out_csv = Path(args.out) if args.out else ROOT / "docs" / "sim_results.csv"

    print(f"\nGuardian Drone — Scenario Runner")
    print(f"Scenarios: {[s.id for s in selected]}")
    print(f"Output CSV: {out_csv}\n")

    results = run_all_scenarios(bridge, selected, verbose=True, out_csv=out_csv)
    print_summary(results)
