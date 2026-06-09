"""
tests/test_simulation.py — Phase 3 simulation module tests

Runs entirely without AirSim or PX4 SITL by using MockBridge.
All tests are fast (< 5 s total) and work in CI.

Test groups
-----------
  test_environment_*    — coordinate conversions, obstacles, victim trajectory
  test_scenario_*       — MockBridge fault injection + scenario runner logic
  test_report_*         — CSV write/read round-trip and report generation
"""
from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import pytest
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulation.environment import (
    ned_to_gps, gps_to_ned, WORLD_OBSTACLES, DRONE_STATION_1,
    VictimTrajectory, WeatherConfig, Weather, WEATHER_PRESETS,
    safe_flight_altitude, is_path_obstructed, ORIGIN_LAT, ORIGIN_LON,
)
from simulation.scenario_runner import (
    MockBridge, Scenario, RunResult, SCENARIOS,
    _run_scenario_once, run_all_scenarios,
)
from simulation.report_generator import load_csv, generate_report


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_bridge():
    return MockBridge()


# ─── Environment tests ───────────────────────────────────────────────────────

class TestEnvironment:

    def test_ned_to_gps_origin(self):
        lat, lon, alt = ned_to_gps(0, 0, 0)
        assert abs(lat - ORIGIN_LAT) < 1e-9
        assert abs(lon - ORIGIN_LON) < 1e-9

    def test_ned_to_gps_north_increases_lat(self):
        lat0, _, _ = ned_to_gps(0, 0, 0)
        lat1, _, _ = ned_to_gps(100, 0, 0)
        assert lat1 > lat0

    def test_ned_to_gps_east_increases_lon(self):
        _, lon0, _ = ned_to_gps(0, 0, 0)
        _, lon1, _ = ned_to_gps(0, 100, 0)
        assert lon1 > lon0

    def test_gps_to_ned_round_trip(self):
        north_in, east_in = 123.4, -56.7
        lat, lon, alt_msl = ned_to_gps(north_in, east_in, 10.0)
        north_out, east_out, _ = gps_to_ned(lat, lon, alt_msl)
        assert abs(north_out - north_in) < 0.01
        assert abs(east_out - east_in) < 0.01

    def test_obstacles_non_empty(self):
        assert len(WORLD_OBSTACLES) > 0

    def test_obstacle_gps_in_range(self):
        for obs in WORLD_OBSTACLES:
            lat, lon, _ = obs.gps
            assert 40 < lat < 55, f"{obs.name} lat out of range"
            assert -130 < lon < -100, f"{obs.name} lon out of range"

    def test_obstacle_clearance_above_height(self):
        for obs in WORLD_OBSTACLES:
            assert obs.clearance_alt > obs.height_m

    def test_safe_flight_altitude(self):
        alt = safe_flight_altitude(WORLD_OBSTACLES)
        max_clearance = max(o.clearance_alt for o in WORLD_OBSTACLES)
        assert abs(alt - max_clearance) < 0.01

    def test_safe_flight_altitude_empty(self):
        assert safe_flight_altitude([]) == 10.0

    def test_path_obstruction_low_altitude(self):
        # flying directly over obstacles at 5 m should be blocked
        blocking = is_path_obstructed((0, 0), (60, 0), WORLD_OBSTACLES, drone_alt_m=5.0)
        assert len(blocking) > 0

    def test_path_obstruction_high_altitude(self):
        # flying at 50 m should clear all obstacles
        blocking = is_path_obstructed((0, 0), (60, 0), WORLD_OBSTACLES, drone_alt_m=50.0)
        assert len(blocking) == 0

    def test_path_empty_obstacles(self):
        blocking = is_path_obstructed((0, 0), (100, 0), [], drone_alt_m=5.0)
        assert blocking == []


class TestVictimTrajectory:

    @pytest.mark.parametrize("ttype", ["walking", "panicked", "stationary", "circling"])
    def test_generates_waypoints(self, ttype):
        traj = VictimTrajectory(ttype, n_waypoints=10)
        assert len(traj.waypoints) == 10

    def test_walking_moves_forward(self):
        traj = VictimTrajectory("walking", n_waypoints=10)
        lat0 = traj.waypoints[0].lat
        lat9 = traj.waypoints[-1].lat
        # walking north → lat increases
        assert lat9 > lat0

    def test_stationary_stays_close(self):
        traj = VictimTrajectory("stationary", n_waypoints=10)
        lats = [wp.lat for wp in traj.waypoints]
        # all within ~1 metre of each other
        assert (max(lats) - min(lats)) * 111_320 < 1.0

    def test_panicked_has_high_speed(self):
        traj = VictimTrajectory("panicked", n_waypoints=5)
        for wp in traj.waypoints:
            assert wp.speed_ms >= 3.0

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            VictimTrajectory("unknown_type", n_waypoints=5)

    def test_deterministic_with_same_seed(self):
        t1 = VictimTrajectory("walking", seed=7, n_waypoints=5)
        t2 = VictimTrajectory("walking", seed=7, n_waypoints=5)
        for w1, w2 in zip(t1.waypoints, t2.waypoints):
            assert w1.lat == w2.lat
            assert w1.lon == w2.lon


class TestWeatherPresets:

    def test_all_presets_defined(self):
        for w in Weather:
            assert w in WEATHER_PRESETS

    def test_night_has_zero_sun(self):
        cfg = WEATHER_PRESETS[Weather.NIGHT]
        assert cfg.sun_angle == 0.0

    def test_fog_has_high_density(self):
        cfg = WEATHER_PRESETS[Weather.FOG]
        assert cfg.fog_density >= 0.5

    def test_clear_has_no_weather(self):
        cfg = WEATHER_PRESETS[Weather.CLEAR]
        assert cfg.fog_density == 0.0
        assert cfg.rain_amount == 0.0
        assert cfg.dust_amount == 0.0


# ─── Scenario runner tests ────────────────────────────────────────────────────

class TestMockBridge:

    def test_launch_returns_success(self, mock_bridge):
        result = mock_bridge.launch_drone("DRONE_1", 1, 47.641, -122.14, 30.0)
        assert result["success"] is True

    def test_get_telemetry_returns_flying(self, mock_bridge):
        # Fresh bridge: no faults
        bridge = MockBridge()
        telem  = bridge.get_telemetry("DRONE_1")
        assert telem.status == "flying"
        assert telem.battery_pct == 85.0

    def test_inject_battery_fault(self, mock_bridge):
        bridge = MockBridge()
        bridge.inject_fault("DRONE_1", "battery_critical")
        telem = bridge.get_telemetry("DRONE_1")
        assert telem.battery_pct == 5.0
        assert telem.fault_code == "battery_critical"

    def test_inject_camera_fault(self, mock_bridge):
        bridge = MockBridge()
        bridge.inject_fault("DRONE_1", "camera_failure")
        telem = bridge.get_telemetry("DRONE_1")
        assert telem.camera_active is False

    def test_inject_motor_fault(self, mock_bridge):
        bridge = MockBridge()
        bridge.inject_fault("DRONE_1", "motor_failure")
        telem = bridge.get_telemetry("DRONE_1")
        assert telem.motor_status == "failed"
        assert telem.status == "fault"

    def test_inject_gps_fault(self, mock_bridge):
        bridge = MockBridge()
        bridge.inject_fault("DRONE_1", "gps_loss")
        telem = bridge.get_telemetry("DRONE_1")
        assert telem.fault_code == "gps_loss"

    def test_disconnect_clears_faults(self, mock_bridge):
        bridge = MockBridge()
        bridge.inject_fault("DRONE_1", "battery_critical")
        bridge.disconnect_all()
        telem = bridge.get_telemetry("DRONE_1")
        assert telem.battery_pct == 85.0


class TestScenarioRunner:

    def test_all_scenarios_defined(self):
        ids = {s.id for s in SCENARIOS}
        assert ids == {"S1", "S2", "S3", "S4", "S5"}

    def test_sla_values_positive(self):
        for s in SCENARIOS:
            assert s.sla_s > 0

    def test_run_result_passed_property(self):
        r = RunResult("S1", 1, "battery_critical", True, 5.0, 30.0, "DRONE_1")
        assert r.passed is True

    def test_run_result_failed_if_over_sla(self):
        r = RunResult("S1", 1, "battery_critical", True, 35.0, 30.0, "DRONE_1")
        assert r.passed is False

    def test_run_result_failed_if_no_success(self):
        r = RunResult("S1", 1, "battery_critical", False, 5.0, 30.0, "DRONE_1")
        assert r.passed is False

    def test_dry_run_single_scenario(self):
        bridge = MockBridge()
        s1 = next(s for s in SCENARIOS if s.id == "S1")
        s1_copy = Scenario(s1.id, s1.name, s1.fault_type, s1.sla_s, n_runs=2)
        results = run_all_scenarios(bridge, [s1_copy], verbose=False)
        assert len(results) == 2
        assert all(r.scenario_id == "S1" for r in results)

    def test_dry_run_all_scenarios(self):
        bridge = MockBridge()
        scenarios_copy = [
            Scenario(s.id, s.name, s.fault_type, s.sla_s, n_runs=1)
            for s in SCENARIOS
        ]
        results = run_all_scenarios(bridge, scenarios_copy, verbose=False)
        assert len(results) == 5

    def test_csv_round_trip(self):
        bridge = MockBridge()
        s1 = Scenario("S1", "Battery Critical", "battery_critical", 30.0, n_runs=2)
        results = run_all_scenarios(bridge, [s1], verbose=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        from simulation.scenario_runner import _write_csv
        _write_csv(results, csv_path)

        rows = load_csv(csv_path)
        assert len(rows) == 2
        assert rows[0]["scenario_id"] == "S1"
        assert rows[0]["fault_type"]  == "battery_critical"
        assert isinstance(rows[0]["latency_s"], float)
        assert isinstance(rows[0]["passed"], bool)

        csv_path.unlink(missing_ok=True)


class TestReportGenerator:

    def test_generate_report_no_crash(self):
        bridge = MockBridge()
        scenarios = [
            Scenario(s.id, s.name, s.fault_type, s.sla_s, n_runs=2)
            for s in SCENARIOS
        ]
        results = run_all_scenarios(bridge, scenarios, verbose=False)
        # Should not raise
        generate_report(results, out_png=None, plot=False)

    def test_generate_report_from_csv(self):
        bridge = MockBridge()
        scenarios = [
            Scenario(s.id, s.name, s.fault_type, s.sla_s, n_runs=1)
            for s in SCENARIOS
        ]
        results = run_all_scenarios(bridge, scenarios, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        from simulation.scenario_runner import _write_csv
        _write_csv(results, csv_path)

        rows = load_csv(csv_path)
        generate_report(rows, out_png=None, plot=False)
        csv_path.unlink(missing_ok=True)
