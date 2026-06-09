"""
Phase 3.7 — Simulation Report Generator

Reads sim_results.csv produced by scenario_runner.py and outputs:
  1. Console summary table (per-scenario pass rates)
  2. docs/sim_report.png  — matplotlib SLA pass/fail bar chart + latency box plot

Usage
-----
  python simulation/report_generator.py                     # reads docs/sim_results.csv
  python simulation/report_generator.py --csv path/to.csv   # custom input
  python simulation/report_generator.py --no-plot           # console only

Can also be called programmatically:
  from simulation.report_generator import generate_report
  generate_report(results, out_png=Path("docs/sim_report.png"))
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["success"]   = row["success"].strip().lower() == "true"
            row["passed"]    = row["passed"].strip().lower()  == "true"
            row["latency_s"] = float(row["latency_s"])
            row["sla_s"]     = float(row["sla_s"])
            row["run_index"] = int(row["run_index"])
            rows.append(row)
    return rows


# ─── Console report ───────────────────────────────────────────────────────────

_SCENARIO_NAMES = {
    "S1": "Battery Critical",
    "S2": "GPS Loss",
    "S3": "Motor Failure",
    "S4": "Signal Loss",
    "S5": "Camera Failure",
}


def print_report(rows: list[dict]) -> None:
    by_sid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sid[r["scenario_id"]].append(r)

    print(f"\n{'='*70}")
    print("  Guardian Drone — Simulation Test Report")
    print(f"{'='*70}")
    print(f"  {'ID':<4} {'Scenario':<20} {'Pass':>5} {'Fail':>5} {'Rate':>7} "
          f"{'AvgLat':>8} {'MaxLat':>8} {'SLA':>6}")
    print(f"  {'-'*70}")

    all_pass = all_total = 0
    for sid in sorted(by_sid):
        rs     = by_sid[sid]
        passed = sum(1 for r in rs if r["passed"])
        failed = len(rs) - passed
        rate   = passed / len(rs) * 100
        avg_l  = sum(r["latency_s"] for r in rs) / len(rs)
        max_l  = max(r["latency_s"] for r in rs)
        sla    = rs[0]["sla_s"]
        name   = _SCENARIO_NAMES.get(sid, sid)
        flag   = "" if rate == 100 else "  !"
        print(f"  {sid:<4} {name:<20} {passed:>5} {failed:>5} {rate:>6.0f}%"
              f" {avg_l:>8.2f}s {max_l:>8.2f}s {sla:>5.0f}s{flag}")
        all_pass  += passed
        all_total += len(rs)

    print(f"  {'─'*70}")
    overall_rate = all_pass / all_total * 100 if all_total else 0
    print(f"  {'TOTAL':<25} {all_pass:>5} {all_total-all_pass:>5} {overall_rate:>6.0f}%")
    verdict = "ALL PASS" if all_pass == all_total else "SOME FAILURES"
    print(f"\n  Result: {verdict}  ({all_pass}/{all_total} runs passed SLA)")
    print(f"{'='*70}\n")


# ─── Plot generation ──────────────────────────────────────────────────────────

def generate_plot(rows: list[dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[report] matplotlib not installed — skipping plot.")
        return

    by_sid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sid[r["scenario_id"]].append(r)

    sids     = sorted(by_sid.keys())
    names    = [_SCENARIO_NAMES.get(s, s) for s in sids]
    pass_rates = [sum(1 for r in by_sid[s] if r["passed"]) / len(by_sid[s]) * 100 for s in sids]
    slas       = [by_sid[s][0]["sla_s"] for s in sids]
    latencies  = [[r["latency_s"] for r in by_sid[s]] for s in sids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Bar chart: pass rates ──────────────────────────────────────────────────
    colours = ["#4CAF50" if pr == 100 else "#F44336" for pr in pass_rates]
    bars = ax1.bar(range(len(sids)), pass_rates, color=colours, edgecolor="white", linewidth=1.5)
    ax1.axhline(100, color="#388E3C", ls="--", lw=1.2, label="100% target")
    ax1.axhline(80,  color="#FB8C00", ls=":",  lw=1.0, alpha=0.7, label="80% minimum")

    for bar, rate in zip(bars, pass_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{rate:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax1.set_xticks(range(len(sids)))
    ax1.set_xticklabels([f"{sid}\n{name}" for sid, name in zip(sids, names)], fontsize=9)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Pass Rate (%)", fontsize=11)
    ax1.set_title("SLA Pass Rate by Scenario", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── Box plot: latency distribution ────────────────────────────────────────
    bp = ax2.boxplot(latencies, patch_artist=True, notch=False,
                     medianprops=dict(color="black", lw=2))
    box_colours = ["#42A5F5"] * len(sids)
    for patch, col in zip(bp["boxes"], box_colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    for i, (sla, lats) in enumerate(zip(slas, latencies)):
        ax2.axhline(sla, color="#E53935", ls="--", lw=1.2,
                    xmin=(i) / len(sids) + 0.02,
                    xmax=(i + 1) / len(sids) - 0.02)
        ax2.text(i + 1.35, sla + 0.3, f"SLA {sla:.0f}s", fontsize=7, color="#B71C1C")

    ax2.set_xticks(range(1, len(sids) + 1))
    ax2.set_xticklabels([f"{sid}\n{name}" for sid, name in zip(sids, names)], fontsize=9)
    ax2.set_ylabel("Response Latency (s)", fontsize=11)
    ax2.set_title("Latency Distribution vs SLA", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Guardian Drone — Phase 3 Simulation Results\n"
                 "5 Fault Scenarios × 10 Runs per Scenario",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_png}")
    plt.close(fig)


# ─── Programmatic entry point ─────────────────────────────────────────────────

def generate_report(
    rows_or_results,
    out_png: Optional[Path] = None,
    plot: bool = True,
) -> None:
    """
    Generate console + optional plot report.

    rows_or_results can be:
      - list[dict] loaded from CSV (has string keys)
      - list[RunResult] from scenario_runner.run_all_scenarios()
    """
    # Normalise RunResult objects to dicts if needed
    rows: list[dict] = []
    for r in rows_or_results:
        if isinstance(r, dict):
            rows.append(r)
        else:
            from dataclasses import asdict
            d = asdict(r)
            d["passed"] = r.passed
            rows.append(d)

    print_report(rows)
    if plot and out_png:
        generate_plot(rows, out_png)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardian Drone — Report Generator")
    parser.add_argument("--csv",      default=str(ROOT / "docs" / "sim_results.csv"),
                        help="Input CSV from scenario_runner.py")
    parser.add_argument("--out",      default=str(ROOT / "docs" / "sim_report.png"),
                        help="Output plot PNG")
    parser.add_argument("--no-plot",  action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("  Run 'python simulation/scenario_runner.py --dry-run' first to generate it.")
        sys.exit(1)

    rows = load_csv(csv_path)
    print(f"  Loaded {len(rows)} runs from {csv_path}")

    generate_report(
        rows,
        out_png=Path(args.out) if not args.no_plot else None,
        plot=not args.no_plot,
    )
