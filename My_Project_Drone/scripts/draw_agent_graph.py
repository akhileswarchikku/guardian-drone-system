"""
Generate docs/agent_graph.png — Guardian Drone 10-Agent pipeline diagram.

Run:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe scripts/draw_agent_graph.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / "docs" / "agent_graph.png"

# ── colour palette ─────────────────────────────────────────────────────────────
C_LLM   = "#7C3AED"   # purple  — LLM agents
C_RULE  = "#0F766E"   # teal    — rule-based agents
C_DB    = "#B45309"   # amber   — DB / geospatial
C_GATE  = "#B91C1C"   # red     — gate / routing nodes
C_SYS   = "#374151"   # dark    — START / END
EDGE_C  = "#6B7280"
ARROW_C = "#374151"

# ── node layout  (x-centre, y-centre, label, subtitle, colour) ────────────────
# Y: 0 = top, 10 = bottom
NODES = [
    # x,    y,    label,                     subtitle,                 colour
    (4.0,  9.5,  "START",                   "",                       C_SYS),
    (4.0,  8.5,  "1  DangerScore",          "LLM + rule fallback",    C_LLM),
    (4.0,  7.2,  "2  SOSBroadcast",         "LLM: structured alert",  C_LLM),
    (4.0,  6.0,  "3  StationFinder",        "SQLite Haversine query", C_DB),
    (4.0,  4.9,  "4  PathPlanner",          "3-waypoint GPS plan",    C_RULE),
    (4.0,  3.8,  "5  Dispatch",             "DB reserve + API launch",C_RULE),
    (4.0,  2.7,  "6  Tracking",             "Telemetry + GPS drift",  C_RULE),
    (4.0,  1.6,  "7  MalfunctionMonitor",   "Battery + fault checks", C_RULE),
    # branches
    (1.4,  0.4,  "8  Handoff",              "LLM: backup drone",      C_LLM),
    (6.6,  0.4,  "9  SceneIntelligence",    "LLM: threat classify",   C_LLM),
    (4.0, -0.7,  "10  ManagementNotify",    "LLM: supervisor update", C_LLM),
    # terminals — unique keys for lookup; "END" is displayed in the box
    (7.5,  8.0,  "END_score",               "score < 70",             C_GATE),
    (1.4, -1.8,  "END_handoff",             "handoff fail",           C_GATE),
    (4.0, -1.8,  "END_complete",            "complete / aborted",     C_GATE),
]

NODE_W = 2.5
NODE_H = 0.52

def _box(ax, x, y, label, sub, color):
    bx = FancyBboxPatch(
        (x - NODE_W/2, y - NODE_H/2), NODE_W, NODE_H,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
    )
    ax.add_patch(bx)
    ax.text(x, y + 0.06, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(x, y - 0.14, sub, ha="center", va="center",
                fontsize=6.5, color="white", alpha=0.88, zorder=4)


def _arrow(ax, x1, y1, x2, y2, label="", color=ARROW_C, style="->", lw=1.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        connectionstyle="arc3,rad=0.0"),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.08, my, label, fontsize=6.5, color=color,
                va="center", fontstyle="italic", zorder=5)


def _curved_arrow(ax, x1, y1, x2, y2, rad=0.3, label="", color=ARROW_C, lw=1.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2 - 0.5
        my = (y1 + y2) / 2
        ax.text(mx, my, label, fontsize=6.5, color=color, zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-2.5, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor("#F9FAFB")

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.text(4.75, 10.2,
            "Guardian Drone System — LangGraph 10-Agent Pipeline",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#111827")
    ax.text(4.75, 9.95,
            "Phase 2  |  LangGraph 1.1.2  |  OpenRouter (Gemini 2.5 Flash)",
            ha="center", va="center", fontsize=8, color="#6B7280")

    # ── Draw nodes ─────────────────────────────────────────────────────────────
    node_map = {}
    for i, (x, y, key, sub, color) in enumerate(NODES):
        # Display "END" for terminal nodes; use full key for node_map lookup
        display = "END" if key.startswith("END_") else key
        _box(ax, x, y, display, sub, color)
        node_map[key] = (x, y)

    # ── Straight pipeline edges ─────────────────────────────────────────────────
    spine = [
        ("START",               "1  DangerScore",       ""),
        ("1  DangerScore",      "2  SOSBroadcast",       "go = True"),
        ("2  SOSBroadcast",     "3  StationFinder",      ""),
        ("3  StationFinder",    "4  PathPlanner",         ""),
        ("4  PathPlanner",      "5  Dispatch",            ""),
        ("5  Dispatch",         "6  Tracking",            ""),
        ("6  Tracking",         "7  MalfunctionMonitor",  ""),
    ]
    for src, dst, lbl in spine:
        x1, y1 = node_map[src]
        x2, y2 = node_map[dst]
        _arrow(ax, x1, y1 - NODE_H/2, x2, y2 + NODE_H/2, label=lbl)

    # ── Conditional: DangerScore → END (go=False) ──────────────────────────────
    x1, y1 = node_map["1  DangerScore"]
    x2, y2 = node_map["END_score"]
    _arrow(ax, x1 + NODE_W/2, y1, x2, y2 + NODE_H/2,
           label="go=False", color=C_GATE)

    # ── MalfunctionMonitor → Handoff (fault) ───────────────────────────────────
    x1, y1 = node_map["7  MalfunctionMonitor"]
    x2, y2 = node_map["8  Handoff"]
    _arrow(ax, x1 - NODE_W/2 + 0.2, y1 - NODE_H/2,
           x2 + NODE_W/2 - 0.2, y2 + NODE_H/2,
           label="fault", color="#DC2626")

    # ── MalfunctionMonitor → SceneIntelligence (ok / arrived) ──────────────────
    x1, y1 = node_map["7  MalfunctionMonitor"]
    x2, y2 = node_map["9  SceneIntelligence"]
    _arrow(ax, x1 + NODE_W/2 - 0.2, y1 - NODE_H/2,
           x2 - NODE_W/2 + 0.2, y2 + NODE_H/2,
           label="ok / arrived", color=C_RULE)

    # ── SceneIntelligence → ManagementNotify ───────────────────────────────────
    x1, y1 = node_map["9  SceneIntelligence"]
    x2, y2 = node_map["10  ManagementNotify"]
    _arrow(ax, x1 - NODE_W/2 + 0.4, y1 - NODE_H/2,
           x2 + NODE_W/2 - 0.4, y2 + NODE_H/2)

    # ── Handoff → tracking (ok) via left-side loop ─────────────────────────────
    x1, y1 = node_map["8  Handoff"]
    x2, y2 = node_map["6  Tracking"]
    _curved_arrow(ax, x1, y1 + NODE_H/2, x2 - NODE_W/2, y2,
                  rad=-0.3, label="new drone", color="#7C3AED", lw=1.5)

    # ── Handoff → END (fail) ───────────────────────────────────────────────────
    x1, y1 = node_map["8  Handoff"]
    x2, y2 = node_map["END_handoff"]
    _arrow(ax, x1, y1 - NODE_H/2, x2, y2 + NODE_H/2,
           label="fail", color=C_GATE)

    # ── ManagementNotify → tracking (active, loop) — right-side feedback ────────
    x1, y1 = node_map["10  ManagementNotify"]
    x2, y2 = node_map["6  Tracking"]
    _curved_arrow(ax, x1 + NODE_W/2, y1, x2 + NODE_W/2 + 0.3, y2,
                  rad=-0.45, label="still flying\n(loop)", color="#0F766E", lw=1.5)

    # ── ManagementNotify → END (complete / aborted) ────────────────────────────
    x1, y1 = node_map["10  ManagementNotify"]
    x2, y2 = node_map["END_complete"]
    _arrow(ax, x1, y1 - NODE_H/2, x2, y2 + NODE_H/2,
           label="complete", color=C_GATE)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=C_LLM,  label="LLM agent (OpenRouter)"),
        mpatches.Patch(facecolor=C_RULE, label="Rule-based agent"),
        mpatches.Patch(facecolor=C_DB,   label="Database / geospatial"),
        mpatches.Patch(facecolor=C_GATE, label="Gate / END node"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
              framealpha=0.9, edgecolor="#D1D5DB")

    plt.tight_layout()
    plt.savefig(str(OUT), dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
