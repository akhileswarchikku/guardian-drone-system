#!/bin/bash
# Phase 3.1 — PX4 SITL Setup Script for WSL2 Ubuntu 22.04
# Run this inside WSL2: bash simulation/setup_px4.sh
# Takes ~20-30 minutes on first run (compiles PX4 firmware)

set -e
echo "======================================================"
echo " Guardian Drone — PX4 SITL Setup (WSL2 Ubuntu 22.04)"
echo "======================================================"

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    git wget curl build-essential cmake ninja-build ccache \
    python3-pip python3-dev python3-future python3-lxml \
    python3-jinja2 python3-numpy python3-matplotlib \
    libssl-dev libgstreamer1.0-dev gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libc6-i386 lib32stdc++6 lib32gcc-s1 \
    ant protobuf-compiler libprotobuf-dev \
    libgstreamer-plugins-base1.0-dev \
    2>/dev/null

pip3 install -q kconfiglib jinja2 jsonschema future toml pyyaml pyros-genmsg packaging

echo "[1/6] Done."

# ── 2. Clone PX4 ─────────────────────────────────────────────────────────────
PX4_DIR="$HOME/PX4-Autopilot"

if [ -d "$PX4_DIR" ]; then
    echo "[2/6] PX4-Autopilot already cloned at $PX4_DIR — skipping."
else
    echo "[2/6] Cloning PX4-Autopilot (this may take a few minutes)..."
    git clone https://github.com/PX4/PX4-Autopilot.git --recursive "$PX4_DIR" --depth 1
    echo "[2/6] Done."
fi

cd "$PX4_DIR"

# ── 3. Run PX4 ubuntu.sh setup script ────────────────────────────────────────
echo "[3/6] Running PX4 ubuntu.sh dependency installer..."
bash ./Tools/setup/ubuntu.sh --no-nuttx 2>&1 | tail -10
echo "[3/6] Done."

# ── 4. Build PX4 SITL for AirSim (gazebo_classic target) ─────────────────────
echo "[4/6] Building PX4 SITL (none_iris target for AirSim) — ~15-25 min..."
DONT_RUN=1 make px4_sitl_default none_iris 2>&1 | tail -20
echo "[4/6] Build complete."

# ── 5. Install pymavlink + dronekit in WSL2 ───────────────────────────────────
echo "[5/6] Installing pymavlink and dronekit in WSL2 Python..."
pip3 install -q pymavlink dronekit
echo "[5/6] Done."

# ── 6. Verify ─────────────────────────────────────────────────────────────────
echo "[6/6] Verifying installation..."
python3 -c "import pymavlink; import dronekit; print('pymavlink + dronekit OK')"
echo ""
echo "======================================================"
echo " PX4 SITL setup complete!"
echo ""
echo " To start PX4 SITL for AirSim, run:"
echo "   cd ~/PX4-Autopilot"
echo "   make px4_sitl_default none_iris"
echo ""
echo " PX4 will listen on TCP 4560 for AirSim."
echo " Start AirSim (Blocks.exe) BEFORE running PX4."
echo "======================================================"
