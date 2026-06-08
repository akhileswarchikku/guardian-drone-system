# Guardian Drone System
**Autonomous Women Safety Response Platform**

> Smartwatch detects physiological distress → AI agents dispatch nearest police drone → drone tracks moving victim → YOLOv8 maps building entries/exits → live police dashboard → fault-tolerant multi-drone handoff.

**Author:** Akhileswar Pathinavalasa (Chikku)  
**Role:** AI/ML Engineer | MS Business Analytics, Wichita State University  
**Stack:** LangGraph · LangChain · PyTorch · YOLOv8 · PX4 · MAVLink · FastAPI · React  
**Timeline:** 8–12 months | **Hardware Budget:** ~$1,030 | **Phases 1–4:** $0 (laptop only)

---

## System Architecture

```
Smartwatch (Wear OS)
    └── ONNX LSTM danger model (runs on-device, every 5s)
            └── Danger score > 70 sustained 15s → SOS payload (encrypted)
                    └── LangGraph Agent Pipeline (10 agents)
                            ├── Station Finder (PostGIS nearest drone station)
                            ├── Path Planner (A* + battery-aware + no-fly zones)
                            ├── Dispatch Agent (MAVLink launch command)
                            ├── Tracking Agent (GPS polling, waypoint updates)
                            ├── Malfunction Monitor (telemetry every 500ms)
                            ├── Handoff Agent (Drone 2 dispatch within 30s)
                            └── Scene Intelligence Agent (YOLOv8 on Jetson)
                                    └── Management Dashboard (React + Mapbox + WebSocket)
```

---

## 6 Phases

| Phase | What | Duration | Cost |
|---|---|---|---|
| 1 | Biometric ML — LSTM danger model on WESAD dataset | Month 1–2 | $0 |
| 2 | LangGraph AI agents + mock drone API | Month 3–4 | $0 |
| 3 | AirSim + PX4 SITL simulation | Month 5–6 | $0 |
| 4 | YOLOv8 + ByteTrack vision on Jetson Orin Nano | Month 7–8 | $0 |
| 5 | Physical drone assembly + 6 outdoor flight tests | Month 9–10 | ~$1,030 |
| 6 | Dashboard + Wear OS app + government proposal | Month 11–12 | ~$1,500–3K |

---

## Repository Structure

```
guardian-drone-system/
├── biometric_ml/       # Phase 1: LSTM, XGBoost, preprocessing, ONNX export
├── agents/             # Phase 2: LangGraph 10-agent pipeline, FastAPI mock drone
├── simulation/         # Phase 3: AirSim + DroneKit adapter, malfunction tests
├── vision/             # Phase 4: YOLOv8, scene classifier, ByteTrack, TensorRT
├── hardware/           # Phase 5: Pixhawk calibration notes, bench test logs
├── dashboard/          # Phase 6: React + TypeScript + Mapbox + FastAPI WebSocket
├── docs/               # Benchmarks, agent graph, flight test videos, daily log
├── tests/              # All unit + integration tests
└── data/               # NOT committed — download instructions in docs/
```

---

## Setup (Local Development)

```bash
# Activate the conda LLM environment
conda activate LLM

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
# Edit .env with your OpenRouter API key

# Download WESAD dataset (Phase 1)
# Register at: https://archive.ics.uci.edu/dataset/465/wesad
# Extract 15 subject folders to: data/raw/WESAD/
```

---

## Phase 1 Exit Criteria (before any code from Phase 2 runs)
- [ ] LSTM F1 > 0.83 on distress class (LOSO-CV mean, all 15 subjects)
- [ ] False positive rate < 15% (contextual gate blocks exercise scenarios)
- [ ] ONNX export, inference < 50ms on CPU
- [ ] All preprocessing unit tests passing
- [ ] Model card written and committed

---

## Hardware Bill of Materials (Phase 5)

| Component | Model | Cost |
|---|---|---|
| Drone frame | Holybro X500 V2 | ~$250 |
| Flight controller | Pixhawk 6C (PX4) | ~$180 |
| AI compute | NVIDIA Jetson Orin Nano | ~$250 |
| Camera | Intel RealSense D435i | ~$200 |
| LTE module | Sixfab LTE HAT + SIM | ~$80 |
| Battery (×2) + charger | 4S LiPo 5000mAh | ~$90 |
| RC override + misc | — | ~$60 |
| **Total** | | **~$1,030** |

---

*Private repository — Akhileswar Pathinavalasa (Chikku) 2026*
