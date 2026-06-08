"""
Phase 1.8 - Model Export and Optimization

Steps:
  1. Load DangerLSTM state dict -> reconstruct model
  2. Export to ONNX (opset 17, dynamic batch)
  3. Validate: ONNX vs PyTorch outputs on 100 random inputs (tolerance 0.001)
  4. INT8 dynamic quantization -> size comparison
  5. Benchmark: 1000 CPU forward passes, target mean < 50ms

Run:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe -m biometric_ml.export_model
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "biometric_ml" / "models"

SEQ_LEN    = 8
N_FEATURES = 16


# ── Model definition (must match lstm_classifier.DangerLSTM exactly) ─────────

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


# ── Step 1: Load model ────────────────────────────────────────────────────────

def load_model() -> DangerLSTM:
    state_dict = torch.load(
        MODELS_DIR / "best_danger_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = DangerLSTM()
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Step 2: ONNX export ───────────────────────────────────────────────────────

def export_onnx(model: DangerLSTM) -> Path:
    onnx_path = MODELS_DIR / "danger_lstm.onnx"
    dummy = torch.randn(1, SEQ_LEN, N_FEATURES)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names  = ["feature_sequence"],
        output_names = ["logits"],
        dynamic_axes = {
            "feature_sequence": {0: "batch_size"},
            "logits":           {0: "batch_size"},
        },
        opset_version = 17,
    )

    # Verify the ONNX graph is valid
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX model valid — saved to {onnx_path.name}")
    print(f"  ONNX file size  : {onnx_path.stat().st_size / 1024:.1f} KB")
    return onnx_path


# ── Step 3: Validate outputs ──────────────────────────────────────────────────

def validate_onnx(model: DangerLSTM, onnx_path: Path, n_samples: int = 100) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng  = np.random.default_rng(42)
    diffs: list[float] = []

    with torch.no_grad():
        for _ in range(n_samples):
            x_np  = rng.standard_normal((1, SEQ_LEN, N_FEATURES)).astype(np.float32)
            pt_out = model(torch.from_numpy(x_np)).numpy()
            ort_out = sess.run(None, {input_name: x_np})[0]
            diffs.append(float(np.max(np.abs(pt_out - ort_out))))

    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    status = "PASS" if max_diff < 0.001 else "FAIL"
    print(f"  Validation [{status}]  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
          f"  (tolerance=0.001, n={n_samples})")
    return max_diff


# ── Step 4: INT8 dynamic quantization ────────────────────────────────────────

def quantize_model(model: DangerLSTM) -> tuple[nn.Module, Path]:
    quant_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.LSTM, nn.Linear},
        dtype=torch.qint8,
    )
    quant_path = MODELS_DIR / "danger_lstm_int8.pt"
    torch.save(quant_model.state_dict(), quant_path)

    orig_path   = MODELS_DIR / "best_danger_model.pt"
    orig_kb     = orig_path.stat().st_size / 1024
    quant_kb    = quant_path.stat().st_size / 1024
    reduction   = orig_kb / quant_kb if quant_kb > 0 else 0
    print(f"  Original  : {orig_kb:.1f} KB")
    print(f"  INT8      : {quant_kb:.1f} KB")
    print(f"  Reduction : {reduction:.2f}x")
    return quant_model, quant_path


# ── Step 5: Benchmark ─────────────────────────────────────────────────────────

def benchmark(model: nn.Module, label: str, n_runs: int = 1000) -> float:
    model.eval()
    x = torch.randn(1, SEQ_LEN, N_FEATURES)
    # warmup
    with torch.no_grad():
        for _ in range(20):
            model(x)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    std_ms  = float(np.std(times))
    status  = "PASS" if mean_ms < 50.0 else "FAIL"
    print(f"  {label:<10} [{status}]  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms  "
          f"(n={n_runs}, target<50ms)")
    return mean_ms


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 64

    print(f"\n{sep}")
    print("  Phase 1.8 -- LSTM Model Export and Optimization")
    print(sep)

    print("\n  [1/4] Loading DangerLSTM from best_danger_model.pt ...")
    model = load_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}  ({total_params*4/1024:.1f} KB weights)")

    print("\n  [2/4] Exporting to ONNX ...")
    onnx_path = export_onnx(model)

    print("\n  [3/4] Validating ONNX vs PyTorch on 100 samples ...")
    max_diff = validate_onnx(model, onnx_path)
    if max_diff >= 0.001:
        print("  WARNING: tolerance exceeded — check ONNX export settings")

    print("\n  [4/4] INT8 dynamic quantization ...")
    quant_model, quant_path = quantize_model(model)

    print("\n  Benchmarking (1000 CPU forward passes) ...")
    bench_fp32 = benchmark(model,       "FP32")
    bench_int8 = benchmark(quant_model, "INT8")
    speedup = bench_fp32 / bench_int8 if bench_int8 > 0 else 0
    print(f"  INT8 speedup vs FP32: {speedup:.2f}x")

    print(f"\n{sep}")
    print("  Phase 1.8 COMPLETE")
    print(f"  ONNX model : {onnx_path}")
    print(f"  INT8 model : {quant_path}")
    print(sep + "\n")


if __name__ == "__main__":
    main()
