"""
FNO Layer-wise Component Profiling
-----------------------------------
Times each component of one FNO forward pass individually.

GPU operations are async, so naive perf_counter() inside forward() gives
dispatch time, not execution time. Fix: synchronize the device before every
timer call so we measure real wall-clock execution.

This slows down profiling (sync is expensive) but gives correct numbers.
Run with: python src/benchmark_components.py
"""

import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from models import FNO2d, SpectralConv2d


# ── device sync helper ────────────────────────────────────────────────────────

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def now(device):
    sync(device)
    return time.perf_counter()


# ── profiled submodules ───────────────────────────────────────────────────────

class ProfiledSpectralConv2d(SpectralConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_fft = self.t_mul = self.t_ifft = 0.0

    def forward(self, x):
        B, H, W, C = x.shape
        dev = x.device
        x = x.permute(0, 3, 1, 2)

        t0 = now(dev)
        x_ft = torch.fft.rfft2(x)

        t1 = now(dev)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=dev)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        t2 = now(dev)
        x = torch.fft.irfft2(out_ft, s=(H, W))

        t3 = now(dev)
        x = x.permute(0, 2, 3, 1)

        self.t_fft += t1 - t0
        self.t_mul += t2 - t1
        self.t_ifft += t3 - t2
        return x


class ProfiledFNO2d(FNO2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace spectral convs with profiled versions (same arch, fresh weights —
        # weight values don't affect timing)
        self.spectral_convs = torch.nn.ModuleList([
            ProfiledSpectralConv2d(self.width, self.width, self.modes, self.modes)
            for _ in range(self.n_layers)
        ])
        self.t_lift = self.t_local = self.t_project = 0.0
        self.calls = 0

    def reset_timers(self):
        self.t_lift = self.t_local = self.t_project = 0.0
        self.calls = 0
        for sc in self.spectral_convs:
            sc.t_fft = sc.t_mul = sc.t_ifft = 0.0

    def forward(self, x):
        dev = x.device

        t0 = now(dev)
        x = self.fc0(x)
        self.t_lift += now(dev) - t0

        for i in range(self.n_layers):
            # spectral path — timers inside ProfiledSpectralConv2d
            x1 = self.spectral_convs[i](x)

            t_loc = now(dev)
            x2 = self.ws[i](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x1 + x2
            x = self.norms[i](x)
            x = F.gelu(x)
            self.t_local += now(dev) - t_loc

        t_proj = now(dev)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        self.t_project += now(dev) - t_proj

        self.calls += 1
        return x


# ── main ──────────────────────────────────────────────────────────────────────

def run_profiling(warmup=50, iterations=500):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    model = ProfiledFNO2d(
        input_channels=3, output_channels=1,
        width=64, modes=12, n_layers=4
    ).to(device)
    model.eval()

    x = torch.randn(1, 64, 64, 3, device=device)

    # Warmup
    print(f"Warming up ({warmup} iters)...")
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    sync(device)

    model.reset_timers()

    # Profile
    print(f"Profiling ({iterations} iters, sync before each timer)...")
    wall_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(x)
    sync(device)
    wall_total = (time.perf_counter() - wall_start) / iterations * 1000  # ms per call

    n = model.calls

    # Aggregate spectral sub-times (summed over 4 layers)
    t_fft  = sum(sc.t_fft  for sc in model.spectral_convs) / n * 1000
    t_mul  = sum(sc.t_mul  for sc in model.spectral_convs) / n * 1000
    t_ifft = sum(sc.t_ifft for sc in model.spectral_convs) / n * 1000

    t_lift    = model.t_lift    / n * 1000
    t_local   = model.t_local   / n * 1000
    t_project = model.t_project / n * 1000

    component_total = t_lift + t_fft + t_mul + t_ifft + t_local + t_project

    def pct(t):
        return t / component_total * 100

    print()
    print("=" * 54)
    print("  FNO LAYER-WISE PROFILING RESULTS  (per sample, ms)")
    print("=" * 54)
    print(f"  1. Lifting Layer (fc0):       {t_lift:6.4f} ms  ({pct(t_lift):4.1f}%)")
    print(f"  2. FFT  (×{model.n_layers} layers):          {t_fft:6.4f} ms  ({pct(t_fft):4.1f}%)")
    print(f"  3. Complex Mul  (×{model.n_layers} layers):  {t_mul:6.4f} ms  ({pct(t_mul):4.1f}%)")
    print(f"  4. IFFT (×{model.n_layers} layers):          {t_ifft:6.4f} ms  ({pct(t_ifft):4.1f}%)")
    print(f"  5. Local Conv + Norm + GELU:  {t_local:6.4f} ms  ({pct(t_local):4.1f}%)")
    print(f"  6. Projection (fc1+fc2):      {t_project:6.4f} ms  ({pct(t_project):4.1f}%)")
    print("-" * 54)
    print(f"  Sum of components:            {component_total:6.4f} ms")
    print(f"  Wall-clock (end-to-end):      {wall_total:6.4f} ms")
    print("=" * 54)
    print()
    print("Note: sync-before-every-timer adds overhead; wall-clock is")
    print("the real end-to-end latency. Percentages from component sum.")

    results = {
        "device": str(device),
        "iterations": iterations,
        "per_sample_ms": {
            "lift":       round(t_lift,    6),
            "fft":        round(t_fft,     6),
            "complex_mul": round(t_mul,    6),
            "ifft":       round(t_ifft,    6),
            "local_conv": round(t_local,   6),
            "projection": round(t_project, 6),
            "component_total": round(component_total, 6),
            "wall_clock": round(wall_total, 6),
        },
        "pct": {
            "lift":       round(pct(t_lift),    2),
            "fft":        round(pct(t_fft),     2),
            "complex_mul": round(pct(t_mul),    2),
            "ifft":       round(pct(t_ifft),    2),
            "local_conv": round(pct(t_local),   2),
            "projection": round(pct(t_project), 2),
        }
    }

    out_path = os.path.join(os.path.dirname(__file__),
                            "../results/benchmark_components.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {os.path.normpath(out_path)}")
    return results


if __name__ == "__main__":
    run_profiling()
