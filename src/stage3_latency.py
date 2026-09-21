"""
Inference-latency benchmark (CUDA events on GPU, perf_counter on CPU) for FNO / MLP surrogates.
Latency does not depend on weight values, so randomly initialised models of the given size are timed.
Reports ms per batch, ms per sample and samples/s for several batch sizes, precisions (fp32 / TF32 / bf16 autocast)
and CPU thread counts. Writes results/stage3/latency.json.   Run on an otherwise idle GPU.
"""
import json
import statistics
import time
from pathlib import Path

import torch

from models import get_model

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "FNO-final (m12,w64,L8)@64": ("fno", dict(width=64, modes=12, n_layers=8), 64),
    "MLP-final (3x4096)@64": ("mlp", dict(hidden_dims=[4096] * 3, height=64, width=64), 64),
    "FNO-base (m12,w64,L4)@64": ("fno", dict(width=64, modes=12, n_layers=4), 64),
    "FNO-base (m12,w64,L4)@128": ("fno", dict(width=64, modes=12, n_layers=4), 128),
    "FNO-base (m12,w64,L4)@32": ("fno", dict(width=64, modes=12, n_layers=4), 32),
    "FNO-small (m8,w32,L4)@64": ("fno", dict(width=32, modes=8, n_layers=4), 64),
    "FNO-large (m24,w96,L4)@64": ("fno", dict(width=96, modes=24, n_layers=4), 64),
    "MLP (3x2048)@64": ("mlp", dict(hidden_dims=[2048] * 3, height=64, width=64), 64),
}


def timeit(fn, dev, n_warm=10, n_rep=30):
    for _ in range(n_warm):
        fn()
    ts = []
    if dev.type == "cuda":
        torch.cuda.synchronize()
        for _ in range(n_rep):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    else:
        for _ in range(n_rep):
            t0 = time.perf_counter(); fn(); ts.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(ts)


def build(kind, kw, res):
    m = get_model(kind, input_channels=3, output_channels=1, **({**kw, "height": res, "width": res} if kind == "mlp" else kw))
    return m.eval()


def main():
    out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__, "rows": []}
    dev = torch.device("cuda")
    for name, (kind, kw, res) in CONFIGS.items():
        m = build(kind, kw, res).to(dev)
        n_params = sum(p.numel() for p in m.parameters())
        for prec in ("fp32", "tf32", "bf16"):
            torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"; torch.backends.cudnn.allow_tf32 = prec == "tf32"
            for bs in (1, 8, 32, 128):
                x = torch.randn(bs, res, res, 3, device=dev)
                def fn():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
                        return m(x)
                try:
                    ms = timeit(fn, dev)
                except Exception as e:  # e.g. complex ops unsupported in bf16
                    out["rows"].append(dict(model=name, device="gpu", precision=prec, batch=bs, error=str(e)[:80])); continue
                out["rows"].append(dict(model=name, params=n_params, device="gpu", precision=prec, batch=bs, ms_batch=ms, ms_per_sample=ms / bs, samples_per_s=1e3 * bs / ms))
                print(name, prec, bs, f"{ms:.3f} ms/batch  {ms/bs:.4f} ms/sample", flush=True)
        del m
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for threads in (1, 8):  # CPU numbers come from a shared 16-core machine: indicative only
        torch.set_num_threads(threads)
        for name, (kind, kw, res) in CONFIGS.items():
            m = build(kind, kw, res)
            for bs in (1, 32):
                x = torch.randn(bs, res, res, 3)
                def fn():
                    with torch.no_grad():
                        return m(x)
                ms = timeit(fn, torch.device("cpu"), n_warm=5, n_rep=25)
                out["rows"].append(dict(model=name, params=sum(p.numel() for p in m.parameters()), device=f"cpu{threads}", precision="fp32", batch=bs, ms_batch=ms, ms_per_sample=ms / bs, samples_per_s=1e3 * bs / ms))
                print(name, f"cpu{threads}", bs, f"{ms:.2f} ms/batch", flush=True)
    (ROOT / "results" / "stage3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "stage3" / "latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
