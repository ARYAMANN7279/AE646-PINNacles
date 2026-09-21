"""
Run a queue of stage3_train.py jobs across GPUs.  Usage:
    python src/run_jobs.py jobs.txt --gpus 1 2 3 --per-gpu 3
jobs.txt: one line of stage3_train.py arguments per job (must contain --name). Jobs whose
results/stage3/runs/<name>/metrics.json already exists are skipped, so the queue is restartable.
"""
import argparse
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--per-gpu", type=int, default=2)
    a = ap.parse_args()
    q = queue.Queue()
    for line in Path(a.jobs).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.search(r"--name\s+(\S+)", line).group(1)
        if (ROOT / "results/stage3/runs" / name / "metrics.json").exists():
            continue
        q.put((name, line))
    total = q.qsize()
    done = [0]
    lock = threading.Lock()
    (ROOT / "results/stage3/logs").mkdir(parents=True, exist_ok=True)

    def worker(gpu):
        while True:
            try:
                name, line = q.get_nowait()
            except queue.Empty:
                return
            log = open(ROOT / "results/stage3/logs" / f"{name}.log", "w")
            env = {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"}
            import os
            env = {**os.environ, **env}
            subprocess.run([sys.executable, str(ROOT / "src/stage3_train.py")] + line.split(), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{total}] {name} (gpu {gpu})", flush=True)

    ts = [threading.Thread(target=worker, args=(g,)) for g in a.gpus for _ in range(a.per_gpu)]
    [t.start() for t in ts]
    [t.join() for t in ts]


if __name__ == "__main__":
    main()
