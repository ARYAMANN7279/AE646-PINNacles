"""PDEBench 2D Darcy label check at native 128x128: are the labels the converged steady solution of
-div(k grad u)=1, u=0 on the boundary? Solve each test map at 128/256/512 (nearest upsampling of the native
coefficient), cell-average back to 128, compare with the label. Also tries boundary offsets and face rules."""
import glob, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path
import numpy as np
ROOT = Path("/kaggle/temp/AE646"); OUT = Path("/kaggle/working/s10"); (ROOT/"src").mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
t = glob.glob("/kaggle/input/**/ae646_s10_code.tar.gz", recursive=True)
if t:
    tarfile.open(t[0]).extractall(ROOT/"src")
else:
    d = Path(glob.glob("/kaggle/input/**/solvers.py", recursive=True)[0]).parent
    [shutil.copy(f, ROOT/"src"/f.name) for f in d.glob("*.py")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h5py", "tqdm", "requests", "pyamg"], check=False)
for s in ("download_data.py", "preprocess.py", "data_cache.py"):
    r = subprocess.run([sys.executable, f"src/{s}"], cwd=ROOT, capture_output=True, text=True); print(s, r.returncode, r.stdout[-300:], r.stderr[-300:], flush=True); assert r.returncode == 0
sys.path.insert(0, str(ROOT/"src")); os.chdir(ROOT); import solvers as S
K = np.load("data/cache/kappa128.npy", mmap_mode="r"); U = np.load("data/cache/u128.npy", mmap_mode="r"); sp = np.load("data/cache/splits.npz")
idx = np.asarray(sp["test"])[:40]; print("K", K.shape, "U", U.shape, "test idx", idx[:5], flush=True)
up = lambda k, r: np.kron(k, np.ones((r, r))); down = lambda u, r: u.reshape(u.shape[0]//r, r, u.shape[1]//r, r).mean((1, 3))
rel = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
res = {"n": len(idx)}
for face in ("harmonic", "arithmetic"):
    sol = {r: [down(S.solve(up(np.asarray(K[i], np.float64), r), method="direct", face=face, bc=0.5), r) for i in idx] for r in (1, 2, 4)}
    G = [np.asarray(U[i], np.float64) for i in idx]
    res[face] = {"self_128_256": np.mean([rel(sol[1][j], sol[2][j]) for j in range(len(idx))]),
                 "self_256_512": np.mean([rel(sol[2][j], sol[4][j]) for j in range(len(idx))]),
                 "vs_label": {str(128*r): np.mean([rel(sol[r][j], G[j]) for j in range(len(idx))]) for r in (1, 2, 4)},
                 "best_scale": np.mean([(G[j]*sol[4][j]).sum()/(sol[4][j]**2).sum() for j in range(len(idx))])}
    s = sol[4]; res[face]["vs_label_after_scale"] = np.mean([rel(s[j]*(G[j]*s[j]).sum()/(s[j]**2).sum(), G[j]) for j in range(len(idx))])
    res[face]["label_boundary_rows_mean_abs"] = float(np.mean([np.abs(np.r_[g[0], g[-1], g[:, 0], g[:, -1]]).mean()/np.abs(g).max() for g in G]))
    print(face, json.dumps(res[face]), flush=True)
    np.savez_compressed(OUT/f"conv_{face}.npz", idx=idx, sol512=np.stack(sol[4]).astype(np.float32), label=np.stack(G).astype(np.float32), kappa=np.stack([K[i] for i in idx]))
# boundary-offset scan at 128 (harmonic): which bc best fits the labels
res["bc_scan"] = {str(bc): np.mean([rel(S.solve(np.asarray(K[i], np.float64), method="direct", face="harmonic", bc=bc), np.asarray(U[i], np.float64)) for i in idx[:20]]) for bc in (0.5, 0.75, 1.0, 1.05, 1.25)}
print("bc_scan", res["bc_scan"]); json.dump(res, open(OUT/"label_check.json", "w"), indent=1); print("ALL-DONE")
