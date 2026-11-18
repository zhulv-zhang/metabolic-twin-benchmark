# -*- coding: utf-8 -*-
"""延迟族 wrapper v1.2（2026-10-05 S1，对锚 c300efff 实文重焊）
用法: python delay_wrapper_v1.2_1005.py --src untitled.py --out_root <夹>
τ 恒 10min：lag(dt)=max(1, round(20*0.5/dt))；delay 臂 DELAY_DIM=0 分量改吃滞后值。"""
import os, sys, json, time, hashlib, argparse, subprocess
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--out_root", required=True)
p.add_argument("--lag_steps", type=int, default=20, help="dt0.5 基准 lag（τ=lag×0.5min）")
p.add_argument("--delay_dim", type=int, default=0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--noise", type=float, default=0.1)
p.add_argument("--reps", type=int, default=1)
p.add_argument("--peak_lo", type=float, default=0.3)
p.add_argument("--peak_hi", type=float, default=3.0)
p.add_argument("--conv_tol", type=float, default=0.05)
args = p.parse_args()

src = open(args.src, encoding="utf-8").read()
md5f = lambda pth: hashlib.md5(open(pth, "rb").read()).hexdigest()[:8]
NL = chr(10)
for a in ["def simulate(cond):", "    traj = [x.copy()]", "dt, tmax = 0.5, 240.0"]:
    assert src.count(a) == 1, f"锚{a!r}计数{src.count(a)}≠1，停手"

block1 = (
 "import os as _dos" + NL +
 "DELAY_DIM = int(_dos.environ.get('DELAY_DIM', '" + str(args.delay_dim) + "'))" + NL +
 "LAG_STEPS = int(_dos.environ.get('DELAY_LAG', '0'))" + NL +
 "_HIST = None" + NL +
 "_orig_rhs = rhs" + NL +
 "def rhs(x, t, cond):" + NL +
 "    d = _orig_rhs(x, t, cond).copy()" + NL +
 "    if LAG_STEPS > 0 and _HIST is not None:" + NL +
 "        _step = int(round(t / dt))" + NL +
 "        _hi = max(_step - LAG_STEPS, 0)" + NL +
 "        if _hi < len(_HIST):" + NL +
 "            d[DELAY_DIM] = float(_HIST[_hi][DELAY_DIM]) - x[DELAY_DIM]" + NL +
 "    return d" + NL + NL
)
patched = src.replace("def simulate(cond):", block1 + "def simulate(cond):", 1)
patched = patched.replace("    traj = [x.copy()]",
    "    traj = [x.copy()]" + NL + "    global _HIST" + NL + "    _HIST = traj", 1)
patched = patched.replace("dt, tmax = 0.5, 240.0",
    'dt, tmax = float(os.environ.get("DELAY_DT", "0.5")), 240.0', 1)
compile(patched, "<patched>", "exec")

PATCHED = os.path.join(args.out_root, "_patched_delay_src.py")
os.makedirs(args.out_root, exist_ok=True)
open(PATCHED, "w", encoding="utf-8").write(patched)

def run_arm(script, out_dir, dt):
    env = dict(os.environ)
    env["DELAY_DT"] = str(dt)
    env["DELAY_LAG"] = str(0 if script == args.src else max(1, round(args.lag_steps * 0.5 / dt)))
    od = os.path.join(args.out_root, out_dir)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "run_console.log"), "w", encoding="utf-8") as logf:
        subprocess.Popen([sys.executable, script, "--out_dir", od, "--seed", str(args.seed),
                          "--noise", str(args.noise), "--reps", str(args.reps)],
                         env=env, stdout=logf, stderr=subprocess.STDOUT).wait()
    diverged = "Traceback" in open(os.path.join(od, "run_console.log"),
                                   encoding="utf-8", errors="ignore").read()
    fp = {fn: (md5f(os.path.join(od, fn)) if os.path.exists(os.path.join(od, fn)) else None)
          for fn in ["train_tensors.npz", "ground_truth.npz", "zstats.npz"]}
    return diverged, fp

report = {"wrapper": "delay_wrapper_v1.2", "lag_rule": "max(1,round(20*0.5/dt)), tau=10min",
          "src_md5": md5f(args.src), "seed": args.seed, "arms": {}, "gates": {}}
arms = [("base_dt0.5", args.src, 0.5), ("delay_dt0.5", PATCHED, 0.5),
        ("delay_dt0.25", PATCHED, 0.25), ("delay_dt0.125", PATCHED, 0.125)]
for name, script, dt in arms:
    t0 = time.time()
    div, fp = run_arm(script, name, dt)
    report["arms"][name] = {"dt": dt, "diverged": bool(div), "fingerprints": fp,
                            "sec": round(time.time() - t0, 1)}
    print(f"[{name}] diverged={div} fp={fp}", flush=True)

g = {}
try:
    pk = {}
    Xb = np.load(os.path.join(args.out_root, "base_dt0.5", "train_tensors.npz"))
    Xh = np.load(os.path.join(args.out_root, "delay_dt0.5", "train_tensors.npz"))
    for k in ["XG", "XP", "XS", "XM"]:
        if k in Xb and k in Xh:
            b, h = np.asarray(Xb[k], float), np.asarray(Xh[k], float)
            pk[k] = float(np.nanmax(np.abs(h)) / max(np.nanmax(np.abs(b)), 1e-12))
    g["peak_ratio"] = pk
    g["peak_ok"] = all(args.peak_lo <= v <= args.peak_hi for v in pk.values())
    g["no_nan_ok"] = all(not report["arms"][n]["diverged"] for n, _, _ in arms)
    rels = {}
    X1 = np.load(os.path.join(args.out_root, "delay_dt0.5", "train_tensors.npz"))
    X2 = np.load(os.path.join(args.out_root, "delay_dt0.25", "train_tensors.npz"))
    for k in ["XG", "XP", "XS", "XM"]:
        if k in X1 and k in X2:
            a1, a2 = np.asarray(X1[k], float), np.asarray(X2[k], float)
            rels[k] = float(np.nanmax(np.abs(a2 - a1)) / max(np.nanmax(np.abs(a1)), 1e-12))
    g["dt_convergence_rel_linf"] = rels
    g["dt_conv_ok"] = all(v < args.conv_tol for v in rels.values())
except Exception as e:
    g["peak_ok"] = g["no_nan_ok"] = g["dt_conv_ok"] = False
    g["error"] = str(e)
g["PASS"] = bool(g.get("peak_ok") and g.get("no_nan_ok") and g.get("dt_conv_ok"))
report["gates"] = g
rp = os.path.join(args.out_root, "wrapper_report_1005.json")
json.dump(report, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("PASS =", g["PASS"], "| report:", rp)