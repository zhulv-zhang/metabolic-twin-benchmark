# -*- coding: utf-8 -*-
"""
hill_wrapper_v1.0.py
================================
Hill 先行件·wrapper 路（Advisor 9-30 终裁：案甲＋wrapper，四护栏两焊点）

【验尸句·改动清单】对造数器本体的全部改动仅此三处文本级补丁（本体文件一字节不动，
补丁作用于运行期生成的副本 半合成仿真_Hill版_wrapper生成.py）：
  补丁A: 第151行 np.tanh(G/P/S/M) → hill(x) 奇对称 Hill 变换
         hill(z) = z*|z|^(H-1) / (K^H + |z|^H)   （z=0 处连续，无 0/0 奇点）
  补丁B: 第145行 dt 硬编码 0.5 → 环境变量 HILL_DT 可控（默认仍 0.5，供 dt 减半收敛验证）
  补丁C: simulate() 返回前加发散硬闸：轨迹出现 NaN/Inf 立即 SystemExit("DIVERGED")
除此之外与造数器 v 版逐字节一致。真值网络抽样、掩码、噪声、缺失、z-score 流程零改动。

【四护栏】
  ①本 wrapper 及其日志只出：轨迹/数据张量/指纹/收敛验证页。不导出、不打印、不保存
    任何向量场定义与标定常数规格件。
  ②验证链写死：本 wrapper → 鲁班对账器(d771b4ea) → Advisor 验页。引擎本体旗诞生后
    本 wrapper 退为对照件，不废。
  ③改动清单即上方验尸句，三处补丁全列，无第四处。
  ④包络前置判据（量化，入 addendum）：Hill 臂 vs 基线臂峰值比值 ∈ [0.3, 3] 且轨迹
    无 NaN/Inf，二者任一不过即 FAIL，不放行后续训练。
【两焊点】数值节：dt 减半收敛（0.5 vs 0.25，采样网格 rel L∞ < 1%）；双轨迹档由
  收敛验证页记录，不出引擎。

用法（服务器 /dfs/data/twin-ode，venv python）：
  python hill_wrapper_v1.0.py --data_dir data --kgml_dir kegg_kgml \
      --out_root data_sim_hill --hill_H 2.0 --hill_K 1.0 --seed 42
输出：
  data_sim_hill/base_dt0.5/    基线（tanh 原场）造数
  data_sim_hill/hill_dt0.5/    Hill 臂
  data_sim_hill/hill_dt0.25/   Hill 臂 dt 减半（收敛验证）
  data_sim_hill/收敛验证页.json 峰值比值 / 发散闸 / dt 收敛 rel-L∞ / 指纹
"""

import os
import re
import sys
import json
import shutil
import hashlib
import argparse
import subprocess
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data_dir", type=str, default="data")
p.add_argument("--kgml_dir", type=str, default="kegg_kgml")
p.add_argument("--out_root", type=str, default="data_sim_hill")
p.add_argument("--src", type=str, default="半合成仿真_已知真值基准_带注释.py")
p.add_argument("--hill_H", type=float, default=2.0)
p.add_argument("--hill_K", type=float, default=1.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--noise", type=float, default=0.1)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--peak_lo", type=float, default=0.3)
p.add_argument("--peak_hi", type=float, default=3.0)
p.add_argument("--conv_tol", type=float, default=0.01, help="dt减半收敛 rel-L∞ 阈值")
args = p.parse_args()

PY = sys.executable
os.makedirs(args.out_root, exist_ok=True)
PATCHED = os.path.join(args.out_root, "半合成仿真_Hill版_wrapper生成.py")

# ---------- 读本体，施三处补丁（本体不动） ----------
with open(args.src, encoding="utf-8") as f:
    src = f.read()

# 补丁B：dt 硬编码 → 环境变量
old_dt = "dt, tmax = 0.5, 240.0"
assert src.count(old_dt) == 1, "补丁B锚点不唯一，停手"
src = src.replace(old_dt, 'dt, tmax = float(os.environ.get("HILL_DT", "0.5")), 240.0')

# 补丁A：tanh → 奇对称 Hill（锚点=rhs 首行四连 tanh）
old_nl = "    G_ = np.tanh(G); P_ = np.tanh(P); M_ = np.tanh(M); S_ = np.tanh(S)"
assert src.count(old_nl) == 1, "补丁A锚点不唯一，停手"
hill_def = (
    f"_HH = {args.hill_H!r}; _KK = {args.hill_K!r}\n"
    "def hill(z):\n"
    "    az = np.abs(z)\n"
    "    return z * az**(_HH - 1.0) / (_KK**_HH + az**_HH)\n"
)
new_nl = "    G_ = hill(G); P_ = hill(P); M_ = hill(M); S_ = hill(S)"
anchor_rhs = "def rhs(x, t, cond):"
assert src.count(anchor_rhs) == 1, "rhs 锚点不唯一，停手"
src = src.replace(anchor_rhs, hill_def + anchor_rhs)
src = src.replace(old_nl, new_nl)

# 补丁C：发散硬闸
old_ret = "    return np.stack(traj)          # (481, dim)"
assert src.count(old_ret) == 1, "补丁C锚点不唯一，停手"
new_ret = (
    "    _tr = np.stack(traj)\n"
    "    if not np.isfinite(_tr).all():\n"
    "        raise SystemExit('DIVERGED: non-finite trajectory (Hill arm)')\n"
    "    return _tr"
)
src = src.replace(old_ret, new_ret)

with open(PATCHED, "w", encoding="utf-8") as f:
    f.write(src)

def md5_8(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

def run_arm(script, out_dir, dt, extra_env=None):
    """子进程跑造数；轨迹/数据只落盘，不回传任何场定义。"""
    env = dict(os.environ)
    env["HILL_DT"] = str(dt)
    if extra_env:
        env.update(extra_env)
    cmd = [PY, script, "--data_dir", args.data_dir, "--kgml_dir", args.kgml_dir,
           "--out_dir", out_dir, "--seed", str(args.seed),
           "--noise", str(args.noise), "--reps", str(args.reps)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log = r.stdout[-2000:] + ("\n[stderr]\n" + r.stderr[-2000:] if r.returncode != 0 else "")
    diverged = ("DIVERGED" in (r.stdout + r.stderr)) or r.returncode != 0
    return diverged, log

report = {"wrapper": "hill_wrapper_v1.0", "hill_H": args.hill_H, "hill_K": args.hill_K,
          "seed": args.seed, "arms": {}, "gates": {}}

# ---------- 三臂：基线 tanh / Hill dt0.5 / Hill dt0.25 ----------
arms = [("base_dt0.5", args.src, 0.5), ("hill_dt0.5", PATCHED, 0.5), ("hill_dt0.25", PATCHED, 0.25)]
for name, script, dt in arms:
    od = os.path.join(args.out_root, name)
    os.makedirs(od, exist_ok=True)
    diverged, log = run_arm(script, od, dt)
    fp = {}
    for fn in ["train_tensors.npz", "ground_truth.npz", "zstats.npz"]:
        fpth = os.path.join(od, fn)
        if os.path.exists(fpth):
            fp[fn] = md5_8(fpth)
    report["arms"][name] = {"dt": dt, "diverged": bool(diverged), "fingerprints": fp}
    print(f"[{name}] diverged={diverged} fp={fp}")
    if diverged:
        print(f"[{name}] 尾部日志:\n{log}")

# ---------- 闸④：包络峰值比值（z 空间峰值，hill_dt0.5 vs base_dt0.5） ----------
def load_X(od):
    d = np.load(os.path.join(od, "train_tensors.npz"))
    return {k: d[k] for k in d.files if re.fullmatch(r"X[GPSM]", k)}

g = report["gates"]
try:
    Xb, Xh = load_X(os.path.join(args.out_root, "base_dt0.5")), load_X(os.path.join(args.out_root, "hill_dt0.5"))
    ratios = {k: float(np.abs(Xh[k]).max() / max(np.abs(Xb[k]).max(), 1e-12)) for k in Xb}
    g["peak_ratio"] = ratios
    g["peak_ok"] = all(args.peak_lo <= v <= args.peak_hi for v in ratios.values())
    g["no_nan_ok"] = not any(report["arms"][a]["diverged"] for a, _, _ in arms)
except Exception as e:
    g["peak_ok"], g["no_nan_ok"] = False, False
    g["peak_error"] = str(e)

# ---------- 焊点：dt 减半收敛（采样网格 rel-L∞） ----------
try:
    X1 = load_X(os.path.join(args.out_root, "hill_dt0.5"))
    X2 = load_X(os.path.join(args.out_root, "hill_dt0.25"))
    rels = {}
    for k in X1:
        a, b = X1[k], X2[k]
        denom = max(np.abs(a).max(), 1e-12)
        rels[k] = float(np.abs(a - b).max() / denom)
    g["dt_convergence_rel_linf"] = rels
    g["dt_conv_ok"] = all(v < args.conv_tol for v in rels.values())
except Exception as e:
    g["dt_conv_ok"] = False
    g["dt_error"] = str(e)

g["PASS"] = bool(g.get("peak_ok") and g.get("no_nan_ok") and g.get("dt_conv_ok"))

vp = os.path.join(args.out_root, "收敛验证页.json")
with open(vp, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n收敛验证页: {vp} (md5前8={md5_8(vp)})")
print(f"闸门总判: {'PASS' if g['PASS'] else 'FAIL'}  peak_ok={g.get('peak_ok')} "
      f"no_nan_ok={g.get('no_nan_ok')} dt_conv_ok={g.get('dt_conv_ok')}")
print("wrapper 只出轨迹/数据/指纹；后续接 鲁班对账器 → Advisor 验页。")
