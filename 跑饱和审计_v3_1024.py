# -*- coding: utf-8 -*-
"""按 P 层 z 均值是否缺失区分有无观测，并统计饱和阈值。"""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--zstats", type=Path, default=Path("data_sim2/zstats.npz"))
parser.add_argument("--out", type=Path, default=None, help="可选：同时保存输出文本")
args = parser.parse_args()

zs = np.load(args.zstats)
mu = np.asarray(zs["zmu_P"], dtype=np.float64)
sd = np.asarray(zs["zsd_P"], dtype=np.float64) if "zsd_P" in zs else np.ones_like(mu)
nan_mask = np.isnan(mu)
obs_mask = ~nan_mask
n_obs = int(obs_mask.sum())
n_unobs = int(nan_mask.sum())
lines = [f"zstats: {args.zstats}",
         f"P 特征: 总={len(mu)}，有观测(非NaN)={n_obs}，无观测(NaN)={n_unobs}"]
mu_clean = np.where(nan_mask, 0.0, mu)

lines.append("\n=== 缺失值置零后的阈值统计 ===")
for thr in [2.0, 3.0, 4.0, 19.0]:
    o = int((np.abs(mu_clean[obs_mask]) > thr).sum())
    u = int((np.abs(mu_clean[nan_mask]) > thr).sum())
    ro = 100.0 * o / n_obs if n_obs else float("nan")
    ru = 100.0 * u / n_unobs if n_unobs else float("nan")
    lines.append(f"|μ|>{thr}: 有观测 {o}/{n_obs} ({ro:.1f}%)，"
                 f"无观测 {u}/{n_unobs} ({ru:.1f}%)")

lines.append("\n=== 与已核验计数对照 ===")
o3 = int((np.abs(mu_clean[obs_mask]) > 3.0).sum())
o19 = int((np.abs(mu_clean[obs_mask]) >= 19.0).sum())
lines.append(f"|μ|>3.0 有观测：{o3}/{n_obs}（预期 127/162）")
lines.append(f"|μ|≥19 有观测：{o19}（预期 35）")
lines.append(f"无观测特征数：{n_unobs}（预期 529）")
checks = [n_obs == 162, n_unobs == 529, o3 == 127, o19 == 35]
lines.append("核对结果：" + ("全部一致" if all(checks) else "不一致"))

lines.append("\n=== 未置零时，仅统计有观测特征 ===")
for thr in [3.0, 19.0]:
    o = int((np.abs(mu[obs_mask]) > thr).sum())
    lines.append(f"|μ|>{thr}: {o}/{n_obs}")

txt = "\n".join(lines)
print(txt)
if args.out:
    args.out.write_text(txt, encoding="utf-8")
