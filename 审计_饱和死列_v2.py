# -*- coding: utf-8 -*-
"""tanh 饱和审计。

仅提供 zstats 时，输出偏移级统计；另行提供内部 train_tensors.npz 时，
再计算轨迹级饱和。train_tensors.npz 不随公开包提供。
"""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--zstats", type=Path, default=Path("data_sim2/zstats.npz"))
parser.add_argument("--train-tensors", type=Path, default=None)
args = parser.parse_args()

zs = np.load(args.zstats)
print("===== 严判: |zmu| >= 19 =====")
with np.errstate(over="ignore", invalid="ignore"):
    for L in "GPSM":
        mu = np.asarray(zs["zmu_" + L], dtype=float)
        mu = mu[~np.isnan(mu)]
        n19 = int((np.abs(mu) >= 19).sum())
        nz = int((1.0 / np.cosh(mu) ** 2 == 0.0).sum())
        print("%s  n=%d  max|mu|=%.1f  死列(|mu|>=19)=%d  sech^2==0.0计数=%d"
              % (L, len(mu), float(np.abs(mu).max()), n19, nz))

print("\n===== 宽判: |zmu| > 3.0 及阈值敏感性 =====")
ths = [2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]
print("层    n   | " + "  ".join(">%.2f" % t for t in ths))
for L in "GPSM":
    mu = np.asarray(zs["zmu_" + L], dtype=float)
    mu = mu[~np.isnan(mu)]
    print("%s  %5d | " % (L, len(mu))
          + "  ".join("%6d" % int((np.abs(mu) > t).sum()) for t in ths))

if args.train_tensors:
    tt = np.load(args.train_tensors)
    print("\n===== 轨迹级: 观测点 |x_phys|>3.0 占比100% 的 P 层特征数 =====")
    mu_P = np.asarray(zs["zmu_P"], dtype=float)
    sd_P = np.asarray(zs["zsd_P"], dtype=float)
    XP = np.asarray(tt["XP"], dtype=float)
    OP = np.asarray(tt["OP"]).astype(bool)
    x_phys = mu_P[:, None, None] + sd_P[:, None, None] * XP
    n_obs = OP.sum(axis=(1, 2))
    n_sat = ((np.abs(x_phys) > 3.0) & OP).sum(axis=(1, 2))
    frac = np.where(n_obs > 0, n_sat / np.maximum(n_obs, 1), np.nan)
    print("轨迹级死列数:", int(((frac >= 1.0) & (n_obs > 0)).sum()))

    print("\n===== 宽判死列的观测归属 =====")
    obs_feat = n_obs > 0
    dead3 = np.abs(mu_P) > 3.0
    dead3 = np.where(np.isnan(mu_P), False, dead3)
    print("P 层宽判死列:", int(dead3.sum()),
          " 其中有观测:", int((dead3 & obs_feat).sum()),
          " 无观测:", int((dead3 & ~obs_feat).sum()))
else:
    print("\n未提供 train_tensors.npz，跳过轨迹级饱和统计。")

print("\n===== 核对已验证计数 =====")
ok = True
ok &= int((np.abs(np.nan_to_num(zs["zmu_P"], nan=0.0)) >= 19).sum()) == 35
ok &= int((np.abs(np.nan_to_num(zs["zmu_G"], nan=0.0)) >= 19).sum()) == 7
ok &= int((np.abs(np.nan_to_num(zs["zmu_S"], nan=0.0)) >= 19).sum()) == 7
ok &= int((np.abs(np.nan_to_num(zs["zmu_M"], nan=0.0)) >= 19).sum()) == 2
ok &= int((np.abs(np.nan_to_num(zs["zmu_P"], nan=0.0)) > 3.0).sum()) == 127
print("严判 P35/G7/S7/M2 + 宽判 P127：", "全部一致" if ok else "不一致")
