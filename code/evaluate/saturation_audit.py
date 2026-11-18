# -*- coding: utf-8 -*-
# 审计_饱和死列_v2.py —— tanh 饱和死列审计（KI-05 终裁口径, 2026-09-11）
# 口径（与论文 §2.6 逐字一致）:
#   严判 = 偏移级 |zmu| >= 19  → sech^2(mu) ~ 1e-16, numerically zero（非 float64 精确零）
#   宽判 = 偏移级 |zmu| > 3.0   → sech^2(mu) < 0.01（敏感性 2.0-4.0）
#   轨迹级 = 观测点上 |x_phys| > 3.0 占比 100%（验证"非全程锁死型"）
# 机制: dS/dt 与 dM/dt 经 tanh(P) 吃 P 层调控; 工作点偏移过大 → sech^2(μ)≈0
#       → W_PS 第 j 列 / W_EC 第 j 列整列梯度消失 = 死列（工作点偏移型饱和）
# 输入: data_sim2/zstats.npz (zmu_*, zsd_*) + train_tensors.npz (XP, OP)
# 输出: 全部原样打印, 不做舍入修饰
import numpy as np

zs = np.load("data_sim2/zstats.npz")
tt = np.load("data_sim2/train_tensors.npz")

print("===== 严判: |zmu| >= 19（numerically zero, sech^2 ~ 1e-16）=====")
for L in "GPSM":
    mu = np.asarray(zs["zmu_" + L], dtype=float)
    mu = mu[~np.isnan(mu)]
    n19 = int((np.abs(mu) >= 19).sum())
    nz  = int((1.0 / np.cosh(mu) ** 2 == 0.0).sum())
    print("%s  n=%d  max|mu|=%.1f  死列(|mu|>=19)=%d  sech^2==0.0精确计数=%d"
          % (L, len(mu), float(np.abs(mu).max()), n19, nz))

print("\n===== 宽判: |zmu| > 3.0（sech^2 < 0.01）及敏感性 =====")
ths = [2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]
print("层    n   | " + "  ".join(">%.2f" % t for t in ths))
for L in "GPSM":
    mu = np.asarray(zs["zmu_" + L], dtype=float)
    mu = mu[~np.isnan(mu)]
    print("%s  %5d | " % (L, len(mu)) + "  ".join("%6d" % int((np.abs(mu) > t).sum()) for t in ths))

print("\n===== 轨迹级: 观测点 |x_phys|>3.0 占比100% 的 P 层特征数 =====")
mu_P = np.asarray(zs["zmu_P"], dtype=float); sd_P = np.asarray(zs["zsd_P"], dtype=float)
XP = np.asarray(tt["XP"], dtype=float); OP = np.asarray(tt["OP"]).astype(bool)
x_phys = mu_P[:, None, None] + sd_P[:, None, None] * XP
n_obs = OP.sum(axis=(1, 2)); n_sat = ((np.abs(x_phys) > 3.0) & OP).sum(axis=(1, 2))
frac = np.where(n_obs > 0, n_sat / np.maximum(n_obs, 1), np.nan)
print("轨迹级死列数:", int(((frac >= 1.0) & (n_obs > 0)).sum()))

print("\n===== 宽判死列的观测归属（finding 3）=====")
obs_feat = n_obs > 0
dead3 = np.abs(mu_P) > 3.0
dead3 = np.where(np.isnan(mu_P), False, dead3)
print("P 层宽判死列:", int(dead3.sum()), " 其中有观测:", int((dead3 & obs_feat).sum()),
      " 无观测:", int((dead3 & ~obs_feat).sum()))

print("\n===== 核对已知值（全过则脚本与已验证数字一致）=====")
ok = True
ok &= abs(int((np.abs(np.nan_to_num(zs["zmu_P"], nan=0.0)) >= 19).sum()) - 35) == 0
ok &= abs(int((np.abs(np.nan_to_num(zs["zmu_G"], nan=0.0)) >= 19).sum()) - 7) == 0
ok &= abs(int((np.abs(np.nan_to_num(zs["zmu_S"], nan=0.0)) >= 19).sum()) - 7) == 0
ok &= abs(int((np.abs(np.nan_to_num(zs["zmu_M"], nan=0.0)) >= 19).sum()) - 2) == 0
ok &= abs(int((np.abs(np.nan_to_num(zs["zmu_P"], nan=0.0)) > 3.0).sum()) - 127) == 0
print("严判 P35/G7/S7/M2 + 宽判 P127 核对:", "全部通过" if ok else "不一致——禁止送审, 原样贴回")
