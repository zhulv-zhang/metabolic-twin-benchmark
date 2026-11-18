import numpy as np, os
D="/dfs/data/twin-ode/data"
OUT="/dfs/data/twin-ode/1024饱和审计_代码与结果"
lines=[]

# 论文主模型：data_sim2/zstats.npz（真实 mask 匹配，净化 529）
zp=f"{D}/../data_sim2/zstats.npz"
if not os.path.exists(zp):
    zp="/dfs/data/twin-ode/data_sim2/zstats.npz"
lines.append(f"zstats: {zp}")
zs=np.load(zp,allow_pickle=True)
mu=zs["zmu_P"].astype(np.float64)
sd=zs["zsd_P"].astype(np.float64) if "zsd_P" in zs else np.ones_like(mu)

nan_mask=np.isnan(mu)          # 无观测
obs_mask=~nan_mask             # 有观测
n_obs=int(obs_mask.sum()); n_unobs=int(nan_mask.sum())
lines.append(f"P 特征: 总=691, 有观测(非NaN)={n_obs}, 无观测(NaN)={n_unobs}")
lines.append(f"论文目标: 有观测=162, 无观测=529")

# 净化：NaN -> 0（与训练脚本 L444-451 一致）
mu_clean=np.where(nan_mask,0.0,mu)

lines.append("\n=== 净化后（NaN→0，与训练脚本一致）===")
for thr in [2.0,3.0,4.0,19.0]:
    o=int((np.abs(mu_clean[obs_mask])>thr).sum())
    u=int((np.abs(mu_clean[nan_mask])>thr).sum())
    ro=100.0*o/n_obs if n_obs else float("nan")
    ru=100.0*u/n_unobs if n_unobs else float("nan")
    lines.append(f"|μ|>{thr}: 有观测 {o}/{n_obs} ({ro:.1f}%), 无观测 {u}/{n_unobs} ({ru:.1f}%)")

lines.append("\n=== 论文数字对照 ===")
o3=int((np.abs(mu_clean[obs_mask])>3.0).sum())
o19=int((np.abs(mu_clean[obs_mask])>=19.0).sum())
lines.append(f"论文 |μ|>3.0 有观测 = 127/162; 本次 = {o3}/{n_obs}")
lines.append(f"论文 |μ|≥19  有观测 = 35;     本次 = {o19}/{n_obs}")
lines.append(f"论文 无观测饱和 = 529; 本次无观测组 |μ|>3.0 = {int((np.abs(mu_clean[nan_mask])>3.0).sum())}/529")

# 额外：未净化（NaN 保留）对照，只对有值特征算
lines.append("\n=== 未净化 mu（只对有观测特征）===")
for thr in [3.0,19.0]:
    o=int((np.abs(mu[obs_mask])>thr).sum())
    lines.append(f"|μ|>{thr}: 有观测 {o}/{n_obs}")

txt="\n".join(lines)
open(f"{OUT}/饱和审计v3_1024.txt","w",encoding="utf-8").write(txt)
print("DONE")
