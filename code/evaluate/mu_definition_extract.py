# 跑什么：把 μ 的坐标系、公式、代码行整理成定稿证据；只读
import os, re, json
ROOT="/dfs/data/twin-ode"; OUT=f"{ROOT}/1024_mu定义_代码与结果"
lines=[]
# 找训练脚本
cands=[]
for root,dirs,files in os.walk(ROOT):
    for f in files:
        if f.startswith("训练脚本") and f.endswith(".py"):
            cands.append(os.path.join(root,f))
lines.append(f"找到训练脚本 {len(cands)} 个")
for c in cands[:5]: lines.append(f"  {c}")

target=None
for c in cands:
    t=open(c,encoding="utf-8",errors="replace").read()
    if "oracle_zconj" in t and "zmu_" in t:
        target=c; break
if target:
    lines.append(f"\n选中: {target}")
    body=open(target,encoding="utf-8",errors="replace").read().splitlines()
    # 抓 μ 相关行
    pat=re.compile(r"zmu_|zsd_|mu\[|sd\[|sech|tanh|bad|_n_nan|zstats|zscore|z-score")
    lines.append("\n--- μ 相关代码行 ---")
    for i,l in enumerate(body,1):
        if pat.search(l):
            lines.append(f"{i}: {l.rstrip()[:200]}")
else:
    lines.append("缺口: 未找到含 oracle_zconj 与 zmu_ 的脚本")

# zstats 键与统计
zp=f"{ROOT}/data_sim2/zstats.npz"
if os.path.exists(zp):
    import numpy as np
    zs=np.load(zp,allow_pickle=True)
    lines.append(f"\n--- {zp} 键 ---")
    for k in zs.keys():
        a=zs[k]
        lines.append(f"  {k}: shape={a.shape}, finite={int(np.isfinite(a).sum())}, nan={int(np.isnan(a).sum())}")
    for k in ["zmu_P","zsd_P"]:
        if k in zs:
            a=zs[k].astype(float)
            lines.append(f"  {k}: min={np.nanmin(a):.4f}, max={np.nanmax(a):.4f}, mean={np.nanmean(a):.4f}")

txt="\n".join(lines)
open(f"{OUT}/mu定义证据_1024.txt","w",encoding="utf-8").write(txt)
print("DONE")
