import os, sys, subprocess, shutil
import numpy as np
ROOT="/dfs/data/twin-ode"; OUT=f"{ROOT}/1024_12KO幅度重跑_代码与结果"

# 1) 选独立批：按 |W| 列和排序，排除已用 12
gt=np.load(f"{ROOT}/data_sim2/ground_truth.npz",allow_pickle=True)
T=gt["T_GE"]; W=gt["W_GE"]; NG=T.shape[0]
cand=[s for s in range(NG) if T[:,s].sum()>=2]
ranked=sorted(cand,key=lambda s:np.abs(W[:,s]).sum(),reverse=True)
dev=ranked[:12]; indep=ranked[12:24]
print(f"候选 {len(cand)} 个")
print(f"开发批(前12): {dev}")
print(f"独立批(13-24): {indep}")

# 2) 复制 v6 脚本，只换 sources
SRC=f"{ROOT}/仿真KO_验证v6_多KO池化.py"
src=open(SRC,encoding="utf-8",errors="replace").read()
anchor='sources = cand[:12]'
assert anchor in src, "找不到 sources = cand[:12]"
new_src=src.replace(anchor, f'sources = {indep!r}', 1)
# 也加 detail 落盘
new_src=new_src.replace("rows = []\n", "rows = []\ndetail_rows = []\n", 1)
a='dir_true, float(np.sign(Dt[-1, j])), float(np.sign(Df[-1, j]))))\n'
det='''        detail_rows.append(dict(
            source=g_ids[s], species=g_ids[j], hop=int(hop), conflict=int(conf),
            true_ep=float(Dt[-1, j]), fit_ep=float(Df[-1, j]),
            maxabs_d_true=float(np.abs(Dt[:, j]).max()),
            maxabs_d_fit=float(np.abs(Df[:, j]).max()),
            r_traj=float(r) if (r == r) else float("nan"),
        ))
'''
new_src=new_src.replace(a, a+det, 1)
DET_INDEP=f"{OUT}/detail_indep_1024.csv"
a2='with open(os.path.join(WD, "仿真KO_v6_多KO池化.csv"), "w", newline="") as f:\n'
dump=f'''import csv as _csv
with open("{DET_INDEP}", "w", newline="", encoding="utf-8") as _f:
    _w = _csv.DictWriter(_f, fieldnames=["source","species","hop","conflict","true_ep","fit_ep","maxabs_d_true","maxabs_d_fit","r_traj"])
    _w.writeheader()
    for _r in detail_rows:
        _w.writerow(_r)
'''
new_src=new_src.replace(a2, dump+a2, 1)

INDEP=f"{OUT}/_v6_indep_1024.py"
open(INDEP,"w",encoding="utf-8").write(new_src)
import ast; ast.parse(new_src)
print(f"\n独立批脚本: {INDEP}")

# 3) 跑独立批
print("\n===== 跑独立批 v6 =====")
r=subprocess.run([sys.executable, INDEP], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-1500:]); print(r.stderr[-1000:])
if r.returncode!=0 or not os.path.exists(DET_INDEP):
    print("FAIL: 独立批运行失败"); sys.exit(0)

# 4) 对比
import pandas as pd
def load_detail(path):
    d=pd.read_csv(path)
    d["near_flat_max"]=d["maxabs_d_true"]<1e-6
    d["rel_err_maxabs"]=np.where(d["near_flat_max"], np.nan,
        (d["fit_ep"]-d["true_ep"]).abs()/d["maxabs_d_true"])
    d["amp_ratio"]=np.where(d["near_flat_max"], np.nan,
        d["maxabs_d_fit"]/d["maxabs_d_true"])
    d["sign_match"]=(np.sign(d["true_ep"])==np.sign(d["fit_ep"])) & (np.sign(d["true_ep"])!=0)
    def grade(r):
        if r["hop"]==1 and r["conflict"]==0: return "Q"
        if r["hop"]==1 and r["conflict"]==1: return "D"
        if r["hop"]==2: return "N"
        return "X"
    d["grade"]=d.apply(grade,axis=1)
    return d

dev_d=load_detail(f"{OUT}/detail_1024.csv")
ind_d=load_detail(DET_INDEP)

lines=["=== 开发批(前12源) vs 独立批(13-24源) ==="]
lines.append(f"开发批 detail: {len(dev_d)} 行; 独立批 detail: {len(ind_d)} 行")
for g in ["Q","D","N","X"]:
    lines.append(f"\n--- {g} ---")
    for tag,d in [("开发",dev_d),("独立",ind_d)]:
        sub=d[d["grade"]==g]
        if len(sub)==0: lines.append(f"  {tag}: n=0"); continue
        v=sub[~sub["near_flat_max"]]
        lines.append(f"  {tag}: n={len(sub)}, 方向对={sub['sign_match'].sum()} ({sub['sign_match'].mean():.1%}), "
                     f"r_traj中位={sub['r_traj'].median():.3f}, "
                     f"amp_ratio中位={v['amp_ratio'].median():.2f} (有效n={len(v)})")

lines.append("\n=== 复现性判读 ===")
q_dev=dev_d[dev_d["grade"]=="Q"]; q_ind=ind_d[ind_d["grade"]=="Q"]
lines.append(f"Q 方向: 开发 {q_dev['sign_match'].mean():.1%} (n={len(q_dev)}) vs 独立 {q_ind['sign_match'].mean():.1%} (n={len(q_ind)})")
lines.append(f"Q r_traj: 开发 {q_dev['r_traj'].median():.3f} vs 独立 {q_ind['r_traj'].median():.3f}")
lines.append(f"Q amp_ratio: 开发 {q_dev['amp_ratio'].median():.2f} vs 独立 {q_ind['amp_ratio'].median():.2f}")

txt="\n".join(lines)
open(f"{OUT}/独立验证_对比_1024.txt","w",encoding="utf-8").write(txt)
print("\n"+txt)
