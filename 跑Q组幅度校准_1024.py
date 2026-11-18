# 跑什么：用 仿真KO_定量验证_逐物种.csv 的 endpoint_d_true/fit 算 Q/D/N/X 幅度误差和校准斜率
import argparse
from pathlib import Path
import pandas as pd, numpy as np

parser = argparse.ArgumentParser(description="计算 Q/D/N/X 组幅度误差和校准斜率")
parser.add_argument("--pool", default="仿真KO_v6_多KO池化.csv", help="KO 池化结果 CSV")
parser.add_argument("--detail", default="仿真KO_定量验证_逐物种.csv", help="逐物种结果 CSV")
parser.add_argument("--out", default="Q组幅度校准结果_1024.txt", help="输出文本")
args = parser.parse_args()
lines=[]

pool=pd.read_csv(args.pool,encoding="utf-8-sig")
detail=pd.read_csv(args.detail,encoding="utf-8-sig")
pool["species"]=pool["species"].astype(str)
detail["species"]=detail["species"].astype(str)

lines.append(f"池化: {len(pool)} 行; 逐物种: {len(detail)} 行")
lines.append("\n池化 hop/conflict 分布:")
for (h,c),g in pool.groupby(["hop","conflict"]):
    lines.append(f"  hop={h} conflict={c}: n={len(g)}")

dg=detail[detail["layer"]=="G"]
rows=[]
for _,p in pool.iterrows():
    m=dg[dg["species"]==p["species"]]
    if len(m)==0: continue
    for _,d in m.iterrows():
        te=float(d["endpoint_d_true"]); fe=float(d["endpoint_d_fit"])
        nf=abs(te)<1e-6
        rows.append(dict(source=p["source"],species=p["species"],hop=p["hop"],
            conflict=p["conflict"],cond=d["cond"],true_endpoint=te,fit_endpoint=fe,
            near_flat=nf, endpoint_rel_err=(np.nan if nf else abs(fe-te)/abs(te)),
            maxabs_d_true=float(d["maxabs_d_true"]),maxabs_d_fit=float(d["maxabs_d_fit"])))
m=pd.DataFrame(rows)
lines.append(f"\n匹配后: {len(m)} 行, 唯一实例数: {m.groupby(['source','species']).ngroups}")

def grade(r):
    if r["hop"]==1 and r["conflict"]==0: return "Q"
    if r["hop"]==1 and r["conflict"]==1: return "D"
    if r["hop"]==2: return "N"
    return "X"
m["grade"]=m.apply(grade,axis=1)

lines.append("\n=== 按 Q/D/N/X 分组 ===")
for g in ["Q","D","N","X"]:
    sub=m[m["grade"]==g]
    if len(sub)==0:
        lines.append(f"\n{g}: 0 行"); continue
    valid=sub[~sub["near_flat"]]
    lines.append(f"\n{g}: n={len(sub)}, near_flat={int(sub['near_flat'].sum())}, 有效={len(valid)}")
    if len(valid)>0:
        lines.append(f"  endpoint_rel_err 中位={valid['endpoint_rel_err'].median():.3f}, mean={valid['endpoint_rel_err'].mean():.3f}")
        ratio=(valid["maxabs_d_fit"]/valid["maxabs_d_true"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna()
        if len(ratio)>0:
            lines.append(f"  maxabs 比中位={ratio.median():.3f}")
        if len(valid)>=2 and valid["true_endpoint"].std()>1e-10:
            sl,ic=np.polyfit(valid["true_endpoint"],valid["fit_endpoint"],1)
            lines.append(f"  calib_slope={sl:.4f}, intercept={ic:.4f}")

va=m[~m["near_flat"]]
if len(va)>=2 and va["true_endpoint"].std()>1e-10:
    sl,ic=np.polyfit(va["true_endpoint"],va["fit_endpoint"],1)
    lines.append(f"\n=== 全局校准 ===\nn={len(va)}, calib_slope={sl:.4f}, intercept={ic:.4f}")

rng=np.random.default_rng(42); src=m["source"].unique()
lines.append(f"\n=== 按 source 聚类 bootstrap (source 数={len(src)}) ===")
for col in ["endpoint_rel_err"]:
    vals=[]
    for _ in range(5000):
        pick=rng.choice(src,size=len(src),replace=True)
        sub=pd.concat([m[m["source"]==s] for s in pick])
        v=sub[col].dropna()
        if len(v)>0: vals.append(v.median())
    if vals:
        lo,hi=np.percentile(vals,[2.5,97.5])
        lines.append(f"  {col} 中位 95% CI = [{lo:.3f}, {hi:.3f}]")

txt="\n".join(lines)
Path(args.out).write_text(txt, encoding="utf-8")
print(txt)
