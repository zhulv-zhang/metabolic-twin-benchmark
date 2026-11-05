import argparse
from pathlib import Path
import pandas as pd, numpy as np
from math import comb

parser = argparse.ArgumentParser(description="比较开发批和独立批的 KO 结果稳健性")
parser.add_argument("--dev", default="detail_1024.csv", help="开发批明细 CSV")
parser.add_argument("--indep", default="detail_indep_1024.csv", help="独立批明细 CSV")
parser.add_argument("--out", default="独立验证_稳健性_1024.txt", help="输出文本")
args = parser.parse_args()

def load(path):
    d=pd.read_csv(path)
    d["sign_match"]=(np.sign(d["true_ep"])==np.sign(d["fit_ep"])) & (np.sign(d["true_ep"])!=0)
    d["near_flat_max"]=d["maxabs_d_true"]<1e-6
    d["amp_ratio"]=np.where(d["near_flat_max"], np.nan, d["maxabs_d_fit"]/d["maxabs_d_true"])
    def g(r):
        if r["hop"]==1 and r["conflict"]==0: return "Q"
        if r["hop"]==1 and r["conflict"]==1: return "D"
        if r["hop"]==2: return "N"
        return "X"
    d["grade"]=d.apply(g,axis=1)
    return d

dev=load(args.dev)
ind=load(args.indep)

def binom_two(k,n,p=0.5):
    if n==0: return float("nan")
    pmf=[comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(n+1)]
    return float(min(1.0,sum(x for x in pmf if x<=pmf[k]+1e-12)))

lines=["=== 独立批 100% 是真复现还是假象 ==="]

lines.append("\n1) Q 组按 source 分布")
for tag,d in [("开发",dev),("独立",ind)]:
    q=d[d["grade"]=="Q"]
    g=q.groupby("source").agg(n=("species","size"), 方向对=("sign_match","sum"))
    g["正确率"]=g["方向对"]/g["n"]
    lines.append(f"\n{tag} Q 组 (n={len(q)}, source 数={len(g)})")
    lines.append(g.to_string())
    if len(g)>0:
        top=g["n"].max()
        lines.append(f"  最大单源贡献: {top}/{len(q)} = {top/len(q):.1%}")

lines.append("\n2) true_ep 绝对值分布对比（信号强度）")
for g in ["Q","D","N","X"]:
    for tag,d in [("开发",dev),("独立",ind)]:
        sub=d[d["grade"]==g]
        if len(sub)==0: continue
        lines.append(f"  {g} {tag}: |true_ep|中位={sub['true_ep'].abs().median():.4f}, "
                     f"maxabs_d_true中位={sub['maxabs_d_true'].median():.4f}, n={len(sub)}")

lines.append("\n3) 各批各组方向正确率 + 二项 p（双侧，H0=0.5）")
for g in ["Q","D","N","X"]:
    for tag,d in [("开发",dev),("独立",ind)]:
        sub=d[d["grade"]==g]
        if len(sub)==0: continue
        k=int(sub["sign_match"].sum()); n=len(sub)
        lines.append(f"  {g} {tag}: {k}/{n} ({k/n:.1%}), 双侧 p={binom_two(k,n):.4f}")

lines.append("\n4) 独立批 N 组逐实例")
nsub=ind[ind["grade"]=="N"][["source","species","hop","conflict","true_ep","fit_ep","sign_match"]]
lines.append(nsub.to_string())

lines.append("\n5) 开发批 N 组逐实例")
nsub2=dev[dev["grade"]=="N"][["source","species","hop","conflict","true_ep","fit_ep","sign_match"]]
lines.append(nsub2.to_string())

txt="\n".join(lines)
Path(args.out).write_text(txt, encoding="utf-8")
print(txt)
