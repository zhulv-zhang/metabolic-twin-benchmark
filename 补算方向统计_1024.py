import argparse
from pathlib import Path
import pandas as pd, numpy as np
from math import comb

parser = argparse.ArgumentParser(description="计算 KO 终点方向准确率及二项检验")
parser.add_argument("--input", default="detail_1024.csv", help="KO 明细 CSV")
parser.add_argument("--out", default="12KO方向统计_1024.txt", help="输出文本")
args = parser.parse_args()
d=pd.read_csv(args.input)
d["sign_true"]=np.sign(d["true_ep"])
d["sign_fit"]=np.sign(d["fit_ep"])
d["sign_match"]=(d["sign_true"]==d["sign_fit"]) & (d["sign_true"]!=0)
d["near_flat_ep"]=d["true_ep"].abs()<1e-6
d["near_flat_max"]=d["maxabs_d_true"]<1e-6
d["rel_err_maxabs"]=np.where(d["near_flat_max"], np.nan,
    (d["fit_ep"]-d["true_ep"]).abs()/d["maxabs_d_true"])
d["amp_ratio"]=np.where(d["near_flat_max"], np.nan,
    d["maxabs_d_fit"]/d["maxabs_d_true"])
d["ratio_ep_maxabs"]=np.where(d["near_flat_max"], np.nan,
    d["true_ep"].abs()/d["maxabs_d_true"])

def grade(r):
    if r["hop"]==1 and r["conflict"]==0: return "Q"
    if r["hop"]==1 and r["conflict"]==1: return "D"
    if r["hop"]==2: return "N"
    return "X"
d["grade"]=d.apply(grade,axis=1)

def binom_two_sided(k, n, p=0.5):
    if n==0: return float("nan")
    # 双侧精确二项
    from math import comb
    pmf=[comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(n+1)]
    obs=pmf[k]
    return float(min(1.0, sum(x for x in pmf if x <= obs + 1e-12)))

lines=[f"detail 行数: {len(d)}"]
lines.append("\n=== 方向正确率与二项检验（H0: p=0.5）===")
for g in ["Q","D","N","X"]:
    sub=d[d["grade"]==g]
    if len(sub)==0: continue
    n=len(sub); k=int(sub["sign_match"].sum()); p=k/n
    pval=binom_two_sided(k,n)
    lines.append(f"{g}: n={n}, 方向对={k} ({p:.1%}), 二项双侧 p={pval:.4f}")

lines.append("\n=== 符号一致子集的幅度（方向对了，幅度如何）===")
for g in ["Q","D","N","X"]:
    sub=d[(d["grade"]==g) & d["sign_match"]]
    v=sub[~sub["near_flat_max"]]
    lines.append(f"\n{g} 符号一致: n={len(sub)}, 有效={len(v)}")
    if len(v)==0: continue
    lines.append(f"  amp_ratio 中位={v['amp_ratio'].median():.4f}, IQR=[{v['amp_ratio'].quantile(.25):.4f},{v['amp_ratio'].quantile(.75):.4f}]")
    lines.append(f"  rel_err_maxabs 中位={v['rel_err_maxabs'].median():.4f}")
    if len(v)>=3 and v["true_ep"].std()>1e-10:
        sl,ic=np.polyfit(v["true_ep"],v["fit_ep"],1)
        lines.append(f"  calib_slope={sl:.4f}, intercept={ic:.4f}")

lines.append("\n=== 端点是否接近零（解释幅度误差）===")
for g in ["Q","D","N","X"]:
    sub=d[d["grade"]==g]
    if len(sub)==0: continue
    lines.append(f"{g}: |true_ep|中位={sub['true_ep'].abs().median():.4f}, "
                 f"maxabs_d_true中位={sub['maxabs_d_true'].median():.4f}, "
                 f"比中位={sub['ratio_ep_maxabs'].median():.4f}")

lines.append("\n=== Q vs D 方向对比（论文说 D 方向可信但动态不对）===")
q=d[d["grade"]=="Q"]; dd=d[d["grade"]=="D"]
lines.append(f"Q: 方向 {q['sign_match'].mean():.1%} (n={len(q)}), amp_ratio 中位={q['amp_ratio'].median():.2f}, r_traj 中位={q['r_traj'].median():.3f}")
lines.append(f"D: 方向 {dd['sign_match'].mean():.1%} (n={len(dd)}), amp_ratio 中位={dd['amp_ratio'].median():.2f}, r_traj 中位={dd['r_traj'].median():.3f}")

lines.append("\n=== 两跳是否显著低于随机（论文 below chance）===")
nsub=d[d["grade"]=="N"]
n=len(nsub); k=int(nsub["sign_match"].sum())
# 单侧 H1: p<0.5
from math import comb
p_one=sum(comb(n,i)*(0.5**n) for i in range(k+1))
lines.append(f"N 组: n={n}, 方向对={k} ({k/n:.1%}), 单侧 p(p<0.5)={p_one:.4f}, 双侧 p={binom_two_sided(k,n):.4f}")

txt="\n".join(lines)
Path(args.out).write_text(txt, encoding="utf-8")
print(txt)
