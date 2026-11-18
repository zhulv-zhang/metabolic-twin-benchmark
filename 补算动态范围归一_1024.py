import argparse
from pathlib import Path
import pandas as pd, numpy as np

parser = argparse.ArgumentParser(description="计算 KO 结果的动态范围归一指标")
parser.add_argument("--input", default="detail_1024.csv", help="KO 明细 CSV")
parser.add_argument("--out", default="12KO幅度指标_归一_1024.txt", help="输出文本")
args = parser.parse_args()
d=pd.read_csv(args.input)
d["near_flat_ep"]=d["true_ep"].abs()<1e-6
d["near_flat_max"]=d["maxabs_d_true"]<1e-6
d["rel_err"]=np.where(d["near_flat_ep"], np.nan,
    (d["fit_ep"]-d["true_ep"]).abs()/d["true_ep"].abs())
d["rel_err_maxabs"]=np.where(d["near_flat_max"], np.nan,
    (d["fit_ep"]-d["true_ep"]).abs()/d["maxabs_d_true"])
d["amp_ratio"]=np.where(d["near_flat_max"], np.nan,
    d["maxabs_d_fit"]/d["maxabs_d_true"])

def grade(r):
    if r["hop"]==1 and r["conflict"]==0: return "Q"
    if r["hop"]==1 and r["conflict"]==1: return "D"
    if r["hop"]==2: return "N"
    return "X"
d["grade"]=d.apply(grade,axis=1)

lines=[f"detail 行数: {len(d)}"]
for g in ["Q","D","N","X"]:
    sub=d[d["grade"]==g]
    v=sub[~sub["near_flat_max"]]
    lines.append(f"\n{g}: n={len(sub)}, 有效={len(v)}")
    if len(v)==0: continue
    lines.append(f"  rel_err_maxabs 中位={v['rel_err_maxabs'].median():.4f}, mean={v['rel_err_maxabs'].mean():.4f}")
    lines.append(f"  amp_ratio 中位={v['amp_ratio'].median():.4f}, IQR=[{v['amp_ratio'].quantile(.25):.4f},{v['amp_ratio'].quantile(.75):.4f}]")

lines.append("\n=== true_ep 与 maxabs_d_true 尺度 ===")
for g in ["Q","D","N","X"]:
    sub=d[d["grade"]==g]
    if len(sub)==0: continue
    r=sub['true_ep'].abs().median()/max(sub['maxabs_d_true'].median(),1e-10)
    lines.append(f"{g}: |true_ep|中位={sub['true_ep'].abs().median():.4f}, maxabs_d_true中位={sub['maxabs_d_true'].median():.4f}, 比={r:.4f}")

lines.append("\n=== Q 组端点 ===")
q=d[d["grade"]=="Q"]
lines.append(f"true_ep: min={q['true_ep'].min():.4f}, max={q['true_ep'].max():.4f}, 中位={q['true_ep'].median():.4f}")
lines.append(f"fit_ep : min={q['fit_ep'].min():.4f}, max={q['fit_ep'].max():.4f}, 中位={q['fit_ep'].median():.4f}")
if len(q)>=2 and q["true_ep"].std()>1e-10:
    sl,ic=np.polyfit(q["true_ep"],q["fit_ep"],1)
    pr=np.corrcoef(q["true_ep"],q["fit_ep"])[0,1]
    lines.append(f"端点 Pearson r={pr:.4f}, slope={sl:.4f}, intercept={ic:.4f}")

q_same=q[np.sign(q["true_ep"])==np.sign(q["fit_ep"])]
lines.append(f"\n=== Q 组符号一致子集 n={len(q_same)} ===")
if len(q_same)>=2:
    lines.append(f"  rel_err 中位={q_same['rel_err'].median():.4f}")
    if len(q_same)>=3 and q_same["true_ep"].std()>1e-10:
        sl,ic=np.polyfit(q_same["true_ep"],q_same["fit_ep"],1)
        lines.append(f"  calib_slope={sl:.4f}, intercept={ic:.4f}")

txt="\n".join(lines)
Path(args.out).write_text(txt, encoding="utf-8")
print(txt)
