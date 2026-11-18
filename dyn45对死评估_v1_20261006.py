# -*- coding: utf-8 -*-
"""评估 dyn45 的 15 次 VIM 结果：GE 域内 AUROC 与 Spearman 相关。

真值、候选掩码、切片数据、VIM 结果和清单均为内部输入，不随公开包提供。
只读取受信任的 pickle 文件。
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gt", type=Path, default=Path("data_sim2/ground_truth.npz"))
parser.add_argument("--mask", type=Path, default=Path("mask_GErel_edges.csv"))
parser.add_argument("--input-csv", type=Path,
                    default=Path("data_sim2izero_restore_b口径228_Insulin.csv"))
parser.add_argument("--vim-dir", type=Path, default=Path("dyn45_1006_zero"))
parser.add_argument("--manifest", default="dyn45_manifest_1006.json")
parser.add_argument("--out", type=Path, default=Path("dyn45_eval_1006.json"))
args = parser.parse_args()

def auroc_tie(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = int(l.sum()), int((~l).sum())
    assert n_pos > 0 and n_neg > 0, "正负类缺一，停止计算"
    r = rankdata(s)
    return (r[l].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

gt = np.load(args.gt, allow_pickle=True)
print("GT keys:", [(k, getattr(gt[k], "shape", None)) for k in gt.keys()], flush=True)
gid = [str(g) for g in gt["sub_gene_ids"]]
gpos = {g: i for i, g in enumerate(gid)}
ge_key = [k for k in gt.keys()
          if k.startswith("W_") and k.endswith("_GE")
          and getattr(gt[k], "shape", None) == (len(gid), len(gid))]
assert len(ge_key) == 1, f"GE 块键不唯一或缺失: {ge_key}"
W = np.asarray(gt[ge_key[0]], float)
print("GE 块键:", ge_key[0], "非零:", int((W != 0).sum()), flush=True)

XI = pd.read_csv(args.input_csv)
pick = [str(c) for c in XI.columns]
assert len(pick) == 228, "切片列数不是 228"
locs = {g: gpos[g] for g in pick if g in gpos}
assert len(locs) == 228, "基因名无法与真值输入对应"

mk = pd.read_csv(args.mask)
pairs = [(str(a), str(b)) for a, b in zip(mk.iloc[:, 0], mk.iloc[:, 1])]
dom = [(a, b) for a, b in pairs if a in locs and b in locs]
print(f"域：掩码对 {len(pairs)} → 228 内 {len(dom)}", flush=True)
assert len(dom) >= 100, "评估域过小，请检查切片和掩码输入"

ia = np.array([locs[a] for a, b in dom])
ib = np.array([locs[b] for a, b in dom])
lab_ba = W[ib, ia] != 0
lab_ab = W[ia, ib] != 0
print(f"真值：b行a列非零 {int(lab_ba.sum())}｜a行b列非零 {int(lab_ab.sum())}（域内）", flush=True)

with (args.vim_dir / args.manifest).open(encoding="utf-8") as f:
    man = json.load(f)
assert len(man) == 15, "manifest 不是 15 行"
pidx = {g: i for i, g in enumerate(pick)}
ia2 = np.array([pidx[a] for a, b in dom])
ib2 = np.array([pidx[b] for a, b in dom])
rows = []
for m in man:
    with (args.vim_dir / f"{m['tag']}_VIM.pkl").open("rb") as f:
        V = np.asarray(pickle.load(f)[0], float)
    s = V[ia2, ib2]
    A_ba, A_ab = auroc_tie(s, lab_ba), auroc_tie(s, lab_ab)
    sp = spearmanr(s, np.abs(W[ib, ia]) + np.abs(W[ia, ib])).statistic
    rows.append({"tag": m["tag"], "trees": m["trees"], "seed": m["seed"],
                 "AUROC_ba": round(A_ba, 4), "AUROC_ab": round(A_ab, 4),
                 "spearman": round(float(sp), 4)})
    print(f"[{m['tag']}] AUROC_ba={A_ba:.4f} AUROC_ab={A_ab:.4f} sp={sp:.4f}", flush=True)

df = pd.DataFrame(rows)
best = "ba" if df.AUROC_ba.mean() > df.AUROC_ab.mean() else "ab"
other = "ab" if best == "ba" else "ba"
df["AUROC"] = df[f"AUROC_{best}"]
mb = df[f"AUROC_{best}"].mean()
mo = df[f"AUROC_{other}"].mean()
print(f"\n择向：{best}（均值 {mb:.4f} vs {mo:.4f}）")
print(df[["tag", "trees", "seed", "AUROC", "spearman"]].to_string(index=False))
print("\n分档均值:")
print(df.groupby("trees")["AUROC"].agg(["mean", "std", "min", "max"]).round(4))
print(f"\n15 次汇总：AUROC 均值 {df.AUROC.mean():.4f}"
      f"（SD {df.AUROC.std():.4f}，min {df.AUROC.min():.4f}）")
c1 = df[df.trees == 500].AUROC
print(f"500 trees 子组：均值 {c1.mean():.4f}，min {c1.min():.4f}")
with args.out.open("w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print(f"评估结果：{args.out}")
