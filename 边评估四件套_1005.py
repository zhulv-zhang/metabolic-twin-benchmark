# -*- coding: utf-8 -*-
# 边评估四件套：|W_GE| / 域 MSK_GE 362 / 标签 T_GE / rankdata（tie-corrected）
# 用法: python 边评估四件套_1005.py <edges_csv> [更多csv...]
import argparse
import numpy as np, pandas as pd
from scipy.stats import rankdata
parser = argparse.ArgumentParser(description="在 362 条 GE 候选边上计算 tie-corrected AUROC")
parser.add_argument("edges", nargs="+", help="含 from/to/weight 列的边评分 CSV")
parser.add_argument("--gt", default="data_sim2/ground_truth.npz", help="内部真值输入，不随公开包提供")
args = parser.parse_args()
z = np.load(args.gt, allow_pickle=True)
gg = [str(g) for g in z["sub_gene_ids"]]; g2i = {g: i for i, g in enumerate(gg)}
M = np.asarray(z["MSK_GE"]) > 0; T = np.asarray(z["T_GE"])
ii, jj = np.where(M); y = T[ii, jj].astype(int)
p2k = {(int(ii[k]), int(jj[k])): k for k in range(len(ii))}
def auc(s):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
for path in args.edges:
    df = pd.read_csv(path)
    s = np.zeros(len(ii))
    for a, b, w in zip(df["from"], df["to"], df["weight"]):
        ka = p2k.get((g2i[a], g2i[b])); kb = p2k.get((g2i[b], g2i[a]))
        if ka is not None: s[ka] = abs(w)
        elif kb is not None: s[kb] = abs(w)
    print(f"{path}: AUROC(tie-corr, 362域) = {auc(s):.4f}  域内边 {int((s>0).sum())}/362")