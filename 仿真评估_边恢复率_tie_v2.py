# -*- coding: utf-8 -*-
"""
仿真评估：学到的边 vs 真值边（边恢复率 = 审稿人要的 ground truth 验证）
====================================================================
对比 train_twin.py 在仿真数据上学到的权重矩阵（best.pt）与
半合成仿真保存的真值（ground_truth.npz），对每类边算三个指标：
  AUROC       : 真边在权重绝对值排序中是否靠前（0.5=瞎猜，1.0=完美）
  Top-k 命中率 : 权重最大的前k条边里有多少是真边（k=真边数）
  权重相关性   : 真边上的 |学出权重| 与 |真值权重| 的 Pearson 相关

用法（在 train_twin.py --tag _sim 跑完之后）：
  python 仿真评估_边恢复率_tie_v2.py --data_dir data --sim_dir data_sim2 --tag _sim2or01
"""

import os
import argparse
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--data_dir", type=str, default="data")
p.add_argument("--sim_dir", type=str, default="data_sim2")
p.add_argument("--tag", type=str, default="_sim2or01")
p.add_argument("--scheme", type=int, default=1)
args = p.parse_args()

gt = np.load(os.path.join(args.sim_dir, "ground_truth.npz"))
ckpt_path = os.path.join(args.data_dir, f"scheme{args.scheme}{args.tag}", "best.pt")
if not os.path.exists(ckpt_path):
    raise SystemExit(f"找不到 {ckpt_path}，先用 --tag {args.tag} 跑训练")
state = torch.load(ckpt_path, map_location="cpu")
print(f"加载模型: {ckpt_path}")

# ---------- 无 sklearn 的 AUROC / AUPRC ----------
from scipy.stats import rankdata
def auroc(labels, scores):
    """秩和法 AUROC（并列平均秩——跨环境确定）"""
    ranks = rankdata(scores)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def topk_hit(labels, scores, k=None):
    k = int(k or labels.sum())
    top = np.argsort(-scores)[:k]
    return labels[top].mean()

def auprc(labels, scores):
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab)
    prec = tp / (np.arange(len(lab)) + 1)
    rec = tp / max(lab.sum(), 1)
    # 阶梯积分
    return float(np.sum(prec * np.diff(np.concatenate([[0], rec]))))

# ---------- 逐类边评估 ----------
# 模型参数是子网尺寸；真值也是子网顺序保存的，直接对位
PAIRS = [  # (模型参数名, 候选掩码名, 真值边名, 中文名)
    ("W_GE", "MSK_GE", "T_GE", "GE 基因调控"),
    ("W_SG", "MSK_SG", "T_SG", "SG 信号→转录耦合"),
    ("W_EC", "MSK_EC", "T_EC", "EC 酶→代谢物"),
    ("W_MM", "MSK_MM", "T_MM", "MM 代谢转化"),
    ("W_FB", "MSK_FB", "T_FB", "FB 代谢反馈"),
    ("W_PS", "MSK_PS", "T_PS", "PS 信号级联"),
]

print(f"\n{'边类型':<14} {'真边数':>7} {'AUROC':>7} {'AUPRC':>7} {'Top-k命中':>9} {'权重相关r':>9}")
print("-" * 60)
rows = []
all_lab, all_sco = [], []
for pname, mname, tname, cname in PAIRS:
    if pname not in state:
        print(f"{cname:<14} ⚠ 模型中无此参数（旧版ckpt?）")
        continue
    W = state[pname].cpu().numpy()
    T = gt[tname].astype(bool)
    if W.shape != T.shape:
        print(f"{cname:<14} ⚠ 形状不符 {W.shape} vs {T.shape}")
        continue
    # 评估域 = 掩码候选全集（ground_truth.npz 已存 MSK_*，真边即从此域抽样）。
    # 绝不能用 真边∪学出边 作域：那会把"没学出的真边"挤出分母，AUROC 严重失真（已踩过）
    mask = gt[mname] > 0
    true_edge = T
    lab = true_edge[mask].ravel()
    sco = np.abs(W[mask]).ravel()
    if lab.sum() == 0:
        continue
    au, ap, tk = auroc(lab, sco), auprc(lab, sco), topk_hit(lab, sco)
    # 权重相关性（仅真边上；需真值权重）
    wname = "W_" + tname.split("_")[1]
    r = np.nan
    if wname in gt:
        w_true = np.abs(gt[wname][true_edge].ravel())
        w_hat = np.abs(W[true_edge].ravel())
        if w_true.std() > 0 and w_hat.std() > 0:
            r = np.corrcoef(w_true, w_hat)[0, 1]
    rows.append((cname, int(true_edge.sum()), au, ap, tk, r))
    print(f"{cname:<14} {int(true_edge.sum()):>7} {au:>7.3f} {ap:>7.3f} {tk:>9.1%} {r:>9.3f}")
    all_lab.append(lab); all_sco.append(sco)

if all_lab:
    L = np.concatenate(all_lab); S = np.concatenate(all_sco)
    print("-" * 60)
    print(f"{'总体':<14} {int(L.sum()):>7} {auroc(L, S):>7.3f} {auprc(L, S):>7.3f} {topk_hit(L, S):>9.1%}")
