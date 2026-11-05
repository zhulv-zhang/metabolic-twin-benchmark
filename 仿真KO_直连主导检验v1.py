# -*- coding: utf-8 -*-
"""检查 D 级实例的终点符号是否跟随真值直连边；输入文件不随公开包提供。"""
import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gt", default="data_sim2/ground_truth.npz")
parser.add_argument("--traj", default="仿真KO_轨迹.npz")
parser.add_argument("--pool", default="仿真KO_v6_多KO池化.csv")
args = parser.parse_args()

gt = np.load(args.gt)
traj = np.load(args.traj)
print("[键结构] traj:", list(traj.keys()))
for k in traj.keys():
    print("  ", k, getattr(traj[k], "shape", None))

gene_ids = [int(str(g).replace("mmu:", "")) for g in gt["sub_gene_ids"]]
gidx = {g: i for i, g in enumerate(gene_ids)}
W_GE = gt["W_GE"]  # [target, source]

with open(args.pool, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
D = [r for r in rows if r["hop"] == "1" and r["conflict"] == "1"]
print(f"\nD级实例 n={len(D)}")

pairs = traj["ko_pairs"]
D_true, D_fit = traj["D_true_ins"], traj["D_fit_ins"]
n_true_f = n_fit_f = n_matched = n_skipped = 0
for r in D:
    src, sp = int(r["source"]), int(r["species"])
    hit = [i for i in range(pairs.shape[0])
           if int(pairs[i][0]) == src and int(pairs[i][1]) == sp]
    if not hit:
        hit = [i for i in range(pairs.shape[0]) if int(pairs[i][0]) == src]
    if not hit:
        print(f"{src}->{sp}: ko_pairs 未匹配，跳过")
        n_skipped += 1
        continue
    i = hit[0]
    t, s = gidx[sp], gidx[src]
    direct = -np.sign(W_GE[t, s])
    s_true = np.sign(D_true[i, -1, t])
    s_fit = np.sign(D_fit[i, -1, t])
    n_matched += 1
    n_true_f += int(s_true == direct)
    n_fit_f += int(s_fit == direct)
    print(f"{src}->{sp}: 直连={direct:+.0f} 真值终点={s_true:+.0f} 拟合终点={s_fit:+.0f} "
          f"真值跟随={s_true == direct} 拟合跟随={s_fit == direct}")
print(f"\n未匹配并跳过: {n_skipped}")
print(f"真值终点跟随直连: {n_true_f}/{n_matched}")
print(f"拟合终点跟随直连: {n_fit_f}/{n_matched}")
