# -*- coding: utf-8 -*-
# 仿真KO验证v5: 结构置信标记 — 从拟合网络(不需要真值)识别「对消节点」
# (从扰动源出发存在≥2条符号相反的≤2跳路径), 检验: 冲突组预测错误率是否显著高于单路径组
import os, glob
import numpy as np
import torch
import csv

NG, DIM = 691, 2106
WD = "/dfs/data/twin-ode"

def find_one(patterns, desc):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(WD, pat), recursive=True))
        if hits:
            print(f"{desc}: {os.path.relpath(hits[0], WD)}")
            return hits[0]
    raise FileNotFoundError(desc)

gt = np.load(find_one(["data_sim2/ground_truth.npz"], "真值"), allow_pickle=True)
ck1 = torch.load(find_one(["**/scheme1_sim2or01/best.pt"], "FIT1"), map_location="cpu")
sd_ck = ck1 if "W_GE" in ck1 else ck1.get("model", ck1.get("state_dict", ck1))
W_eff = (sd_ck["W_GE"].detach().float() * torch.tensor(gt["MSK_GE"].astype(np.float32))).numpy()
g_ids = [str(x).split(":")[-1] for x in gt["sub_gene_ids"]]
gpos = {v: i for i, v in enumerate(g_ids)}
i_sre = gpos["20787"]

S = np.sign(W_eff)
S[np.abs(W_eff) < 0.05] = 0.0   # 弱边忽略(噪声级)
direct = np.nonzero(S[:, i_sre])[0]
print(f"拟合网络中 Srebf1 出边数(|w|≥0.05): {len(direct)}")

# 每个 G 层物种: 收集来自 Srebf1 的 ≤2 跳路径符号
conflict = np.zeros(NG, dtype=bool)
n_paths = np.zeros(NG, dtype=int)
for j in range(NG):
    signs = []
    if S[j, i_sre] != 0:
        signs.append(S[j, i_sre])                       # 1跳
    for k in direct:
        if S[j, k] != 0:
            signs.append(S[j, k] * S[k, i_sre])         # 2跳
    n_paths[j] = len(signs)
    conflict[j] = len(signs) >= 2 and (min(signs) < 0 < max(signs))

# 读 v4 CSV: 远点两条件下的端点符号对错(只含响应物种)
rows = list(csv.DictReader(open(os.path.join(WD, "仿真KO_v4_置信校准.csv"))))
for cname in ["insulin", "pbs"]:
    sub = [r for r in rows if r["cond"] == cname and r["layer"] == "G"]
    gid2idx = {r["species"]: gpos.get(r["species"]) for r in sub}
    ok = np.array([r["fit1_correct"] == "True" for r in sub])
    idx = np.array([gid2idx[r["species"]] for r in sub])
    reach = np.array([n_paths[i] > 0 for i in idx])
    conf = np.array([conflict[i] for i in idx])
    print(f"\n== 远点/{cname} == G层响应物种 {len(sub)} | 其中G层网络可达 {int(reach.sum())}")
    for lab, m in [("单路径(不冲突)", reach & ~conf), ("对消节点(冲突)", reach & conf), ("网络外(0路径)", ~reach)]:
        n = int(m.sum())
        if n:
            print(f"  {lab}: n={n:4d}, 端点符号正确率={ok[m].mean():.0%}")
    i_p53, i_stat3 = gpos["22059"], gpos["20848"]
    tag = {True: "冲突", False: "单路径"}
    print(f"  抽查: p53={tag[conflict[i_p53]]}({n_paths[i_p53]}条路径) | Stat3={tag[conflict[i_stat3]]}({n_paths[i_stat3]}条路径)")

print("\n判读: 对消节点组正确率显著低于单路径组 ⇒ 结构标记可作置信信号(不需要真值)")
