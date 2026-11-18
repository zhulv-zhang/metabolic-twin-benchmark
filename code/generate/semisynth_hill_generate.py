# -*- coding: utf-8 -*-
"""
半合成仿真：已知真值基准（论文审稿人必问项：没有真值怎么证明学的边是对的）
==========================================================================
思路：用真实掩码拓扑 + 随机权重造一个"已知答案的网络" → 用与训练完全相同的
动力学方程仿真出四层组学数据（带噪声、带缺失、按真实时间网格采样）→
再用 train_twin.py 原封不动地训练 → 对比学到的边 vs 真值边。

这样训练代码零改动（只加 --tensor_file / --tag 两个参数），杜绝"仿真一套代码、
训练一套代码"的对不上问题。

用法：
  python 半合成仿真_已知真值基准_带注释.py            # 生成 data_sim/train_tensors.npz + ground_truth.npz
  python train_twin.py --scheme 1 --tensor_file data_sim/train_tensors.npz --tag _sim --data_dir data
  python 仿真评估_边恢复率_带注释.py                   # 对比 data/scheme1_sim/best.pt vs 真值
"""

import os
import json
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data_dir", type=str, default="GSE166336_小鼠组学数据",
               help="真实数据目录（读掩码/索引；服务器上为 /dfs/data/twin-ode/data）")
p.add_argument("--kgml_dir", type=str, default="kegg_kgml")
p.add_argument("--out_dir", type=str, default="data_sim", help="仿真输出目录")
p.add_argument("--p_true", type=float, default=0.3,
               help="掩码候选边中为真边的比例（GE/SG/MM/FB 用；先验是超集，真值是其子集——模拟'掩码宽优于紧'）")
p.add_argument("--p_true_ps", type=float, default=0.01,
               help="PS 层专用真边比例：594² 候选极密，v1 用 0.3→178 真边/节点（vs 16 方程/节点）不可辨识；0.01→约 5 条/节点")
p.add_argument("--p_true_ec", type=float, default=0.1,
               help="EC 层专用真边比例：33 列，v1 用 0.3→约 15 底物/代谢物偏密；0.1→约 5 条")
p.add_argument("--noise", type=float, default=0.1, help="观测噪声 std（z-score空间）")
p.add_argument("--reps", type=int, default=3, help="每条件重复数（模拟生物学重复）")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--ideal", action="store_true",
               help="理想数据对照：全层5min密集网格+零缺失+零噪声（测观测预算上限；真值抽样序列不变，与data_sim2同一真值网络）")
args = p.parse_args()
rng = np.random.default_rng(args.seed)

DATA_DIR = args.data_dir
KGML_DIR = args.kgml_dir
if not os.path.isdir(KGML_DIR) and os.path.isdir(os.path.join(DATA_DIR, "kegg_kgml")):
    KGML_DIR = os.path.join(DATA_DIR, "kegg_kgml")
os.makedirs(args.out_dir, exist_ok=True)

def locate(fname):
    for cand in [fname, os.path.join(DATA_DIR, fname)]:
        if os.path.exists(cand):
            return cand
    return None

# ===================== 1. 索引与掩码（与训练脚本同规则） =====================
with open(locate("gene_index_map.json"), encoding="utf-8") as f:
    gmap = json.load(f)
with open(locate("metabolite_index_map.json"), encoding="utf-8") as f:
    mmap = json.load(f)
gene_ids  = [g for g, _ in sorted(gmap.items(),  key=lambda x: x[1])]
metab_ids = [m for m, _ in sorted(mmap.items(), key=lambda x: x[1])]
gpos = {g: i for i, g in enumerate(gene_ids)}
mpos = {m: i for i, m in enumerate(metab_ids)}
gs_full = np.array([gmap[g] for g in gene_ids])
ms_full = np.array([mmap[m] for m in metab_ids])

M_GE = np.load(locate("mask_GErel.npy"))[np.ix_(gs_full, gs_full)].astype(np.float32)
D_GE = np.load(locate("direction_GErel.npy"))[np.ix_(gs_full, gs_full)].astype(np.float32)
M_EC = np.load(locate("mask_ECrel.npy"))[np.ix_(gs_full, ms_full)].astype(np.float32)
D_EC = np.load(locate("direction_mask.npy"))[np.ix_(gs_full, ms_full)].astype(np.float32)
M_MM = np.load(locate("mask_metabolite2metabolite.npy"))[np.ix_(ms_full, ms_full)].astype(np.float32)
M_FB = np.load(locate("mask_feedback.npy"))[np.ix_(ms_full, gs_full)].astype(np.float32)

# ===================== 2. 子网（与训练脚本同一函数逻辑） =====================
def subnet_from_kgml():
    import re
    import xml.etree.ElementTree as ET
    genes, mets = set(), set()
    for fn in sorted(os.listdir(KGML_DIR)):
        if not fn.endswith((".xml", ".kgml")):
            continue
        root = ET.parse(os.path.join(KGML_DIR, fn)).getroot()
        for ent in root.iter("entry"):
            t, name = ent.get("type", ""), ent.get("name", "")
            if t == "gene":
                for tok in name.split():
                    if tok.startswith("mmu:") and tok in gpos:
                        genes.add(tok)
            elif t == "compound":
                for tok in name.split():
                    tok = tok.replace("cpd:", "")
                    if re.fullmatch(r"C\d{5}", tok) and tok in mpos:
                        mets.add(tok)
    return sorted(genes, key=lambda g: gpos[g]), sorted(mets, key=lambda m: mpos[m])

sub_g, sub_m = subnet_from_kgml()
gi = np.array([gpos[g] for g in sub_g])
mi = np.array([mpos[m] for m in sub_m])
n_g, n_m = len(gi), len(mi)
print(f"仿真子网: {n_g} 基因, {n_m} 代谢物")

sGE = M_GE[np.ix_(gi, gi)]; sD_GE = D_GE[np.ix_(gi, gi)]
sEC = M_EC[np.ix_(gi, mi)]; sD_EC = D_EC[np.ix_(gi, mi)]
sMM = M_MM[np.ix_(mi, mi)]
sFB = M_FB[np.ix_(mi, gi)]
sig = (sFB.sum(0) > 0)
sPS = (sig[:, None] & sig[None, :]).astype(np.float32)
sBeta = np.ones(n_g, dtype=np.float32)   # 仿真里所有基因都有翻译（真值自定）

# ===================== 3. 造真值网络 =====================
def sample_W(mask, dirmask=None, p=None):
    """掩码边里抽 p（默认 args.p_true）为真边；权重~N(0,0.4)；有方向先验的80%服从符号"""
    p = args.p_true if p is None else p
    true_edge = (mask > 0) & (rng.random(mask.shape) < p)
    W = np.zeros(mask.shape, dtype=np.float32)
    W[true_edge] = rng.normal(0, 0.4, true_edge.sum()).astype(np.float32)
    if dirmask is not None:
        flip = (dirmask != 0) & true_edge & (rng.random(mask.shape) < 0.8)
        W[flip] = np.abs(W[flip]) * np.sign(dirmask[flip])
    return W.astype(np.float32), true_edge

W_GE, T_GE = sample_W(sGE, sD_GE)
W_SG, T_SG = sample_W(sGE, sD_GE)      # S→G 复用GErel拓扑（与训练模型一致）
W_EC, T_EC = sample_W(sEC, sD_EC, p=args.p_true_ec)
W_MM, T_MM = sample_W(sMM)
W_FB, T_FB = sample_W(sFB)
W_PS, T_PS = sample_W(sPS, p=args.p_true_ps)
beta_true = rng.uniform(0.3, 1.0, n_g).astype(np.float32)
decay_true = rng.uniform(0.02, 0.08, 4).astype(np.float32)   # 四层各一个衰减率
b_true = rng.normal(0, 0.1, 2 * n_g + n_g + n_m).astype(np.float32)
u_true = np.zeros(n_g, dtype=np.float32)
u_true[sig] = rng.normal(1.0, 0.3, sig.sum()).astype(np.float32)   # 胰岛素驱动信号层

n_true = sum(t.sum() for t in [T_GE, T_SG, T_EC, T_MM, T_FB, T_PS])
print(f"真值网络: {int(n_true)} 条真边")
# ---- 自检：每层真边数 / 候选数 / 平均每目标节点入度（v1 教训：PS 178入边/节点 vs 16方程 → 不可辨识）----
# 掩码约定: GE/SG/PS/MM 行=目标; EC/FB 为转置使用(EC 列=代谢物目标, FB 列=基因目标) → 入度按对应 axis 求和
for nm, T, M, ax in [("GE", T_GE, sGE, 1), ("SG", T_SG, sGE, 1), ("EC", T_EC, sEC, 0),
                     ("MM", T_MM, sMM, 1), ("FB", T_FB, sFB, 0), ("PS", T_PS, sPS, 1)]:
    indeg = T.sum(axis=ax)
    nz = indeg[indeg > 0]
    print(f"  {nm}: 真边 {int(T.sum()):>6} / 候选 {int((M > 0).sum()):>6} | 目标节点平均入度 {indeg.mean():.1f}"
          f"（有入边节点: {nz.mean():.1f}, 最大 {int(indeg.max())}）")

# ===================== 4. 仿真轨迹（与训练rhs同构 + RK4） =====================
dt, tmax = float(os.environ.get("HILL_DT", "0.5")), 240.0
n_step = int(tmax / dt)
t_input = 10.0

_HH = 2.0; _KK = 1.0
def hill(z):
    az = np.abs(z)
    return z * az**(_HH - 1.0) / (_KK**_HH + az**_HH)
def rhs(x, t, cond):
    G = x[:n_g]; P = x[n_g:2*n_g]; S = x[2*n_g:2*n_g+n_g]; M = x[2*n_g+n_g:]
    G_ = hill(G); P_ = hill(P); M_ = hill(M); S_ = hill(S)
    dG = W_GE @ G_ + W_SG @ S_ - decay_true[0] * G + b_true[:n_g]
    dP = beta_true * G - decay_true[1] * P + b_true[n_g:2*n_g]
    dS = W_PS @ P_ + W_FB.T @ M_ - decay_true[2] * S + b_true[2*n_g:2*n_g+n_g]
    if cond == 0 and t <= t_input:
        dS = dS + u_true
    dM = W_EC.T @ P_ + W_MM @ M_ - decay_true[3] * M + b_true[2*n_g+n_g:]
    return np.concatenate([dG, dP, dS, dM])

def rk4(x, t, h, cond):
    k1 = rhs(x, t, cond); k2 = rhs(x + .5*h*k1, t + .5*h, cond)
    k3 = rhs(x + .5*h*k2, t + .5*h, cond); k4 = rhs(x + h*k3, t + h, cond)
    return x + h * (k1 + 2*k2 + 2*k3 + k4) / 6

def simulate(cond):
    x = rng.normal(0, 0.5, 2 * n_g + n_g + n_m).astype(np.float32)  # 个体化初态
    traj = [x.copy()]
    t = 0.0
    for _ in range(n_step):
        x = rk4(x, t, dt, cond)
        traj.append(x.copy())
        t += dt
    _tr = np.stack(traj)
    if not np.isfinite(_tr).all():
        raise SystemExit('DIVERGED: non-finite trajectory (Hill arm)')
    return _tr

# ===================== 5. 按真实时间网格采样 + 噪声 + 缺失 =====================
GRIDS = {"G": [0, 60, 120, 180, 240],
         "P": [0, 60, 120, 180, 240],
         "S": [0, 7.5, 15, 30, 60, 120, 180, 240],
         "M": [0, 2.5, 5, 7.5, 10, 15, 20, 30, 45, 60, 120, 180, 240]}
MISS_RATE = {"G": 0.05, "P": 0.75, "S": 0.80, "M": 0.05}   # 仿真实覆盖率(蛋白/磷酸化大面积缺失)
SLICES = {"G": (0, n_g), "P": (n_g, 2*n_g), "S": (2*n_g, 3*n_g), "M": (3*n_g, 3*n_g+n_m)}

if args.ideal:
    # 理想数据臂:观测预算上限测试。网格加密到5min(49点/层,vs现实16方程/节点→98方程/节点)、
    # 零缺失、零噪声。rng消耗序列与常规臂不同(噪声形状随网格变),但真值网络抽样在前、不受影响。
    GRIDS = {k: [float(t) for t in range(0, 241, 5)] for k in GRIDS}
    MISS_RATE = {k: 0.0 for k in MISS_RATE}
    args.noise = 0.0
    print("【ideal模式】全层 49 点密集网格, 观测率 100%, 噪声 0")

out = {}
zstats = {}          # z分数统计量(v5 --oracle_zconj 用),单独存档不动旧文件
for key, grid in GRIDS.items():
    idx = [int(round(t / dt)) for t in grid]
    a, b = SLICES[key]
    n_feat = b - a
    # reps×2条件 取均值（与真实数据准备一致：重复→平均）
    vals = {0: [], 1: []}
    for cond in range(2):
        for r in range(args.reps):
            tr = simulate(cond)
            seg = tr[idx, a:b]                              # (T, feat)
            seg += rng.normal(0, args.noise, seg.shape).astype(np.float32)
            vals[cond].append(seg)
    X = np.full((n_feat, len(grid), 2), np.nan, dtype=np.float32)
    for cond in range(2):
        X[:, :, cond] = np.nanmean(np.stack(vals[cond]), axis=0).T
    # 注入缺失（按特征整体缺失，模拟"没测到这个分子"）
    miss = rng.random(n_feat) < MISS_RATE[key]
    X[miss] = np.nan
    # nan-aware z-score（与真实管线一致）
    mu = np.nanmean(X.reshape(n_feat, -1), axis=1, keepdims=True)
    sd = np.nanstd(X.reshape(n_feat, -1), axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    zstats[f"zmu_{key}"] = mu.reshape(-1).astype(np.float32)
    zstats[f"zsd_{key}"] = sd.reshape(-1).astype(np.float32)
    X = (X - mu.reshape(n_feat, 1, 1)) / sd.reshape(n_feat, 1, 1)
    O = (~np.isnan(X)).astype(bool)
    X = np.nan_to_num(X)
    out[f"X{key}"] = X
    out[f"t{key}"] = np.array(grid, dtype=np.float32)
    out[f"O{key}"] = O
    print(f"层{key}: {n_feat}节点 × {len(grid)}时间点 × 2条件, 观测率 {O.mean():.1%}")

# ===================== 6. 保存（npz键名与真实数据完全一致） =====================
np.savez_compressed(
    os.path.join(args.out_dir, "train_tensors.npz"),
    **out,
    conditions=np.array(["Insulin", "PBS"]),
    # 关键：必须存子网自身ID（691/33），不能存全节点表！
    # 训练脚本把 npz 的 gene_ids 当"紧凑索引全集"建 gpos，
    # 存全表(27999)会让子网索引越界、index_select 触发 CUDA 断言（已踩过）
    gene_ids=np.array(sub_g), metab_ids=np.array(sub_m),
)
# z分数统计量单独存档(v5 --oracle_zconj 用):不碰 train_tensors/ground_truth 一个字节——
# 同种子重跑后,用旧 best.pt 复评须逐位复现旧表(如 _sim2or01 的 0.838/0.859/...),作为数据未变的验收
np.savez_compressed(os.path.join(args.out_dir, "zstats.npz"), **zstats)
print(f"z分数统计量已存: {os.path.join(args.out_dir, 'zstats.npz')}")
np.savez_compressed(
    os.path.join(args.out_dir, "ground_truth.npz"),
    W_GE=W_GE, T_GE=T_GE, W_SG=W_SG, T_SG=T_SG, W_EC=W_EC, T_EC=T_EC,
    W_MM=W_MM, T_MM=T_MM, W_FB=W_FB, T_FB=T_FB, W_PS=W_PS, T_PS=T_PS,
    beta_true=beta_true, decay_true=decay_true,
    b_true=b_true, u_true=u_true,   # oracle 初始化测试需要(真值起手)
    # 候选掩码也存上：评估AUROC需要在"全部候选边"上打分，不是只在真边上
    MSK_GE=sGE, MSK_SG=sGE, MSK_EC=sEC, MSK_MM=sMM, MSK_FB=sFB, MSK_PS=sPS,
    sub_gene_ids=np.array(sub_g), sub_metab_ids=np.array(sub_m),
)
size_mb = os.path.getsize(os.path.join(args.out_dir, "train_tensors.npz")) / 1e6
print(f"\n已保存: {args.out_dir}/train_tensors.npz ({size_mb:.1f}MB) + ground_truth.npz")
print("下一步: python train_twin.py --scheme 1 --tensor_file data_sim/train_tensors.npz --tag _sim --data_dir", DATA_DIR)