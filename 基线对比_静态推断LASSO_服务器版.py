# -*- coding: utf-8 -*-
"""
基线对比：有限差分 + LASSO 静态边推断（paper 1 砖③：非 ODE 方法对照）
=========================================================================
用法：
    python 基线对比_静态推断LASSO_服务器版.py --sim_dir data_sim2
依赖：仅 numpy（服务器铁律：不装任何包）。

与掩码 ODE 完全同数据、同候选集、同评估口径的"静态基线"：
  1) 有限差分：相邻观测点 Δx/Δt（各层用自己的真实时间网格；跨层预测变量按观测点线性插值）；
  2) 逐目标节点累积采样：所有 (时间区间 × 双条件) 的 FD 样本累积后做一次 LASSO；
  3) LASSO = ISTA 软阈值迭代，λ 按 BIC 在 {0.03,0.1,0.3,1.0}×λ_max 中选择；
  4) 边分数=|w|，在 MSK_* 候选集上对 T_* 真边算 AUROC/AUPRC/Top-k（与 eval 脚本同口径）。
公平性声明：S 层回归加"脉冲指示"特征（区间中点≤t_input 且 Insulin 条件）以对待输入项 u(t)。
"""
import argparse, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--sim_dir", type=str, default="data_sim2")
p.add_argument("--t_input", type=float, default=10.0, help="胰岛素脉冲宽度(min)，与训练脚本一致")
p.add_argument("--ista_iters", type=int, default=300)
p.add_argument("--out", type=str, default="baseline_scores.npz")
ARGS = p.parse_args()

tt = np.load(os.path.join(ARGS.sim_dir, "train_tensors.npz"), allow_pickle=True)
gt = np.load(os.path.join(ARGS.sim_dir, "ground_truth.npz"), allow_pickle=True)

LAYERS = {k: dict(X=tt[f"X{k}"].astype(np.float64), t=tt[f"t{k}"].astype(np.float64),
                  O=tt[f"O{k}"].astype(bool)) for k in ["G", "P", "S", "M"]}
nG, nM = LAYERS["G"]["X"].shape[0], LAYERS["M"]["X"].shape[0]
print(f"子网: 基因 {nG}, 代谢物 {nM}")

# ---------- 1. 插值：层 L 在时刻 tq、条件 c 的特征向量 ----------
def interp_at(L, tq, c):
    X, t, O = LAYERS[L]["X"], LAYERS[L]["t"], LAYERS[L]["O"]
    v = np.zeros(X.shape[0])
    for j in range(X.shape[0]):
        o = O[j, :, c]
        if o.sum() >= 2:
            v[j] = np.interp(tq, t[o], X[j, o, c])
        elif o.sum() == 1:
            v[j] = X[j, o, c][0]
    return v

# ---------- 2. ISTA-LASSO（BIC 选 λ） ----------
def soft(w, th):
    return np.sign(w) * np.maximum(np.abs(w) - th, 0.0)

def lasso_bic(X, y, iters):
    n, p_ = X.shape
    Xty = X.T @ y / n
    lam_max = np.abs(Xty).max()
    if lam_max < 1e-12:
        return np.zeros(p_)
    v = np.ones(p_); v /= np.linalg.norm(v)          # 幂迭代估 L=λmax(XᵀX/n)
    for _ in range(30):
        v = X.T @ (X @ v) / n
        nv = np.linalg.norm(v)
        if nv < 1e-12: break
        v /= nv
    L = max(float(v @ (X.T @ (X @ v) / n)), 1e-8)
    best_w, best_bic = np.zeros(p_), np.inf
    for lam in [0.03, 0.1, 0.3, 1.0]:
        la = lam * lam_max
        w = np.zeros(p_)
        for _ in range(iters):
            w = soft(w + (X.T @ (y - X @ w)) / (n * L), la / L)
        rss = float(((y - X @ w) ** 2).sum())
        k = int((np.abs(w) > 1e-8).sum())
        bic = n * np.log(max(rss / n, 1e-12)) + k * np.log(n)
        if bic < best_bic:
            best_bic, best_w = bic, w
    return best_w

# ---------- 3. 逐目标累积采样回归 ----------
SRC_OF = {"GE": "G", "SG": "S", "PS": "P", "FB": "M", "EC": "P", "MM": "M"}

def fit_targets(tgt, specs, extra_pulse=False):
    """specs: [(名称, 掩码, transpose)]。掩码[i,j]=1 表 j→i 候选；
    transpose=True 表示掩码按 (源,目标) 存储（EC/FB）。返回 {名称: 分数矩阵(同掩码形)}"""
    Xt, tg, Ot = LAYERS[tgt]["X"], LAYERS[tgt]["t"], LAYERS[tgt]["O"]
    scores = {nm: np.zeros_like(mk, dtype=np.float64) for nm, mk, _ in specs}
    # 3a. 预计算全部 FD 样本（与目标无关的部分）
    samples = []
    for c in range(2):
        for a in range(len(tg) - 1):
            dt = tg[a + 1] - tg[a]
            if dt <= 0: continue
            t_mid = 0.5 * (tg[a] + tg[a + 1])
            sv = {sl: np.tanh(interp_at(sl, t_mid, c)) for sl in {SRC_OF[nm] for nm, _, _ in specs}}
            samples.append(dict(
                dy=(Xt[:, a + 1, c] - Xt[:, a, c]) / dt,
                valid=Ot[:, a, c] & Ot[:, a + 1, c],
                vt=interp_at(tgt, t_mid, c),          # 自身衰减项（线性）
                sv=sv,
                pulse=1.0 if (extra_pulse and c == 0 and t_mid <= ARGS.t_input) else 0.0,
            ))
    # 3b. 逐目标累积样本 → LASSO → 计分
    # 候选列统一用五元组 (矩阵名, 计分行, 计分列, 源层, 源索引)：
    #   tr=False（掩码[目标,源]存，GE/SG/PS/MM）：cand j 来自 mk[i,:]，计分 [i,j]，源值 sv[sl][j]
    #   tr=True （掩码[源,目标]存，EC/FB）     ：cand j 来自 mk[:,i]，计分 [j,i]，源值 sv[sl][j]
    for i in range(Xt.shape[0]):
        cand_cols = []
        for nm, mk, tr in specs:
            sl = SRC_OF[nm]
            cand = np.where(mk[:, i] > 0)[0] if tr else np.where(mk[i, :] > 0)[0]
            for j in cand:
                cand_cols.append((nm, j, i, sl, j) if tr else (nm, i, j, sl, j))
        if not cand_cols:
            continue
        rows, ys = [], []
        for s in samples:
            if not s["valid"][i]:
                continue
            row = [s["sv"][sl][sj] for (_, _, _, sl, sj) in cand_cols]
            row += [s["vt"][i], s["pulse"]]           # 自身衰减项 + 脉冲指示（均不计分）
            rows.append(row); ys.append(s["dy"][i])
        if len(rows) < 3:                             # 样本太少：跳过，分数留 0
            continue
        X = np.array(rows); y = np.array(ys)
        y = y - y.mean()
        mu, sd = X.mean(0), X.std(0)
        keep = sd > 1e-12
        Xs = (X[:, keep] - mu[keep]) / sd[keep]
        w = lasso_bic(Xs, y, ARGS.ista_iters)
        w_full = np.zeros(X.shape[1]); w_full[keep] = w / sd[keep]  # 折回原尺度
        for ci, (nm, sr, sc_, _, _) in enumerate(cand_cols):
            scores[nm][sr, sc_] = abs(w_full[ci])
    return scores

# ---------- 4. 评估（与 eval 脚本同口径：候选集上 AUROC/AUPRC/Top-k） ----------
def auroc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    ss = scores[order]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels > 0.5
    np_, nn_ = pos.sum(), (~pos).sum()
    if np_ == 0 or nn_ == 0: return float("nan")
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_))

def auprc(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order] > 0.5
    tp = np.cumsum(lab); prec = tp / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / lab.sum()) if lab.sum() else float("nan")

def topk(scores, labels):
    k = int((labels > 0.5).sum())
    if k == 0: return float("nan")
    top = np.argsort(-scores, kind="mergesort")[:k]
    return float((labels[top] > 0.5).mean())

print("\n矩阵    真边数   候选数   AUROC   AUPRC   Top-k命中")
print("-" * 60)
all_scores = {}
for tgt, specs, pulse in [("G", [("GE", gt["MSK_GE"], False), ("SG", gt["MSK_SG"], False)], False),
                          ("S", [("PS", gt["MSK_PS"], False), ("FB", gt["MSK_FB"], True)], True),
                          ("M", [("EC", gt["MSK_EC"], True), ("MM", gt["MSK_MM"], False)], False)]:
    sc = fit_targets(tgt, specs, extra_pulse=pulse)
    for nm, mk, _ in specs:
        T = gt[f"T_{nm}"]
        m = mk > 0
        s, l = sc[nm][m], T[m]
        all_scores[nm] = sc[nm]
        print(f"{nm:<6} 真边 {int((T>0).sum()):>6}  候选 {int(m.sum()):>6}  "
              f"AUROC {auroc(s,l):.3f}  AUPRC {auprc(s,l):.3f}  Top-k {topk(s,l):.1%}")

np.savez_compressed(os.path.join(ARGS.sim_dir, ARGS.out), **all_scores)
print(f"\n分数已存: {os.path.join(ARGS.sim_dir, ARGS.out)}")
