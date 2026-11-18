# -*- coding: utf-8 -*-
# 仿真KO验证v6: 多KO池化 — 12个源基因各做全出边KO, 池化响应物种,
# 检验: ①结构冲突标记的错误率分离(带z检验) ②保真随跳数/时间衰减
import os, glob, time, csv, math
import numpy as np
import torch

torch.set_num_threads(8)
NG, NM, DIM = 691, 33, 2106
DT, TMAX, T_INPUT = 0.5, 240.0, 10.0
N_STEP = int(round(TMAX / DT))
EARLY = 60   # t=30min
WD = "/dfs/data/twin-ode"
t0 = time.time()

def find_one(patterns, desc):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(WD, pat), recursive=True))
        if hits:
            print(f"{desc}: {os.path.relpath(hits[0], WD)}")
            return hits[0]
    raise FileNotFoundError(desc)

gt = np.load(find_one(["data_sim2/ground_truth.npz"], "真值"), allow_pickle=True)
zs = np.load(find_one(["data_sim2/zstats.npz"], "zstats"))
ck1 = torch.load(find_one(["**/scheme1_sim2or01/best.pt"], "FIT1"), map_location="cpu")
sd_ck = ck1 if "W_GE" in ck1 else ck1.get("model", ck1.get("state_dict", ck1))
pvo = np.load(os.path.join(os.path.dirname(find_one(["**/scheme1_sim2or01/best.pt"], "_")), "pred_vs_obs.npz"))

g_ids = [str(x).split(":")[-1] for x in gt["sub_gene_ids"]]
KEYS = ["W_GE","W_SG","W_EC","W_MM","W_FB","W_PS","beta","b_G","b_P","b_S","b_M","d_raw","u","x0"]
FP = {k: sd_ck[k].detach().float() for k in KEYS}

mu, sdv = {}, {}
for k in "GPSM":
    m = zs[f"zmu_{k}"].astype(np.float64); s = zs[f"zsd_{k}"].astype(np.float64)
    bad = ~np.isfinite(m) | ~np.isfinite(s) | (s < 1e-8)
    mu[k] = np.where(bad, 0.0, m); sdv[k] = np.where(bad, 1.0, s)
mu_cat = np.concatenate([mu["G"], mu["P"], mu["S"], mu["M"]])
sd_cat = np.concatenate([sdv["G"], sdv["P"], sdv["S"], sdv["M"]])

beta_t = gt["beta_true"].astype(np.float64); dec_t = gt["decay_true"].astype(np.float64)
b_t = gt["b_true"].astype(np.float64); u_t = gt["u_true"].astype(np.float64)
W0 = {k: gt[f"W_{k}"].astype(np.float64) for k in ["GE","SG","EC","MM","FB","PS"]}
T_GE = gt["T_GE"]

# 源基因: 真值出边≥2, 按 sum|w| 取 top12
cand = [s for s in range(NG) if T_GE[:, s].sum() >= 2]
cand.sort(key=lambda s: -np.abs(W0["GE"][T_GE[:, s] > 0, s]).sum())
sources = cand[:12]
print(f"KO源基因 {len(sources)} 个: {sources}")

def rhs_gt(x, t, W):
    G = x[:NG]; P = x[NG:2*NG]; S = x[2*NG:3*NG]; M = x[3*NG:]
    dG = W["GE"] @ np.tanh(G) + W["SG"] @ np.tanh(S) - dec_t[0]*G + b_t[:NG]
    dP = beta_t * G - dec_t[1]*P + b_t[NG:2*NG]
    dS = W["PS"] @ np.tanh(P) + W["FB"].T @ np.tanh(M) - dec_t[2]*S + b_t[2*NG:3*NG]
    if t <= T_INPUT:
        dS = dS + u_t
    dM = W["EC"].T @ np.tanh(P) + W["MM"] @ np.tanh(M) - dec_t[3]*M + b_t[3*NG:]
    return np.concatenate([dG, dP, dS, dM])

MSK = {k: torch.tensor(gt[f"MSK_{k}"].astype(np.float32)) for k in ["GE","EC","MM","FB","PS"]}
sig_gate = (MSK["PS"].sum(dim=0) > 0).float()
def gene_mask(O):
    O = np.asarray(O)
    ax = [i for i, s in enumerate(O.shape) if s == NG][0]
    return np.any(O > 0, axis=tuple(i for i in range(O.ndim) if i != ax))
sBeta = torch.tensor((gene_mask(pvo["omask_G"]) | gene_mask(pvo["omask_P"])).astype(np.float32))
sft = torch.nn.functional.softplus

def rhs_z(x, t, W_GE_):
    G = x[:NG]; P = x[NG:2*NG]; S = x[2*NG:3*NG]; M = x[3*NG:]
    dec = sft(FP["d_raw"]) + 1e-3
    dG = (W_GE_ * MSK["GE"]) @ torch.tanh(G) + (FP["W_SG"] * MSK["GE"]) @ torch.tanh(S) - dec[0]*G + FP["b_G"]
    dP = sft(FP["beta"]) * sBeta * G - dec[1]*P + FP["b_P"]
    dS = (FP["W_PS"] * MSK["PS"]) @ torch.tanh(P) + (FP["W_FB"] * MSK["FB"]).t() @ torch.tanh(M) - dec[2]*S + FP["b_S"]
    if t <= T_INPUT:
        dS = dS + FP["u"] * sig_gate
    dM = (FP["W_EC"] * MSK["EC"]).t() @ torch.tanh(P) + (FP["W_MM"] * MSK["MM"]) @ torch.tanh(M) - dec[3]*M + FP["b_M"]
    return torch.cat([dG, dP, dS, dM])

def rk4(f, x, t, h):
    k1 = f(x, t); k2 = f(x + 0.5*h*k1, t + 0.5*h)
    k3 = f(x + 0.5*h*k2, t + 0.5*h); k4 = f(x + h*k3, t + h)
    return x + h*(k1 + 2*k2 + 2*k3 + k4) / 6

def roll_true(W, x0):
    x = x0.copy(); out = [x.copy()]; t = 0.0
    for _ in range(N_STEP):
        x = rk4(lambda a, b: rhs_gt(a, b, W), x, t, DT); out.append(x.copy()); t += DT
    return np.stack(out)

def roll_fit(W_GE_, x0_z):
    x = torch.tensor(x0_z, dtype=torch.float32)
    out = [x.numpy().astype(np.float64).copy()]; t = 0.0
    for _ in range(N_STEP):
        x = rk4(lambda a, b: rhs_z(a, b, W_GE_), x, t, DT); out.append(x.numpy().astype(np.float64).copy()); t += DT
    return np.stack(out)

# 标定点(insulin), WT 只积一次
x0_z = FP["x0"][0].numpy().astype(np.float64)
x0_raw = mu_cat + sd_cat * x0_z
TRU_WT = (roll_true(W0, x0_raw) - mu_cat) / sd_cat
FIT_WT = roll_fit(FP["W_GE"], x0_z)
print("WT 轨迹完成")

# 拟合网络结构(标记住: 不需要真值)
W_eff = (FP["W_GE"] * MSK["GE"]).numpy()
Snet = np.sign(W_eff); Snet[np.abs(W_eff) < 0.05] = 0.0

rows = []
for s in sources:
    out_edges = np.nonzero(T_GE[:, s])[0]
    Wk = {k: v.copy() for k, v in W0.items()}
    Wk["GE"][out_edges, s] = 0.0
    Wkf = FP["W_GE"].clone()
    Wkf[out_edges, s] = 0.0
    Dt = (roll_true(Wk, x0_raw) - mu_cat) / sd_cat - TRU_WT
    Df = roll_fit(Wkf, x0_z) - FIT_WT
    direct_f = np.nonzero(Snet[:, s])[0]
    for j in range(NG):
        if np.abs(Dt[:, j]).max() <= 0.05:
            continue
        signs = []
        if Snet[j, s] != 0:
            signs.append(Snet[j, s])
        for k in direct_f:
            if Snet[j, k] != 0:
                signs.append(Snet[j, k] * Snet[k, s])
        hop = 1 if Snet[j, s] != 0 else (2 if len(signs) > 0 else 0)
        conf = len(signs) >= 2 and min(signs) < 0 < max(signs)
        if Dt[:, j].std() > 1e-10 and Df[:, j].std() > 1e-10:
            r = np.corrcoef(Df[:, j], Dt[:, j])[0, 1]
        else:
            r = np.nan
        dir_true = float(np.sign(W0["GE"][j, s])) if W0["GE"][j, s] != 0 else 0.0
        rows.append((g_ids[s], g_ids[j], hop, int(conf),
                     int(np.sign(Df[-1, j]) == np.sign(Dt[-1, j])),
                     int(np.sign(Df[EARLY, j]) == np.sign(Dt[EARLY, j])),
                     r, float(np.abs(Dt[:, j]).max()),
                     dir_true, float(np.sign(Dt[-1, j])), float(np.sign(Df[-1, j]))))

print(f"池化响应实例(G层): {len(rows)}")
def grp(rows, pred):
    sub = [r for r in rows if pred(r)]
    if not sub:
        return "n=0"
    ep = np.mean([r[4] for r in sub]); ea = np.mean([r[5] for r in sub])
    rs = np.nanmedian([r[6] for r in sub])
    return f"n={len(sub):3d} | 终点符号 {ep:.0%} | 早期(30min)符号 {ea:.0%} | 轨迹r中位 {rs:+.2f}"
print("\n== 池化分组(拟合网络结构标记) ==")
print("  单路径1跳:      " + grp(rows, lambda r: r[2] == 1 and not r[3]))
print("  单路径2跳:      " + grp(rows, lambda r: r[2] == 2 and not r[3]))
print("  对消节点(冲突): " + grp(rows, lambda r: r[3] == 1))
print("  GE不可达但响应: " + grp(rows, lambda r: r[2] == 0))
a = [r for r in rows if r[2] >= 1 and not r[3]]
b = [r for r in rows if r[3] == 1]
if a and b:
    p1, p2 = np.mean([r[4] for r in a]), np.mean([r[4] for r in b])
    p = (sum(r[4] for r in a) + sum(r[4] for r in b)) / (len(a) + len(b))
    z = (p1 - p2) / math.sqrt(max(p * (1 - p) * (1 / len(a) + 1 / len(b)), 1e-12))
    print(f"\n  单路径({p1:.0%}, n={len(a)}) vs 冲突({p2:.0%}, n={len(b)}): z={z:+.2f}, 双侧p≈{math.erfc(abs(z)/math.sqrt(2)):.4f}")

with open(os.path.join(WD, "仿真KO_v6_多KO池化.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "species", "hop", "conflict", "correct_endpoint", "correct_early", "r_traj", "maxabs_d_true", "direct_true_sign", "true_endpoint_sign", "fit_endpoint_sign"])
    w.writerows(rows)
print(f"\n验证v6完成, 耗时 {(time.time()-t0)/60:.1f} min → 仿真KO_v6_多KO池化.csv")
