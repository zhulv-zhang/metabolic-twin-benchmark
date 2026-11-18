# -*- coding: utf-8 -*-
# 仿真KO验证v8: ①钉边修正(原生尺度: 拟合权重≈raw真值量级, 直接钉raw值)
#               ②排序可信度(Spearman |Δ| 与 top-K召回) — 「哪些分子动得最多」的优先级是否可信
import os, glob, time, csv
import numpy as np
import torch

torch.set_num_threads(8)
NG, NM, DIM = 691, 33, 2106
DT, TMAX, T_INPUT = 0.5, 240.0, 10.0
N_STEP = int(round(TMAX / DT))
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
ck_path = find_one(["**/scheme1_sim2or01/best.pt"], "FIT1")
pvo = np.load(os.path.join(os.path.dirname(ck_path), "pred_vs_obs.npz"))

g_ids = [str(x).split(":")[-1] for x in gt["sub_gene_ids"]]
gpos = {v: i for i, v in enumerate(g_ids)}
i_sre, i_p53, i_stat3 = gpos["20787"], gpos["22059"], gpos["20848"]
ko_pairs = [(i_p53, i_sre), (i_stat3, i_sre)]

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

def rhs_gt(x, t, W):
    G = x[:NG]; P = x[NG:2*NG]; S = x[2*NG:3*NG]; M = x[3*NG:]
    dG = W["GE"] @ np.tanh(G) + W["SG"] @ np.tanh(S) - dec_t[0]*G + b_t[:NG]
    dP = beta_t * G - dec_t[1]*P + b_t[NG:2*NG]
    dS = W["PS"] @ np.tanh(P) + W["FB"].T @ np.tanh(M) - dec_t[2]*S + b_t[2*NG:3*NG]
    if t <= T_INPUT:
        dS = dS + u_t
    dM = W["EC"].T @ np.tanh(P) + W["MM"] @ np.tanh(M) - dec_t[3]*M + b_t[3*NG:]
    return np.concatenate([dG, dP, dS, dM])

ck = torch.load(ck_path, map_location="cpu")
sd_ck = ck if "W_GE" in ck else ck.get("model", ck.get("state_dict", ck))
KEYS = ["W_GE","W_SG","W_EC","W_MM","W_FB","W_PS","beta","b_G","b_P","b_S","b_M","d_raw","u","x0"]
FP = {k: sd_ck[k].detach().float() for k in KEYS}
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

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return np.corrcoef(ra, rb)[0, 1]

E_DS, E_IS, E_IP = (i_stat3, i_sre), (i_p53, i_sre), (i_stat3, i_p53)
PATCH_SETS = [("0_现状", []), ("1_钉Sre→Stat3", [E_DS]),
              ("2_钉Stat3两入边", [E_DS, E_IP]), ("3_三边全钉(raw尺度)", [E_DS, E_IP, E_IS])]

rows = []
for sname, x0_raw in [("标定点", mu_cat + sd_cat * FP["x0"][0].numpy().astype(np.float64)),
                      ("远点", np.random.default_rng(2026).normal(0, 0.5, DIM))]:
    x0_z = (x0_raw - mu_cat) / sd_cat
    Wk_true = {k: v.copy() for k, v in W0.items()}
    for r, c in ko_pairs:
        Wk_true["GE"][r, c] = 0.0
    Dt = (roll_true(Wk_true, x0_raw) - mu_cat) / sd_cat - (roll_true(W0, x0_raw) - mu_cat) / sd_cat
    mx_t = np.abs(Dt).max(axis=0)
    resp = np.where(mx_t > 0.05)[0]
    top20_t = set(np.argsort(-mx_t)[:20]); top50_t = set(np.argsort(-mx_t)[:50])
    print(f"\n== {sname} (insulin, 响应{len(resp)}) ==")
    for pname, edges in PATCH_SETS:
        Wp = FP["W_GE"].clone()
        for r, c in edges:
            Wp[r, c] = float(W0["GE"][r, c])   # 原生尺度钉边
        Wko = Wp.clone()
        for r, c in ko_pairs:
            Wko[r, c] = 0.0
        Df = roll_fit(Wko, x0_z) - roll_fit(Wp, x0_z)
        mx_f = np.abs(Df).max(axis=0)
        sign = np.mean(np.sign(Df[-1][resp]) == np.sign(Dt[-1][resp]))
        rs = [np.corrcoef(Df[:, j], Dt[:, j])[0, 1] for j in resp
              if Df[:, j].std() > 1e-10 and Dt[:, j].std() > 1e-10]
        sp = spearman(mx_f[resp], mx_t[resp])
        r20 = len(set(np.argsort(-mx_f)[:20]) & top20_t) / 20
        r50 = len(set(np.argsort(-mx_f)[:50]) & top50_t) / 50
        def tg(tag, i):
            if Df[:, i].std() < 1e-10 or Dt[:, i].std() < 1e-10:
                return f"{tag}: r=nan"
            rr = np.corrcoef(Df[:, i], Dt[:, i])[0, 1]
            ok = "对" if np.sign(Df[-1, i]) == np.sign(Dt[-1, i]) else "错"
            return f"{tag} r={rr:+.2f}/{ok}"
        print(f"  {pname}: 符号{sign:.0%} r中位{np.nanmedian(rs):+.2f} | 排序ρ={sp:+.2f} top20召回{r20:.0%} top50召回{r50:.0%} | {tg('p53', i_p53)} | {tg('Stat3', i_stat3)}")
        rows.append((sname, pname, sign, float(np.nanmedian(rs)), sp, r20, r50))

with open(os.path.join(WD, "仿真KO_v8_排序与钉边.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x0set", "patch_set", "sign_acc", "r_median", "spearman_rank", "top20_recall", "top50_recall"])
    w.writerows(rows)
print(f"\n验证v8完成, 耗时 {(time.time()-t0)/60:.1f} min → 仿真KO_v8_排序与钉边.csv")
