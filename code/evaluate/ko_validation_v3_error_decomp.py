# -*- coding: utf-8 -*-
# 仿真KO验证v3: 误差分解 — 真值 vs 精确共轭臂(CONJ,无线性化外误差) vs 拟合臂(FIT)
# 三系统同删 Srebf1 真值出边; 双评估点(标定点=学习x0 / 远点=随机x0)
# CONJvs真值=近似天花板; FITvsCONJ=拟合误差; FITvs真值=总误差
import os, glob, json, time
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
ck_path = find_one(["**/scheme1_sim2or01/best.pt"], "ckpt")
arm = os.path.dirname(ck_path)
pvo = np.load(os.path.join(arm, "pred_vs_obs.npz"))

g_ids = [str(x).split(":")[-1] for x in gt["sub_gene_ids"]]
m_ids = [str(x) for x in gt["sub_metab_ids"]]
gpos = {v: i for i, v in enumerate(g_ids)}
i_sre = gpos["20787"]

sym = {}
try:
    cache = json.load(open(find_one(["**/kegg_api_cache.json"], "kegg缓存"), encoding="utf-8"))
    for k, v in cache.items():
        if "list/mmu" in k:
            txt = v if isinstance(v, str) else v.get("text", "")
            for line in txt.splitlines():
                parts = line.split("\t")
                if len(parts) >= 4 and parts[1] == "CDS":
                    desc = parts[3]
                elif len(parts) >= 2:
                    desc = parts[1]
                else:
                    continue
                s0 = desc.split(";")[0].split(",")[0].strip()
                if s0:
                    sym[parts[0].strip().split(":")[-1]] = s0
            break
except Exception as e:
    print(f"symbol反查跳过: {e}")
def nm(i):
    return f"{sym.get(g_ids[i], '?')}({g_ids[i]})"

T_GE = gt["T_GE"]
ko_pairs = [(int(r), i_sre) for r in np.nonzero(T_GE[:, i_sre])[0]]
if not ko_pairs:
    w_true = gt["W_GE"] * T_GE
    ko_pairs = [divmod(int(f), NG) for f in np.argsort(np.abs(w_true), axis=None)[::-1][:3]]
print("KO目标边:", [(nm(c), nm(r), round(float(gt["W_GE"][r, c]), 4)) for r, c in ko_pairs])
direct = sorted({r for r, c in ko_pairs})
hop2 = sorted({int(rr) for d in direct for rr in np.nonzero(T_GE[:, d])[0]} - set(direct))
print(f"hop分组: 直接靶标 {len(direct)} 个, 二跳 {len(hop2)} 个")

mu, sdv = {}, {}
for k in "GPSM":
    m = zs[f"zmu_{k}"].astype(np.float64); s = zs[f"zsd_{k}"].astype(np.float64)
    bad = ~np.isfinite(m) | ~np.isfinite(s) | (s < 1e-8)
    mu[k] = np.where(bad, 0.0, m); sdv[k] = np.where(bad, 1.0, s)
mu_cat = np.concatenate([mu["G"], mu["P"], mu["S"], mu["M"]])
sd_cat = np.concatenate([sdv["G"], sdv["P"], sdv["S"], sdv["M"]])

beta_t = gt["beta_true"].astype(np.float64); dec_t = gt["decay_true"].astype(np.float64)
b_t = gt["b_true"].astype(np.float64); u_t = gt["u_true"].astype(np.float64)
W0 = {k: gt[f"W_{k}"].astype(np.float64) for k in ["GE", "SG", "EC", "MM", "FB", "PS"]}

def rhs_gt(x, t, cond, W):
    G = x[:NG]; P = x[NG:2*NG]; S = x[2*NG:3*NG]; M = x[3*NG:]
    dG = W["GE"] @ np.tanh(G) + W["SG"] @ np.tanh(S) - dec_t[0]*G + b_t[:NG]
    dP = beta_t * G - dec_t[1]*P + b_t[NG:2*NG]
    dS = W["PS"] @ np.tanh(P) + W["FB"].T @ np.tanh(M) - dec_t[2]*S + b_t[2*NG:3*NG]
    if cond == 0 and t <= T_INPUT:
        dS = dS + u_t
    dM = W["EC"].T @ np.tanh(P) + W["MM"] @ np.tanh(M) - dec_t[3]*M + b_t[3*NG:]
    return np.concatenate([dG, dP, dS, dM])

th = {k: np.tanh(mu[k]) for k in "GPSM"}
sh2 = {k: 1.0 - th[k] ** 2 for k in "GPSM"}
def sp_inv(y):
    return np.log(np.expm1(np.clip(y, 1e-12, None)))
def conj_from(W):
    spb = np.log1p(np.exp(beta_t))
    return {
        "W_GE": W["GE"] * (sdv["G"] * sh2["G"])[None, :] / sdv["G"][:, None],
        "W_SG": W["SG"] * (sdv["S"] * sh2["S"])[None, :] / sdv["G"][:, None],
        "W_PS": W["PS"] * (sdv["P"] * sh2["P"])[None, :] / sdv["S"][:, None],
        "W_MM": W["MM"] * (sdv["M"] * sh2["M"])[None, :] / sdv["M"][:, None],
        "W_FB": W["FB"] * (sdv["M"] * sh2["M"])[:, None] / sdv["S"][None, :],
        "W_EC": W["EC"] * (sdv["P"] * sh2["P"])[:, None] / sdv["M"][None, :],
        "beta": sp_inv(spb * sdv["G"] / sdv["P"]),
        "d_raw": sp_inv(dec_t - 1e-3),
        "b_G": (b_t[:NG] + W["GE"] @ th["G"] + W["SG"] @ th["S"] - dec_t[0]*mu["G"]) / sdv["G"],
        "b_P": (b_t[NG:2*NG] + spb * mu["G"] - dec_t[1]*mu["P"]) / sdv["P"],
        "b_S": (b_t[2*NG:3*NG] + W["PS"] @ th["P"] + W["FB"].T @ th["M"] - dec_t[2]*mu["S"]) / sdv["S"],
        "b_M": (b_t[3*NG:] + W["EC"].T @ th["P"] + W["MM"] @ th["M"] - dec_t[3]*mu["M"]) / sdv["M"],
        "u": u_t / sdv["S"],
    }

ck = torch.load(ck_path, map_location="cpu")
sd_ck = ck if "W_GE" in ck else ck.get("model", ck.get("state_dict", ck))
KEYS = ["W_GE","W_SG","W_EC","W_MM","W_FB","W_PS","beta","b_G","b_P","b_S","b_M","d_raw","u","x0"]
FP = {k: sd_ck[k].detach().float() for k in KEYS}
MSK = {k: torch.tensor(gt[f"MSK_{k}"].astype(np.float32)) for k in ["GE", "EC", "MM", "FB", "PS"]}
sig_gate = (MSK["PS"].sum(dim=0) > 0).float()
def gene_mask(O):
    O = np.asarray(O)
    ax = [i for i, s in enumerate(O.shape) if s == NG][0]
    return np.any(O > 0, axis=tuple(i for i in range(O.ndim) if i != ax))
sBeta = torch.tensor((gene_mask(pvo["omask_G"]) | gene_mask(pvo["omask_P"])).astype(np.float32))
sft = torch.nn.functional.softplus

def rhs_z(x, t, cond, P_, W_GE_):
    G = x[:NG]; P = x[NG:2*NG]; S = x[2*NG:3*NG]; M = x[3*NG:]
    dec = sft(P_["d_raw"]) + 1e-3
    dG = (W_GE_ * MSK["GE"]) @ torch.tanh(G) + (P_["W_SG"] * MSK["GE"]) @ torch.tanh(S) - dec[0]*G + P_["b_G"]
    dP = sft(P_["beta"]) * sBeta * G - dec[1]*P + P_["b_P"]
    dS = (P_["W_PS"] * MSK["PS"]) @ torch.tanh(P) + (P_["W_FB"] * MSK["FB"]).t() @ torch.tanh(M) - dec[2]*S + P_["b_S"]
    if cond == 0 and t <= T_INPUT:
        dS = dS + P_["u"] * sig_gate
    dM = (P_["W_EC"] * MSK["EC"]).t() @ torch.tanh(P) + (P_["W_MM"] * MSK["MM"]) @ torch.tanh(M) - dec[3]*M + P_["b_M"]
    return torch.cat([dG, dP, dS, dM])

def rk4(f, x, t, h, cond):
    k1 = f(x, t, cond)
    k2 = f(x + 0.5*h*k1, t + 0.5*h, cond)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, cond)
    k4 = f(x + h*k3, t + h, cond)
    return x + h*(k1 + 2*k2 + 2*k3 + k4) / 6

def roll_z(P_, W_GE_, cond, x0_z):
    f = lambda a, b, c: rhs_z(a, b, c, P_, W_GE_)
    x = torch.tensor(np.asarray(x0_z), dtype=torch.float32)
    out = [x.numpy().astype(np.float64).copy()]; t = 0.0
    for _ in range(N_STEP):
        x = rk4(f, x, t, DT, cond); out.append(x.numpy().astype(np.float64).copy()); t += DT
    return np.stack(out)

def roll_true(W, cond, x0_raw):
    f = lambda a, b, c: rhs_gt(a, b, c, W)
    x = x0_raw.copy(); out = [x.copy()]; t = 0.0
    for _ in range(N_STEP):
        x = rk4(f, x, t, DT, cond); out.append(x.copy()); t += DT
    return np.stack(out)

def koW(W):
    Wk = {k: v.copy() for k, v in W.items()}
    for r, c in ko_pairs:
        Wk["GE"][r, c] = 0.0
    return Wk

W_ko = koW(W0)
CONJ_WT = {k: torch.tensor(v, dtype=torch.float32) for k, v in conj_from(W0).items()}
CONJ_KO = {k: torch.tensor(v, dtype=torch.float32) for k, v in conj_from(W_ko).items()}
FIT_KO_W = FP["W_GE"].clone()
for r, c in ko_pairs:
    FIT_KO_W[r, c] = 0.0

def per_species_r(Dm, Dt):
    rs = np.full(DIM, np.nan)
    for j in range(DIM):
        a, b = Dm[:, j], Dt[:, j]
        if a.std() > 1e-10 and b.std() > 1e-10:
            rs[j] = np.corrcoef(a, b)[0, 1]
    return rs

def report(tag, Dm, Dt):
    rs = per_species_r(Dm, Dt)
    mx_t = np.abs(Dt).max(axis=0); mx_m = np.abs(Dm).max(axis=0)
    resp = mx_t > 0.05
    if resp.sum() <= 2:
        print(f"  {tag}: 响应物种不足"); return rs, mx_t
    r_all = np.corrcoef(Dm[:, resp].ravel(), Dt[:, resp].ravel())[0, 1]
    sign = np.mean(np.sign(Dm[-1][resp]) == np.sign(Dt[-1][resp]))
    amp = np.median(mx_m[resp] / np.clip(mx_t[resp], 1e-12, None))
    print(f"  {tag}: 响应{int(resp.sum())} | 展平r={r_all:+.3f} | 物种r中位={np.nanmedian(rs[resp]):+.3f} | r≥0.8占{np.nanmean(rs[resp]>=0.8):.0%} | 终点符号{sign:.0%} | 幅度比中位{amp:.2f}")
    return rs, mx_t

x0_fresh = np.random.default_rng(2026).normal(0, 0.5, DIM)
rows = []
for sname, x0r in [("标定点", None), ("远点", x0_fresh)]:
    for cond, cname in [(0, "insulin"), (1, "pbs")]:
        x0_raw = (mu_cat + sd_cat * FP["x0"][cond].numpy().astype(np.float64)) if x0r is None else x0r
        x0_z = (x0_raw - mu_cat) / sd_cat
        Dt = (roll_true(W_ko, cond, x0_raw) - mu_cat) / sd_cat - (roll_true(W0, cond, x0_raw) - mu_cat) / sd_cat
        Dc = roll_z(CONJ_KO, CONJ_KO["W_GE"], cond, x0_z) - roll_z(CONJ_WT, CONJ_WT["W_GE"], cond, x0_z)
        Df = roll_z(FP, FIT_KO_W, cond, x0_z) - roll_z(FP, FP["W_GE"], cond, x0_z)
        print(f"\n== {sname} / {cname} ==")
        rs_c, mx_t = report("CONJvs真值(近似天花板)", Dc, Dt)
        rs_f, _ = report("FIT vs真值(总误差)  ", Df, Dt)
        if rs_c is None:
            continue
        for glab, idxs in [("直接靶", direct), ("二跳", hop2)]:
            if idxs:
                print(f"    {glab} G层r中位: CONJ={np.nanmedian(rs_c[idxs]):+.3f} | FIT={np.nanmedian(rs_f[idxs]):+.3f}")
        msl = slice(3*NG, DIM)
        mm = np.where(mx_t[msl] > 0.05)[0]
        if len(mm):
            print(f"    M层(响应{len(mm)}/33)r中位: CONJ={np.nanmedian(rs_c[msl][mm]):+.3f} | FIT={np.nanmedian(rs_f[msl][mm]):+.3f}")
        else:
            print("    M层无响应物种")
        for r, c in ko_pairs:
            print(f"    {nm(r)}: G CONJ={rs_c[r]:+.2f}/FIT={rs_f[r]:+.2f} | P CONJ={rs_c[NG+r]:+.2f}/FIT={rs_f[NG+r]:+.2f}")
        for j in range(DIM):
            ly = "G" if j < NG else "P" if j < 2*NG else "S" if j < 3*NG else "M"
            sid = g_ids[j] if j < NG else g_ids[j-NG] if j < 2*NG else g_ids[j-2*NG] if j < 3*NG else m_ids[j-3*NG]
            rows.append((sname, cname, ly, sid, rs_c[j], rs_f[j], mx_t[j]))

import csv
with open(os.path.join(WD, "仿真KO_v3_误差分解.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x0set", "cond", "layer", "species", "r_conj", "r_fit", "maxabs_d_true"])
    w.writerows(rows)
print(f"\n验证v3完成, 耗时 {(time.time()-t0)/60:.1f} min → 仿真KO_v3_误差分解.csv")
