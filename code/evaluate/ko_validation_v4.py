# -*- coding: utf-8 -*-
# 仿真KO验证v4: 置信度校准 — ①系综一致性(or01+or03) ②敏感性定位命门边 ③校准统计
# 回答: 孪生何时可信? 脆弱结论追到哪个边? → 「给答案+给置信+给实验单」
import os, glob, time
import numpy as np
import torch

torch.set_num_threads(8)
NG, NM, DIM = 691, 33, 2106
DT, TMAX, T_INPUT = 0.5, 240.0, 10.0
N_STEP = int(round(TMAX / DT))
WD = "/dfs/data/twin-ode"
t0 = time.time()

def find_one(patterns, desc, required=True):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(WD, pat), recursive=True))
        if hits:
            print(f"{desc}: {os.path.relpath(hits[0], WD)}")
            return hits[0]
    if required:
        raise FileNotFoundError(desc)
    print(f"{desc}: 未找到, 跳过对应分析")
    return None

gt = np.load(find_one(["data_sim2/ground_truth.npz"], "真值"), allow_pickle=True)
zs = np.load(find_one(["data_sim2/zstats.npz"], "zstats"))
ck1_path = find_one(["**/scheme1_sim2or01/best.pt"], "FIT1(or01)")
ck2_path = find_one(["**/scheme1_sim2or03/best.pt", "**/*or03*/best.pt"], "FIT2(or03)", required=False)
pvo = np.load(os.path.join(os.path.dirname(ck1_path), "pred_vs_obs.npz"))

g_ids = [str(x).split(":")[-1] for x in gt["sub_gene_ids"]]
m_ids = [str(x) for x in gt["sub_metab_ids"]]
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

def load_fit(path):
    ck = torch.load(path, map_location="cpu")
    sd_ck = ck if "W_GE" in ck else ck.get("model", ck.get("state_dict", ck))
    KEYS = ["W_GE","W_SG","W_EC","W_MM","W_FB","W_PS","beta","b_G","b_P","b_S","b_M","d_raw","u","x0"]
    return {k: sd_ck[k].detach().float() for k in KEYS}

FP1 = load_fit(ck1_path)
FP2 = load_fit(ck2_path) if ck2_path else None
MSK = {k: torch.tensor(gt[f"MSK_{k}"].astype(np.float32)) for k in ["GE", "EC", "MM", "FB", "PS"]}
sig_gate = (MSK["PS"].sum(dim=0) > 0).float()
def gene_mask(O):
    O = np.asarray(O)
    ax = [i for i, s in enumerate(O.shape) if s == NG][0]
    return np.any(O > 0, axis=tuple(i for i in range(O.ndim) if i != ax))
sBeta = torch.tensor((gene_mask(pvo["omask_G"]) | gene_mask(pvo["omask_P"])).astype(np.float32))
sft = torch.nn.functional.softplus

# ---- Stat3 双路径平衡诊断(真值 vs 各拟合) ----
print("\n== Stat3 双路径平衡(决定净效应符号) ==")
print(f"  真值: Sre→Stat3 {W0['GE'][i_stat3, i_sre]:+.4f} | Sre→p53 {W0['GE'][i_p53, i_sre]:+.4f} | p53→Stat3 {W0['GE'][i_stat3, i_p53]:+.4f}")
for tag, FP in [("FIT1", FP1), ("FIT2", FP2)]:
    if FP is None: continue
    We = FP["W_GE"] * MSK["GE"]
    print(f"  {tag}: Sre→Stat3 {We[i_stat3, i_sre].item():+.4f} | Sre→p53 {We[i_p53, i_sre].item():+.4f} | p53→Stat3 {We[i_stat3, i_p53].item():+.4f}")

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
    k1 = f(x, t, cond); k2 = f(x + 0.5*h*k1, t + 0.5*h, cond)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, cond); k4 = f(x + h*k3, t + h, cond)
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

def fit_delta(FP, cond, x0_z, WGE_override=None):
    Wwt = FP["W_GE"] if WGE_override is None else WGE_override
    Wko = Wwt.clone()
    for r, c in ko_pairs:
        Wko[r, c] = 0.0
    return roll_z(FP, Wko, cond, x0_z) - roll_z(FP, Wwt, cond, x0_z)

# ---- 主评估(远点, N大才有统计) ----
x0_raw = np.random.default_rng(2026).normal(0, 0.5, DIM)
x0_z = (x0_raw - mu_cat) / sd_cat
motif = [(i_p53, i_sre), (i_stat3, i_sre), (i_stat3, i_p53)]
all_rows = []
for cond, cname in [(0, "insulin"), (1, "pbs")]:
    Dt = (roll_true(koW_global := {k: v.copy() for k, v in W0.items()}, cond, x0_raw) - mu_cat) / sd_cat
    for r, c in ko_pairs:
        koW_global["GE"][r, c] = 0.0
    Dt = (roll_true(koW_global, cond, x0_raw) - mu_cat) / sd_cat - Dt
    D1 = fit_delta(FP1, cond, x0_z)
    D2 = fit_delta(FP2, cond, x0_z) if FP2 is not None else None
    mx_t = np.abs(Dt).max(axis=0)
    resp = np.where(mx_t > 0.05)[0]
    s_true = np.sign(Dt[-1]); s1 = np.sign(D1[-1])
    correct1 = (s1 == s_true)
    print(f"\n== 远点 / {cname} ==  响应物种 {len(resp)}")
    # ① 系综校准
    if D2 is not None:
        s2 = np.sign(D2[-1])
        agree = (s1 == s2)
        a, d = resp[agree[resp]], resp[~agree[resp]]
        pa = correct1[a].mean() if len(a) else float("nan")
        pd = correct1[d].mean() if len(d) else float("nan")
        print(f"  ①系综校准: 两孪生一致 n={len(a)}, 一致时FIT1正确率={pa:.0%} | 打架 n={len(d)}, 打架时正确率={pd:.0%}")
        print(f"    p53端点符号: 真值{s_true[i_p53]:+.0f} FIT1 {s1[i_p53]:+.0f} FIT2 {s2[i_p53]:+.0f} | Stat3: 真值{s_true[i_stat3]:+.0f} FIT1 {s1[i_stat3]:+.0f} FIT2 {s2[i_stat3]:+.0f}")
    # ② 敏感性: motif 三边各 ±20%
    base_sign = s1
    robust = np.ones(DIM, dtype=bool)
    for (er, ec) in motif:
        for fac in [0.8, 1.2]:
            Wv = FP1["W_GE"].clone()
            Wv[er, ec] = Wv[er, ec] * fac
            Dv = fit_delta(FP1, cond, x0_z, WGE_override=Wv)
            robust &= (np.sign(Dv[-1]) == base_sign)
    rb, fr = resp[robust[resp]], resp[~robust[resp]]
    pr = correct1[rb].mean() if len(rb) else float("nan")
    pf = correct1[fr].mean() if len(fr) else float("nan")
    print(f"  ②敏感性校准: 稳健 n={len(rb)}, 稳健时正确率={pr:.0%} | 脆弱 n={len(fr)}, 脆弱时正确率={pf:.0%}")
    print(f"    p53: {'稳健' if robust[i_p53] else '脆弱'}({'对' if correct1[i_p53] else '错'}) | Stat3: {'稳健' if robust[i_stat3] else '脆弱'}({'对' if correct1[i_stat3] else '错'})")
    for j in resp:
        ly = "G" if j < NG else "P" if j < 2*NG else "S" if j < 3*NG else "M"
        sid = g_ids[j] if j < NG else g_ids[j-NG] if j < 2*NG else g_ids[j-2*NG] if j < 3*NG else m_ids[j-3*NG]
        all_rows.append((cname, ly, sid, Dt[-1, j], D1[-1, j], "" if D2 is None else D2[-1, j],
                         bool(correct1[j]), bool(robust[j])))

import csv
with open(os.path.join(WD, "仿真KO_v4_置信校准.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cond", "layer", "species", "endpoint_d_true", "endpoint_d_fit1", "endpoint_d_fit2", "fit1_correct", "robust"])
    w.writerows(all_rows)
print(f"\n验证v4完成, 耗时 {(time.time()-t0)/60:.1f} min → 仿真KO_v4_置信校准.csv")
