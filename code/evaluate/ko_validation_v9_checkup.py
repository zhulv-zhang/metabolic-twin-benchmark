# -*- coding: utf-8 -*-
# ============================================================
# 仿真KO_验证v9_体检.py —— 三模块体检（A-32 尺度体检 + A-36 证据分级 + hub偏差标注）
# 服务器: /dfs/data/twin-ode  解释器: /dfs/data/envs/twin-ode/bin/python
# 运行: cd /dfs/data/twin-ode && /dfs/data/envs/twin-ode/bin/python 仿真KO_验证v9_体检.py
# 纪律: NaN 不进结论; 缺数据打印跳过不崩; 不写根目录
# ============================================================
import os, json
import numpy as np

D = "."
GT_NPZ   = "data_sim2/ground_truth.npz"
ZS_NPZ   = "data_sim2/zstats.npz"
TRAJ_NPZ = "仿真KO_轨迹.npz"          # v2+ 存档的KO/WT轨迹
FIT1_PT  = "data/scheme1_sim2or01/best.pt"
EDGES_DIR = "GSE166336_小鼠组学数据"   # 掩码边表csv所在
OUT_CSV  = "仿真KO_v9_体检_33代谢点_raw尺度.csv"

def load_or_skip(path, kind="npz"):
    if not os.path.exists(path):
        print(f"[跳过] 找不到 {path}")
        return None
    return np.load(path, allow_pickle=True) if kind == "npz" else path

# ========== 模块0: 输入发现 ==========
print("=" * 60); print("v9 体检 · 模块0 输入发现"); print("=" * 60)
gt = load_or_skip(GT_NPZ); zs = load_or_skip(ZS_NPZ)
if gt: print("ground_truth keys:", list(gt.keys()))
if zs: print("zstats keys:", list(zs.keys()))

# ========== 模块1(A-32): NaN 硬规则名单 + 33代谢点 raw 尺度体检 ==========
print("\n" + "=" * 60); print("模块1: NaN硬规则 + 33代谢点 raw 尺度"); print("=" * 60)
nan_manifest = {}
if zs:
    for L in "GPSM":
        sd = np.asarray(zs[f"zsd_{L}"], dtype=np.float64)
        idx = np.where(~np.isfinite(sd))[0]
        nan_manifest[L] = idx
        print(f"层{L}: NaN sd 特征 {len(idx)} 个 → 索引 {idx.tolist()[:40]}{'...' if len(idx)>40 else ''}（硬排除，不进任何结论）")
else:
    print("[跳过] 无 zstats，模块1中止")

# 33代谢点: Δraw = Δz × zsd_M; 基线水平比 = max|Δraw| / |zmu_M|（代理口径:
# 轨迹npz只存KO−WT差分, 拿不到WT自身极差; 分母=该代谢物观测基线|μ|, 论文标注proxy）
if zs and os.path.exists(TRAJ_NPZ):
    tr = np.load(TRAJ_NPZ, allow_pickle=True)
    print("轨迹npz keys:", list(tr.keys()))
    if "ko_pairs" in tr: print("KO对象:", tr["ko_pairs"])
    metab_ids = [str(x) for x in tr["metab_ids"]] if "metab_ids" in tr else None
    dsides = {c: {s: tr[f"D_{s}_{c}"] for s in ("true", "fit") if f"D_{s}_{c}" in tr}
              for c in ("ins", "pbs")}
    if metab_ids is not None and all(dsides[c] for c in dsides):
        sd_M = np.asarray(zs["zsd_M"], dtype=np.float64)
        mu_M = np.asarray(zs["zmu_M"], dtype=np.float64)
        nM = len(metab_ids)
        rows = []
        for c in ("ins", "pbs"):
            for s in ("true", "fit"):
                if s not in dsides[c]: continue
                DM = dsides[c][s][:, -nM:]
                for j, mid in enumerate(metab_ids):
                    sd, mu = sd_M[j], mu_M[j]
                    if not np.isfinite(sd):
                        rows.append((mid, c, s, "NaN_sd", "", "", "EXCLUDED")); continue
                    dz = DM[:, j]; draw = dz * sd
                    pct = 100 * np.abs(draw).max() / abs(mu) if np.isfinite(mu) and abs(mu) > 1e-12 else np.nan
                    rows.append((mid, c, s, f"{sd:.4g}", f"{np.abs(dz).max():.4f}",
                                 f"{np.abs(draw).max():.4g}", f"{pct:.1f}%"))
        with open(OUT_CSV, "w", encoding="utf-8") as f:
            f.write("metab_id,cond,side,zsd,max_abs_dz,max_abs_draw,pct_of_baseline(proxy)\n")
            for r in rows: f.write(",".join(map(str, r)) + "\n")
        print(f"已写 {OUT_CSV}（{len(rows)} 行；EXCLUDED=NaN sd 硬排除；pct 分母=|基线μ| 代理口径）")
        # 只打印 true 侧摘要, fit 侧查 CSV
        for r in rows:
            if r[2] == "true": print("  ", r)
    else:
        print("[跳过] 轨迹键仍不识别——请把上面的 keys 贴回执行侧")
else:
    print("[跳过] 无轨迹npz；33点体检待轨迹文件")

# ========== 模块2(A-36): 先验证据分级 × 边恢复率 ==========
print("\n" + "=" * 60); print("模块2: 先验证据分级 × FIT1 边恢复"); print("=" * 60)
if gt and os.path.exists(FIT1_PT):
    import torch
    ck = torch.load(FIT1_PT, map_location="cpu")
    Wfit = ck["W_GE"].numpy() if "W_GE" in ck else None
    T_GE = gt["T_GE"]; W0_GE = gt["W_GE"]
    Msk  = gt["MSK_GE"] if "MSK_GE" in gt else np.ones_like(T_GE)
    if Wfit is not None:
        from numpy import ndarray
        def auroc(scores, labels):
            o = np.argsort(-scores); lab = labels[o]
            tp = np.cumsum(lab); fp = np.cumsum(1 - lab)
            if tp[-1] == 0 or fp[-1] == 0: return np.nan
            tpr = tp / tp[-1]; fpr = fp / fp[-1]
            return float(np.trapz(tpr, fpr))
        cand = Msk > 0
        truth = (T_GE > 0)[cand]
        sc_all = np.abs(Wfit)[cand]
        print(f"GE 候选边 {cand.sum()}，真边 {truth.sum()}")
        # 证据分级: GE边表csv（含PMID=TRRUST文献级；无PMID=KGML级）
        import glob
        cands = glob.glob(os.path.join("**", "mask_GErel_edges.csv"), recursive=True)
        ge_csv = cands[0] if cands else None
        print("GE边表定位:", ge_csv)
        sub_ids = list(gt["sub_gene_ids"])
        if ge_csv:
            import csv
            pmid_edges, kgml_edges = set(), set()
            with open(ge_csv, encoding="utf-8") as f:
                rd = csv.reader(f); head = next(rd)
                print("GE边表表头:", head)
                # 防御解析: 找 src/dst/pmid 列
                low = [h.lower() for h in head]
                si = low.index(next(h for h in low if "src" in h or "source" in h or "from" in h))
                di = low.index(next(h for h in low if "dst" in h or "target" in h or "to" in h))
                pi = next((k for k, h in enumerate(low) if "pmid" in h), None)
                pos = {g: k for k, g in enumerate(sub_ids)}
                for row in rd:
                    s_, d_ = row[si].strip(), row[di].strip()
                    if s_ in pos and d_ in pos:
                        (pmid_edges if (pi is not None and row[pi].strip()) else kgml_edges).add((pos[s_], pos[d_]))
            tier_lab, tier_sc = {"TRRUST(PMID)": ([], []), "KGML_only": ([], []), "未分级(不在边表)": ([], [])}
            for (i2, j2) in zip(*np.where(cand)):
                key = (j2, i2)  # mask 行=源→列=靶；GE 真值 T_GE[i,j]: i=靶? 以训练 rhs dG=W_GE@tanh(G) ⇒ W_GE[靶,源]
                t = "TRRUST(PMID)" if key in pmid_edges else ("KGML_only" if key in kgml_edges else "未分级(不在边表)")
                tier_lab[t][0].append(bool(T_GE[i2, j2] > 0)); tier_lab[t][1].append(abs(Wfit[i2, j2]))
            for t, (lab, sc) in tier_lab.items():
                if len(set(lab)) > 1:
                    print(f"  {t}: n={len(lab)} 真边{sum(lab)} AUROC={auroc(np.array(sc), np.array(lab, int)):.3f}")
                else:
                    print(f"  {t}: n={len(lab)} 真边{sum(lab)}（单类,跳过AUROC）")
        else:
            print(f"[降级] glob 未找到 mask_GErel_edges.csv，只报总体 AUROC={auroc(sc_all, truth.astype(int)):.3f}")
else:
    print("[跳过] 缺 ground_truth 或 FIT1 best.pt")

# ========== 模块3: hub 偏差标注（顾问侧 A-53 提示） ==========
print("\n" + "=" * 60); print("模块3: KO源 hub 偏差标定"); print("=" * 60)
SRC12 = [53, 375, 242, 380, 356, 409, 178, 403, 16, 499, 293, 126]
if gt:
    T_GE = gt["T_GE"]; W0 = gt["W_GE"]
    out2 = [s for s in range(T_GE.shape[1]) if T_GE[:, s].sum() >= 2]
    mag = {s: float(np.abs(W0[T_GE[:, s] > 0, s]).sum()) for s in out2}
    vals = np.array(sorted(mag.values()))
    print(f"全体≥2出边基因 {len(out2)} 个; 12个KO源的 sum|w| 分位数:")
    for s in SRC12:
        pct = 100 * (vals < mag[s]).mean()
        print(f"  索引{s}: sum|w|={mag[s]:.3f}, 分位={pct:.0f}%")
    print("\n[论文/申报口径警示] v6的12个KO源按 top|w| 选取, 偏向枢纽基因;")
    print("「一跳定量可信区」对外周(低|w|)基因可能偏乐观——33分诊器阈值标定与论文表述须标注此选择偏差。")

print("\nv9 体检完成")