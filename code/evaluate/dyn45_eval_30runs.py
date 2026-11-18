# -*- coding: utf-8 -*-
# dyn45_eval_30runs_重建_1016.py
# 重建口径——本件为 10-04/10-05 白昼评估件的重建件（登记"重建口径"，不冒充原件）；
# 母体：dyn45对死评估 v1.4 A25FCC90；地基/朝向段对源直抄自扫件 v2.2（dyn45口径钉死扫件_20261006.py，纹 5DFC6D08）；
# 锚件：sweep 字据 dyn45_convention_sweep_1006.json（EC0B1D2C 三纹定谳）——朝向终裁以字据为准；
# 纪律：只跑 Insulin（正稿 §4.7 不区分条件）；禁 auto-select（朝向钉死）；三硬闸复现不出即停手报 Advisor；
# 重建日 2026-10-16（SE 主号按 D1 令写）。
import pickle, json, hashlib, sys, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

# ---------- 输入（全内嵌·禁外读） ----------
GT2   = Path(r"D:\lv\sctenifoldknk\fullbackup_0924_解包_1003\data_sim2\ground_truth.npz")
GT2I  = Path(r"D:\lv\sctenifoldknk\fullbackup_0924_解包_1003\data_sim2i\ground_truth.npz")
MASK  = Path(r"D:\lv\生信平台\scTenifoldKnk_test\GSE166336_小鼠组学数据\mask_GErel_edges.csv")
ISO   = Path(r"D:\lv\sctenifoldknk\_隔离_重切片_b口径")
VIM_DIR  = Path(r"D:\lv\sctenifoldknk\_隔离_\dyn45_1004")
MANIFEST = VIM_DIR / "dyn45_manifest_1004.json"
OUT = Path(r"D:\lv\sctenifoldknk\论文台账1005\dyn45重算核验_1016\dyn45_eval_30runs_重建_1016.csv")

# ---------- GT 双实核（对不上停手报） ----------
def md5_8(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

for p in (GT2, GT2I):
    assert p.exists(), f"停线：缺 {p}"
    assert md5_8(p) == "8490eb11", f"停线：GT 纹 {md5_8(p)} != 8490EB11（{p}）"

# ---------- auroc_tie（A25FCC90 源第 14-19 行逐字直贴版，D1 令三替换） ----------
def auroc_tie(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = int(l.sum()), int((~l).sum())
    assert n_pos > 0 and n_neg > 0, "正负类缺一，停线"
    r = rankdata(s)
    return (r[l].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

# ---------- 地基（对源直抄自扫件 v2.2：GT 读入/域索引/W 标签；
# W 一尺量两宇宙为母体原设计——双 GT 同纹已断言同源，此处照抄不改） ----------
gt = np.load(GT2, allow_pickle=True)
gid = [str(g) for g in gt["sub_gene_ids"]]; gpos = {g:i for i,g in enumerate(gid)}
W = np.asarray(gt["W_GE"], float)
mk = pd.read_csv(MASK); pairs = [(str(a), str(b)) for a, b in zip(mk.iloc[:,0], mk.iloc[:,1])]

# ---------- 朝向钉死（纪律②；组合定义照扫件第 44-52 行） ----------
PINS = {"data_sim2": ("V", "raw", "ba"), "data_sim2i": ("VT", "raw", "ba")}

LIMIT = int(os.environ.get("LIMIT", "0"))  # 试水闸：LIMIT=1 先跑每宇宙首发自检，0＝全量 30
man_all = json.load(open(MANIFEST, encoding="utf-8"))
rows = []
for uni in ("data_sim2", "data_sim2i"):
    man = [m for m in man_all if m["tag"].startswith(uni + "_")]
    assert man, f"停线：{uni} 无 VIM 登记"
    XI = pd.read_csv(ISO / f"{uni}_b口径{man[0]['n_genes']}_Insulin.csv")  # 纪律①：只跑 Insulin
    pick = [str(c) for c in XI.columns]
    locs = {g:gpos[g] for g in pick if g in gpos}
    dom = [(a,b) for a,b in pairs if a in locs and b in locs]
    ia = np.array([locs[a] for a,b in dom]); ib = np.array([locs[b] for a,b in dom])
    lab = {"ba": (W[ib,ia] != 0), "ab": (W[ia,ib] != 0)}
    pidx = {g:i for i,g in enumerate(pick)}
    ia2 = np.array([pidx[a] for a,b in dom]); ib2 = np.array([pidx[b] for a,b in dom])
    vn, mn, ln = PINS[uni]
    for m in (man[:LIMIT] if LIMIT else man):
        with open(VIM_DIR / f"{m['tag']}_VIM.pkl", "rb") as f:
            V = np.asarray(pickle.load(f)[0], float)
        M = V if vn == "V" else V.T
        if mn == "colN":
            with np.errstate(invalid="ignore", divide="ignore"):
                M = M / M.sum(axis=0, keepdims=True)
            M[~np.isfinite(M)] = 0
        a = auroc_tie(M[ia2, ib2], lab[ln])
        rows.append((uni, m["tag"], m["trees"], m["seed"], round(float(a), 4)))

df = pd.DataFrame(rows, columns=["universe", "tag", "trees", "seed", "AUROC"])
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False, encoding="utf-8-sig")

# ---------- 汇总与硬闸（1000t 档 5 发口径——正稿 §4.7"0.607±0.016 on the sim2i 1000t universe"；复现不出即停手报） ----------
if LIMIT:
    print(f"[试水模式 LIMIT={LIMIT}] 行数 {len(df)}——读入/朝向/写盘通路已验；硬闸与断言全量时生效")
    sys.exit(0)
assert len(df) == 30, f"停线：总行 {len(df)} != 30——查 manifest tag 过滤（v2.1 修过 sim2 吞并 sim2i）"
t1000 = df["trees"].astype(str).str.replace(r"\D", "", regex=True) == "1000"  # 兼容 1000/"1000"/"1000t"
sub_i = df[(df.universe == "data_sim2i") & t1000]["AUROC"]
sub_2 = df[(df.universe == "data_sim2") & t1000]["AUROC"]
assert len(sub_i) == 5 and len(sub_2) == 5, f"停线：1000t 子集 {len(sub_i)}/{len(sub_2)} 发 != 5/5——查 manifest trees 档"
mu_i, sd_i = float(sub_i.mean()), float(sub_i.std(ddof=0))
mu_2 = float(sub_2.mean())
print(f"sim2i 1000t（{len(sub_i)} 发）：{mu_i:.4f}±{sd_i:.4f}（闸：0.607±0.016）")
print(f"sim2 1000t（{len(sub_2)} 发）：{mu_2:.4f}（闸：0.562）")
print("p：锚 1b0f2080（源无置换段，照锚不现算；500-permutation p=0.0008 正稿在案）")
ok = (abs(mu_i - 0.607) <= 1e-3) and (abs(sd_i - 0.016) <= 5e-4) and (abs(mu_2 - 0.562) <= 1e-3)
if not ok:
    print("❌ 硬闸不过——与登记值不合，停手报 Advisor（CSV 留档对质，不算数）")
    sys.exit(1)
print("✅ 硬闸过——报四格（件纹=CSV md5 / 代码纹=本件 md5 / 三数 / 闸态）")
