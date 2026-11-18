# -*- coding: utf-8 -*-
# ============================================================
# 数据准备：从 mmc3.xlsx 重建白盒训练张量（v1）
# ============================================================
#
# 【为什么不用旧的 cleaned 矩阵】
#   旧流程（见《GSE166336多组学数据处理文档》）的 time_map 把四层组学
#   统一标成 0/7.5/15/30/60，但实测真实网格是：
#     转录组/蛋白组: 0/60/120/180/240 min（5点）
#     磷酸化组:      0/7.5/15/30/60/120/180/240 min（8点）
#     代谢组:        0/2.5/5/7.5/10/15/20/30 min（8点）
#   旧版时间轴四层错位（转录组240min被标成60、代谢组10min被标成60），
#   且只取了前5个时间点。本脚本直接从 mmc3.xlsx 重建：
#   每层用自己的真实时间轴、全部时间点、不插补（NaN+观测掩码）。
#
# 【处理链】（每步都写入报告，Methods 可直接引用）
#   1. 解析 #! 元数据行：样本列、条件（Insulin/PBS）、真实时间（分钟）
#   2. 特征→ID：优先用 sheet 自带的 KeggId 注释列（mmu:/C 编号）；
#      代谢组若无 KeggId 列则用 metabolome_kegg_map.csv 按名称映射
#   3. 同 ID 聚合：磷酸化多个位点→基因取平均；重复基因取平均
#   4. Z-score 标准化：按特征、跨全部样本，nan-aware（不插补）
#   5. 对齐掩码索引：紧凑排序（14076 唯一基因 / 193 唯一代谢物，
#      按 gene/metabolite_index_map.json 的行号顺序），未测到=NaN+掩码0
#
# 【输出】（都在 GSE166336_小鼠组学数据\）
#   train_tensors.npz：
#     XG (14076,5,2) 转录组  / tG (5,)  / OG 观测掩码
#     XP (14076,5,2) 蛋白组  / tP (5,)  / OP
#     XS (14076,8,2) 磷酸化  / tS (8,)  / OS
#     XM (193,8,2)   代谢组  / tM (8,)  / OM
#     conditions = ['Insulin','PBS']（第3维的顺序）
#     gene_ids (14076,) / metab_ids (193,)（行对应的节点ID）
#   数据处理报告_白盒训练张量.md（Methods 素材）
#   build_data_log.txt
#
# 【运行】python 数据准备_从mmc3重建训练张量_带注释.py
# ============================================================

import os
import re
import json
import numpy as np
import pandas as pd

DATA_DIR = "GSE166336_小鼠组学数据"
MMC3 = "mmc3.xlsx"                     # 在工作目录（scTenifoldKnk_test）

def locate(fname):
    for cand in [fname, os.path.join(DATA_DIR, fname)]:
        if os.path.exists(cand):
            return cand
    return None

GENE_MAP  = locate("gene_index_map.json")
METAB_MAP = locate("metabolite_index_map.json")
METAB_KEGG_MAP = locate("metabolome_kegg_map.csv")   # 代谢组名称→C编号备用
LOG_FILE = os.path.join(DATA_DIR, "build_data_log.txt")
REPORT   = os.path.join(DATA_DIR, "数据处理报告_白盒训练张量.md")
NPZ_OUT  = os.path.join(DATA_DIR, "train_tensors.npz")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=" * 60)
log("数据准备：mmc3.xlsx → 训练张量（真实时间轴版）")
log("=" * 60)

# ---- 掩码索引（紧凑唯一顺序） ----
with open(GENE_MAP, encoding="utf-8") as f:
    gene_map = json.load(f)
with open(METAB_MAP, encoding="utf-8") as f:
    metab_map = json.load(f)
gene_ids  = [g for g, _ in sorted(gene_map.items(),  key=lambda x: x[1])]   # 14076
metab_ids = [m for m, _ in sorted(metab_map.items(), key=lambda x: x[1])]   # 193
gpos = {g: i for i, g in enumerate(gene_ids)}
mpos = {m: i for i, m in enumerate(metab_ids)}
log(f"紧凑索引: 基因 {len(gene_ids)}, 代谢物 {len(metab_ids)}")

# ============================================================
# 通用 sheet 解析器
# ============================================================
def strip_tag(v):
    """元数据单元格形如 '#!{C:Cond_Time}Insulin_1'（标签和值黏在一起，
    尤其第0列），剥掉 '#!{...}' 前缀取真值；普通值原样返回。"""
    v = str(v)
    return v.split("}", 1)[-1].strip() if v.startswith("#!") else v.strip()

def parse_sheet(sheet_name, id_pattern, map_file=None):
    """解析一个组学 sheet。
    返回: values (n_features × n_samples) float,
          feat_ids (n_features,) 该特征的 mmu:/C 编号（None=未解析）,
          times (n_samples,) 分钟, conds (n_samples,) 'Insulin'/'PBS'
    map_file: 对应的 *_kegg_map.csv；若行数与sheet特征数一致且已解析行
              一致率≥99%，用 map 位置对齐回填缺失ID。
    """
    raw = pd.read_excel(MMC3, sheet_name=sheet_name, header=None)
    ncol = raw.shape[1]
    col0 = raw.iloc[:, 0].astype(str)

    # 定位 #! 元数据行
    def find_row(tag):
        hits = [i for i in range(min(10, len(raw))) if col0.iloc[i].startswith(tag)]
        return hits[0] if hits else None
    r_condtime = find_row("#!{C:Cond_Time}")
    r_time     = find_row("#!{C:Time_Unit}")
    r_cond     = find_row("#!{C:Cond}")
    if r_condtime is None or r_time is None:
        raise SystemExit(f"[{sheet_name}] 找不到 #! 元数据行（Cond_Time/Time_Unit）")
    data_start = max(i for i in range(min(10, len(raw)))
                     if col0.iloc[i].startswith("#!")) + 1

    # 样本列 = Cond_Time 行里形如 Insulin_x / PBS_x 的列（先剥 #! 前缀，
    # 否则第0列的 '#!{C:Cond_Time}Insulin_1' 匹配不上，会丢 t=0 基线样本）
    sample_cols, times, conds = [], [], []
    for c in range(ncol):
        v = strip_tag(raw.iloc[r_condtime, c])
        m = re.fullmatch(r"(Insulin|PBS)_\d+", v)
        if m:
            sample_cols.append(c)
            conds.append(m.group(1))
            try:
                times.append(float(strip_tag(raw.iloc[r_time, c])))
            except Exception:
                times.append(np.nan)
    log(f"[{sheet_name}] 样本列 {len(sample_cols)} 个 "
        f"({dict(pd.Series(conds).value_counts())})，时间点 {sorted(set(times))} min")

    # 特征 ID 列：优先找表头叫 KeggId 的列；否则找内容匹配率最高的注释列
    header = raw.iloc[0].astype(str).tolist()
    annot_cols = [c for c in range(ncol) if c not in sample_cols]
    id_col = None
    if "KeggId" in header:
        id_col = header.index("KeggId")
    else:
        best, best_rate = None, 0
        probe = raw.iloc[data_start:data_start + 500]
        for c in annot_cols:
            vals = probe.iloc[:, c].astype(str)
            rate = vals.str.fullmatch(id_pattern).mean()
            if rate > best_rate:
                best, best_rate = c, rate
        if best_rate > 0.3:
            id_col = best
        log(f"[{sheet_name}] 无KeggId表头，自动选中列 {id_col}（匹配率 {best_rate:.0%}）")

    # 提取数值矩阵 + 特征ID
    data = raw.iloc[data_start:]
    values = data.iloc[:, sample_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    if id_col is not None:
        feat_ids = data.iloc[:, id_col].astype(str).str.strip()
        feat_ids = [v if re.fullmatch(id_pattern, v) else None for v in feat_ids]
    else:
        feat_ids = [None] * len(data)

    # 位置对齐回填：sheet行数==map行数 且 已解析行一致率≥99% → 用map回填缺失
    if map_file and any(v is None for v in feat_ids):
        mm = pd.read_csv(map_file)
        if len(mm) == len(feat_ids):
            both = [(a, b) for a, b in zip(feat_ids, mm["kegg_id"].astype(str)) if a]
            agree = np.mean([a == b for a, b in both]) if both else 0
            if agree >= 0.99:
                n_fill = 0
                for i in range(len(feat_ids)):
                    if feat_ids[i] is None:
                        v = str(mm["kegg_id"].iloc[i]).strip()
                        if re.fullmatch(id_pattern, v):
                            feat_ids[i] = v
                            n_fill += 1
                log(f"[{sheet_name}] 位置对齐回填: +{n_fill} 个ID（一致率 {agree:.1%}）")
            else:
                log(f"[{sheet_name}] 位置对齐验证失败（一致率 {agree:.1%} <99%），不回填")
        else:
            log(f"[{sheet_name}] 行数不等（sheet {len(feat_ids)} vs map {len(mm)}），不回填")

    n_ok = sum(1 for v in feat_ids if v)
    log(f"[{sheet_name}] 特征 {len(data)} 行，ID解析成功 {n_ok} "
        f"({n_ok / max(len(data),1) * 100:.1f}%)，缺失率 {np.isnan(values).mean():.1%}")
    return values, feat_ids, np.array(times), conds

def aggregate_by_id(values, feat_ids):
    """同一ID的多行（磷酸化位点/重复基因）取nan均值，返回 {id: 样本向量}。
    全NaN的特征聚合后仍全NaN（观测掩码会处理），此处静默告警。"""
    acc = {}
    for i, fid in enumerate(feat_ids):
        if fid:
            acc.setdefault(fid, []).append(values[i])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 全NaN行的 nanmean 告警，无害
        return {fid: np.nanmean(np.vstack(rows), axis=0) for fid, rows in acc.items()}

def zscore(x):
    """按行（特征）nan-aware z-score；常数行置0"""
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd

def build_tensor(id2vec, times, conds, pos, id_list, sheet_name):
    """{id: 向量} → (n_nodes, n_time, 2) 张量 + 观测掩码"""
    t_unique = np.array(sorted(set(times)))
    X = np.full((len(id_list), len(t_unique), 2), np.nan, dtype=np.float32)
    for fid, vec in id2vec.items():
        if fid not in pos:
            continue
        for s in range(len(times)):
            c = 0 if conds[s] == "Insulin" else 1
            ti = int(np.where(t_unique == times[s])[0][0])
            X[pos[fid], ti, c] = vec[s]
    O = (~np.isnan(X)).astype(np.float32)
    n_measured = int((O.sum(axis=(1, 2)) > 0).sum())
    log(f"[{sheet_name}] 入张量的节点: {n_measured}/{len(id_list)}，"
        f"观测覆盖率 {O.sum() / O.size:.1%}")
    return X.astype(np.float32), t_unique.astype(np.float32), O

# ============================================================
# 逐层处理
# ============================================================
log("\n--- 转录组 ---")
v, ids, t, c = parse_sheet("transcriptome", r"mmu:\d+",
                           map_file=locate("transcriptome_kegg_map.csv"))
id2v = aggregate_by_id(v, ids)
id2v = {k: zscore(val[None, :])[0] for k, val in id2v.items()}
XG, tG, OG = build_tensor(id2v, t, c, gpos, gene_ids, "transcriptome")

log("\n--- 蛋白组 ---")
v, ids, t, c = parse_sheet("proteome", r"mmu:\d+",
                           map_file=locate("proteome_kegg_map.csv"))
id2v = aggregate_by_id(v, ids)
id2v = {k: zscore(val[None, :])[0] for k, val in id2v.items()}
XP, tP, OP = build_tensor(id2v, t, c, gpos, gene_ids, "proteome")

log("\n--- 磷酸化组（位点→基因聚合） ---")
v, ids, t, c = parse_sheet("phosphoproteome", r"mmu:\d+",
                           map_file=locate("phosphoproteome_kegg_map.csv"))
id2v = aggregate_by_id(v, ids)          # 位点均值→基因
id2v = {k: zscore(val[None, :])[0] for k, val in id2v.items()}
XS, tS, OS = build_tensor(id2v, t, c, gpos, gene_ids, "phosphoproteome")

log("\n--- 代谢组 ---")
v, ids, t, c = parse_sheet("metabolome", r"C\d{5}")   # 行数与map不等(204≠194)，不用位置回填
if METAB_KEGG_MAP:
    # 代谢组用名称列 + metabolome_kegg_map.csv 回填/补全缺失的C编号
    # （sheet 有C编号列但只覆盖76%，名称映射可捞回其余~24%）
    mm = pd.read_csv(METAB_KEGG_MAP)
    name2c = dict(zip(mm["feature_name"].astype(str).str.strip(),
                      mm["kegg_id"].astype(str).str.strip()))
    n_missing = sum(1 for i in ids if i is None)
    if n_missing > 0:
        raw = pd.read_excel(MMC3, sheet_name="metabolome", header=None)
        # 与parse_sheet相同的data_start计算，保证名称行与ID行严格对齐
        c0 = raw.iloc[:, 0].astype(str)
        data_start = max(i for i in range(min(10, len(raw)))
                         if c0.iloc[i].startswith("#!")) + 1
        # 找名称列：与map的feature_name匹配率最高的列
        best, best_rate = None, 0
        for ci in range(16, raw.shape[1]):
            vals = raw.iloc[data_start:data_start + 200, ci].astype(str).str.strip()
            rate = vals.isin(name2c).mean()
            if rate > best_rate:
                best, best_rate = ci, rate
        names = raw.iloc[data_start:, best].astype(str).str.strip().tolist()
        n_fill = 0
        for i in range(len(ids)):
            if ids[i] is None and i < len(names):
                cid = name2c.get(names[i])
                if cid:
                    ids[i] = cid
                    n_fill += 1
        log(f"  名称映射回填: 名称列=第{best}列（map匹配率 {best_rate:.0%}），"
            f"缺失 {n_missing} → 回填 {n_fill}，仍缺 {n_missing - n_fill}")
id2v = aggregate_by_id(v, ids)
id2v = {k: zscore(val[None, :])[0] for k, val in id2v.items()}
XM, tM, OM = build_tensor(id2v, t, c, mpos, metab_ids, "metabolome")

# ============================================================
# 保存
# ============================================================
np.savez_compressed(
    NPZ_OUT,
    XG=XG, tG=tG, OG=OG,
    XP=XP, tP=tP, OP=OP,
    XS=XS, tS=tS, OS=OS,
    XM=XM, tM=tM, OM=OM,
    conditions=np.array(["Insulin", "PBS"]),
    gene_ids=np.array(gene_ids), metab_ids=np.array(metab_ids),
)
log(f"\n已保存: {NPZ_OUT} ({os.path.getsize(NPZ_OUT)/1e6:.1f}MB)")

# ---- 报告（Methods 素材） ----
rep = f"""# 数据处理报告：GSE166336 白盒训练张量（{pd.Timestamp.now():%Y-%m-%d}）

**数据源**：mmc3.xlsx（GSE166336 补充材料 Table S2），四层组学直接从原始 sheet 重建，
不使用旧 cleaned 矩阵（旧流程 time_map 存在四层时间轴错位：转录组 240min 误标为 60min、
代谢组 10min 误标为 60min，本版已修正）。

## 各层时间网格与规模
| 组学 | 时间点(min) | 条件 | 节点覆盖 | 观测覆盖率 |
|---|---|---|---|---|
| 转录组 | {list(tG)} | Insulin/PBS | {int((OG.sum(axis=(1,2))>0).sum())}/{len(gene_ids)} 基因 | {OG.sum()/OG.size:.1%} |
| 蛋白组 | {list(tP)} | Insulin/PBS | {int((OP.sum(axis=(1,2))>0).sum())}/{len(gene_ids)} 基因 | {OP.sum()/OP.size:.1%} |
| 磷酸化组 | {list(tS)} | Insulin/PBS | {int((OS.sum(axis=(1,2))>0).sum())}/{len(gene_ids)} 基因（位点已聚合） | {OS.sum()/OS.size:.1%} |
| 代谢组 | {list(tM)} | Insulin/PBS | {int((OM.sum(axis=(1,2))>0).sum())}/{len(metab_ids)} 代谢物 | {OM.sum()/OM.size:.1%} |

## 处理链
1. 从各 sheet 的 #! 元数据行解析样本列、条件与真实采样时间（分钟）。
2. 特征→KEGG ID：优先使用 sheet 自带 KeggId 注释列；代谢组用名称映射
   （metabolome_kegg_map.csv）。
3. 同 ID 聚合：磷酸化多位点→基因（nan 均值）；重复基因同理。
4. Z-score 标准化：按特征、跨全部样本，nan-aware；**不做缺失值插补**，
   缺失由观测掩码（OG/OP/OS/OM）在掩码损失中原生处理。
5. 对齐掩码紧凑索引：14076 唯一基因 / 193 唯一代谢物。

## 输出
train_tensors.npz：XG/XP/XS/XM（值，z-score），tG/tP/tS/tM（分钟），
OG/OP/OS/OM（观测掩码），conditions=[Insulin, PBS]，gene_ids/metab_ids。
"""
with open(REPORT, "w", encoding="utf-8") as f:
    f.write(rep)
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"已保存: {REPORT}")
log(f"已保存: {LOG_FILE}")
log("完成。")