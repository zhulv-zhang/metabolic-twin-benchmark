# -*- coding: utf-8 -*-
# ============================================================
# mask_GErel（基因→基因调控掩码）构建脚本
# ============================================================
#
# 【功能】构建 基因×基因 的调控掩码（有向，带±1方向），作为 ODE 基因层
#        W_GE ⊙ mask_GErel 的硬约束：掩码=1 的位置才允许学习权重。
#
# 【为什么需要这张掩码 / 救什么】
#   数字孪生的基因层动力学需要"转录因子→靶基因"的调控边。
#   但 KGML 代谢通路图里 GErel（gene expression relation）极其稀疏——
#   8/28 旧版 mask_GErel.npy 有 3.18GB 却几乎是空掩码（边表仅439字节），
#   因此本脚本用双数据源重建：
#     数据源1：本地 13 个 KGML 文件中的 GErel 关系（通路内基因调控）
#     数据源2：TRRUST v2 mouse（文献人工注释的 TF→靶基因，7057对，
#               带激活/抑制方向和 PubMed 证据）——主力来源
#
# 【方向语义】（与 ECrel 一致：方向是软先验 L_direction，不是硬断言）
#   +1 = 激活/促进表达（KGML subtype=expression；TRRUST=Activation）
#   -1 = 抑制表达    （KGML subtype=repression；TRRUST=Repression）
#    0 = 方向未知    （TRRUST=Unknown）
#
# 【防坑设计】（吸收 ECrel/MM 的全部教训）
#   1. 多ID拆分：KGML 一个 entry 可含多个 mmu: 编号（空格分隔），全部展开
#   2. 索引对齐：基因索引用"读 all_kegg_nodes.txt + startswith('mmu:') 字典
#      保留最后出现位置"规则（与 ECrel 主脚本一致），并与矫正后的
#      gene_index_map.json 逐一比对，不一致立即报错停止
#   3. 旧文件保护：检测到旧版 mask_GErel.npy 先改名备份再写新文件
#   4. symbol→mmu 转换：走 KEGG list/mmu（一次请求），存入共享缓存
#      kegg_api_cache.json，重跑不联网
#   5. 生物学抽查：Srebf1→Fasn/Scd1、Foxo1→G6pc 等胰岛素轴已知调控对
#
# 【输入】
#   GSE166336_小鼠组学数据/all_kegg_nodes.txt     节点列表
#   GSE166336_小鼠组学数据/gene_index_map.json    ECrel矫正后的基因索引
#   kegg_kgml/*.xml                               13个KGML通路文件
#   GSE166336_小鼠组学数据/trrust_rawdata.mouse.tsv（或 TRRUST_v2_mouse_转录因子调控数据.tsv）
#   GSE166336_小鼠组学数据/kegg_api_cache.json    共享KEGG缓存（自动追加 list/mmu）
#
# 【输出】
#   mask_GErel.npy        (27999×27999) int8 0/1掩码（int8省空间，训练时转float32）
#   direction_GErel.npy   (27999×27999) int8 方向（+1/-1/0）
#   mask_GErel_edges.csv  边表（含来源和PMID，人工审查用）
#   build_GErel_log.txt   构建日志
#
# 【运行】放在工作目录直接执行：python GErel掩码构建_基因到基因_KGML加TRRUST_带注释.py
# ============================================================

import os
import re
import sys
import json
import time
import shutil
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter

# ----------------- 路径配置（自动定位：当前目录 或 数据子目录） -----------------
DATA_DIR = "GSE166336_小鼠组学数据"

def locate(fname):
    """文件可能在工作目录，也可能在 DATA_DIR 子目录里，两处都找"""
    for cand in [fname, os.path.join(DATA_DIR, fname)]:
        if os.path.exists(cand):
            return cand
    return None

NODE_FILE     = locate("all_kegg_nodes.txt")
GENE_MAP_FILE = locate("gene_index_map.json")
CACHE_FILE    = locate("kegg_api_cache.json") or os.path.join(DATA_DIR, "kegg_api_cache.json")
TRRUST_FILE   = locate("trrust_rawdata.mouse.tsv") or locate("TRRUST_v2_mouse_转录因子调控数据.tsv")

# KGML 目录候选位置
KGML_DIR = None
for cand in ["kegg_kgml", os.path.join(DATA_DIR, "kegg_kgml")]:
    if os.path.isdir(cand):
        KGML_DIR = cand
        break

MASK_OUT = os.path.join(DATA_DIR, "mask_GErel.npy")
DIR_OUT  = os.path.join(DATA_DIR, "direction_GErel.npy")
EDGE_CSV = os.path.join(DATA_DIR, "mask_GErel_edges.csv")
LOG_FILE = os.path.join(DATA_DIR, "build_GErel_log.txt")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=" * 60)
log("mask_GErel 构建：KGML GErel + TRRUST v2 mouse")
log("=" * 60)
for name, fp in [("节点文件", NODE_FILE), ("基因索引", GENE_MAP_FILE),
                 ("KEGG缓存", CACHE_FILE), ("TRRUST", TRRUST_FILE), ("KGML目录", KGML_DIR)]:
    log(f"[路径] {name}: {fp}")

if not NODE_FILE or not GENE_MAP_FILE:
    raise SystemExit("错误：找不到 all_kegg_nodes.txt / gene_index_map.json")

# ============================================================
# 步骤 0：基因索引（与 ECrel 主脚本同规则）+ 对齐校验
# ============================================================
with open(NODE_FILE, "r", encoding="utf-8") as f:
    all_nodes = [l.strip() for l in f if l.strip()]

# 与 ECrel 完全一致：行顺序保留、字典保留最后出现位置（文件含重复行与"-"分隔行）
gene_nodes  = [n for n in all_nodes if n.startswith("mmu:")]
gene_to_idx = {g: i for i, g in enumerate(gene_nodes)}
n_genes = len(gene_nodes)          # 掩码维度（含重复行，27999）
log(f"\n基因节点: {n_genes} 行（唯一ID: {len(gene_to_idx)}）")

with open(GENE_MAP_FILE, "r", encoding="utf-8") as f:
    ecrel_gene_map = json.load(f)
mismatch = [g for g in gene_to_idx
            if g not in ecrel_gene_map or ecrel_gene_map[g] != gene_to_idx[g]]
if mismatch:
    raise SystemExit(f"错误：{len(mismatch)} 个基因与ECrel基因索引不一致，已停止。")
log(f"与ECrel基因索引对齐校验: 通过（{len(gene_to_idx)} 个唯一基因全部一致）")

# 旧文件备份（8/28旧版mask_GErel.npy约3.18GB，为近乎空掩码，备份后可删）
for fp in [MASK_OUT, DIR_OUT, EDGE_CSV]:
    if os.path.exists(fp):
        bak = fp.replace(".npy", "_备份_旧版8月28.npy").replace(".csv", "_备份_旧版8月28.csv")
        if not os.path.exists(bak):
            os.rename(fp, bak)
            log(f"旧文件已改名备份: {bak}（确认为空掩码后可手动删除，省3GB空间）")

# ============================================================
# 步骤 1：解析 KGML 的 GErel 关系（数据源1，预期贡献很少）
# ============================================================
log("\n" + "=" * 60)
log("步骤1: 解析 KGML GErel 关系")
log("=" * 60)

edges_kgml = {}   # (src_gene, dst_gene) -> direction
if KGML_DIR:
    # 注意：文件后缀可能是 .kgml 也可能是 .xml（2026-08-31实测为.kgml）
    xml_files = [f for f in os.listdir(KGML_DIR) if f.endswith((".xml", ".kgml"))]
    log(f"找到 {len(xml_files)} 个 KGML 文件")
    for xf in xml_files:
        tree = ET.parse(os.path.join(KGML_DIR, xf))
        root = tree.getroot()
        # entry_id -> 该entry包含的全部 mmu: 基因编号（多ID拆分，修正1的教训）
        entry_genes = {}
        for entry in root.findall("entry"):
            if entry.get("type") != "gene":
                continue
            ids = [k for k in entry.get("name", "").split() if k.startswith("mmu:")]
            if ids:
                entry_genes[entry.get("id")] = ids
        # relation: type="GErel"，subtype name=expression(+1)/repression(-1)
        for rel in root.findall("relation"):
            if rel.get("type") != "GErel":
                continue
            srcs = entry_genes.get(rel.get("entry1"), [])
            dsts = entry_genes.get(rel.get("entry2"), [])
            d = 0
            for sub in rel.findall("subtype"):
                if sub.get("name") == "expression":
                    d = +1
                elif sub.get("name") == "repression":
                    d = -1
            for s in srcs:
                for t in dsts:
                    if s != t and s in gene_to_idx and t in gene_to_idx:
                        edges_kgml[(s, t)] = d
else:
    log("警告：未找到 kegg_kgml 目录，KGML 数据源跳过（TRRUST 仍可建网）")

log(f"KGML GErel 有效边（两端都在基因节点内）: {len(edges_kgml)} 条")

# ============================================================
# 步骤 2：TRRUST symbol→mmu 转换（需要一次 KEGG list/mmu，带缓存）
# ============================================================
log("\n" + "=" * 60)
log("步骤2: TRRUST symbol→mmu编号 转换")
log("=" * 60)

cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    log(f"已加载共享缓存: {len(cache)} 条")

LIST_URL = "https://rest.kegg.jp/list/mmu"
if LIST_URL in cache and cache[LIST_URL]:
    list_text = cache[LIST_URL]
    log("symbol映射: 命中缓存，未联网")
else:
    log("symbol映射: 请求 KEGG list/mmu（约1MB，仅此一次，之后进缓存）...")
    list_text = None
    for attempt in range(3):
        try:
            r = requests.get(LIST_URL, timeout=60)
            if r.status_code == 200:
                list_text = r.text
                break
        except Exception as e:
            log(f"  第{attempt+1}次请求失败: {e}")
            time.sleep(2)
    if not list_text:
        raise SystemExit("错误：KEGG list/mmu 请求失败，无法做symbol→mmu转换")
    cache[LIST_URL] = list_text
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

# 解析 list/mmu。兼容两种已知格式（2026-08-31实测为第二种）：
#   标准格式(2列): "mmu:11461\tActb; actin, beta; K05693"
#   CDS格式(4列):  "mmu:212427\tCDS\t1:complement(390..7732)\tA730008H23Rik, Fleg1; Holliday junction..."
# 描述列中分号前为symbol区，可能含多个别名（逗号分隔）——别名全部进映射表，
# 能显著提高 TRRUST 的命中率（TRRUST用的symbol不一定是KEGG首选名）
symbol_to_mmu = {}
for line in list_text.strip().split("\n"):
    parts = line.split("\t")
    if not parts[0].startswith("mmu:"):
        continue
    if len(parts) >= 4 and parts[1] == "CDS":
        desc = parts[3]              # CDS格式：第4列是描述
    elif len(parts) >= 2:
        desc = parts[1]              # 标准格式：第2列是描述
    else:
        continue
    sym_field = desc.split(";")[0]   # 分号前 = symbol区
    for sym in sym_field.split(","):
        sym = sym.strip()
        if sym:
            symbol_to_mmu.setdefault(sym, parts[0])

if len(symbol_to_mmu) < 1000:
    raise SystemExit(f"错误：symbol映射仅解析出 {len(symbol_to_mmu)} 个，"
                     f"list/mmu 响应格式可能又变了，请把诊断输出发回")
log(f"symbol→mmu 映射表: {len(symbol_to_mmu)} 个symbol（含别名）")

def sym2mmu(sym):
    """symbol转mmu编号：先精确匹配，再大小写不敏感兜底"""
    if sym in symbol_to_mmu:
        return symbol_to_mmu[sym]
    low = sym.lower()
    for s, m in symbol_to_mmu.items():
        if s.lower() == low:
            return m
    return None

# ============================================================
# 步骤 3：解析 TRRUST，生成基因→基因边（数据源2，主力）
# ============================================================
log("\n" + "=" * 60)
log("步骤3: 解析 TRRUST v2 mouse")
log("=" * 60)

if not TRRUST_FILE:
    raise SystemExit("错误：找不到 TRRUST 文件（trrust_rawdata.mouse.tsv 或 中文名版本），"
                     "请放入 GSE166336_小鼠组学数据\\ 目录")

edges_trrust = {}   # (src_gene, dst_gene) -> (direction, pmid)
n_nomatch_tf = n_nomatch_tg = n_notin_nodes = 0
with open(TRRUST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        tf_sym, tg_sym, mode = parts[0], parts[1], parts[2]
        pmid = parts[3] if len(parts) > 3 else ""
        tf = sym2mmu(tf_sym)
        tg = sym2mmu(tg_sym)
        if not tf:
            n_nomatch_tf += 1
            continue
        if not tg:
            n_nomatch_tg += 1
            continue
        if tf not in gene_to_idx or tg not in gene_to_idx:
            n_notin_nodes += 1          # 基因不在节点列表（未测到/不在范围）
            continue
        if tf == tg:
            continue
        d = {"Activation": 1, "Repression": -1}.get(mode, 0)
        edges_trrust[(tf, tg)] = (d, pmid)

log(f"TRRUST 原始7057对 → 有效边 {len(edges_trrust)} 条")
log(f"  过滤明细: TF无symbol匹配 {n_nomatch_tf}，靶基因无symbol匹配 {n_nomatch_tg}，"
    f"基因不在节点列表 {n_notin_nodes}")

# ============================================================
# 步骤 4：合并两数据源 → 写入掩码
# ============================================================
log("\n" + "=" * 60)
log("步骤4: 合并并写入掩码")
log("=" * 60)

# int8 足够表达 0/±1，比 float32 省 3/4 空间（27999² int8 ≈ 784MB，float32 ≈ 3.1GB）
mask = np.zeros((n_genes, n_genes), dtype=np.int8)
dirmask = np.zeros((n_genes, n_genes), dtype=np.int8)

n_conflict = 0
rows = []
# TRRUST 边
for (s, t), (d, pmid) in edges_trrust.items():
    i, j = gene_to_idx[s], gene_to_idx[t]
    mask[i, j] = 1
    dirmask[i, j] = d
    rows.append({"source_gene": s, "target_gene": t, "direction_code": d,
                 "direction_label": {1: "activation", -1: "repression", 0: "unknown"}[d],
                 "source": "TRRUST_v2", "pmid": pmid})
# KGML GErel 边（不覆盖 TRRUST 已有方向；方向冲突时记0并计数）
for (s, t), d in edges_kgml.items():
    i, j = gene_to_idx[s], gene_to_idx[t]
    if mask[i, j] == 0:
        mask[i, j] = 1
        dirmask[i, j] = d
        rows.append({"source_gene": s, "target_gene": t, "direction_code": d,
                     "direction_label": {1: "expression", -1: "repression", 0: "unknown"}[d],
                     "source": "KGML_GErel", "pmid": ""})
    else:
        if dirmask[i, j] != 0 and d != 0 and dirmask[i, j] != d:
            n_conflict += 1
            dirmask[i, j] = 0      # 方向冲突 → 降级为未知，不硬选边
        elif dirmask[i, j] == 0 and d != 0:
            dirmask[i, j] = d      # TRRUST未知方向，KGML补上

n_total = int(mask.sum())
n_dir = int((dirmask != 0).sum())
log(f"合并后总边数: {n_total}（KGML {len(edges_kgml)} + TRRUST {len(edges_trrust)}，有重叠）")
log(f"方向已知: {n_dir}（{n_dir / max(n_total, 1) * 100:.1f}%）；方向冲突降级: {n_conflict} 条")
log(f"掩码形状: {mask.shape}，密度: {n_total / (n_genes * (n_genes - 1)):.6%}")

# ============================================================
# 步骤 5：生物学抽查（胰岛素-脂合成轴已知调控对）
# ============================================================
log("\n" + "=" * 60)
log("步骤5: 生物学抽查")
log("=" * 60)
edge_set = set(zip([r["source_gene"] for r in rows], [r["target_gene"] for r in rows]))
for tf_sym, tg_sym, note in [
    ("Srebf1", "Fasn",  "SREBP1c激活脂肪酸合成酶，胰岛素脂合成核心轴"),
    ("Srebf1", "Scd1",  "SREBP1c→硬脂酰CoA去饱和酶"),
    ("Foxo1",  "G6pc",  "FoxO1激活糖异生（胰岛素抑制FoxO1）"),
    ("Foxo1",  "Pck1",  "FoxO1→PEPCK糖异生"),
    ("Mlxipl", "Pklr",  "ChREBP→肝型丙酮酸激酶（糖信号）"),
]:
    tf, tg = sym2mmu(tf_sym), sym2mmu(tg_sym)
    if not tf or not tg:
        log(f"  {tf_sym}→{tg_sym}: symbol无法映射")
        continue
    hit = (tf, tg) in edge_set
    extra = ""
    if hit:
        d = int(dirmask[gene_to_idx[tf], gene_to_idx[tg]])
        extra = f"，方向 {d}"
    log(f"  {tf_sym}→{tg_sym}: {'✓ 在网'+extra if hit else '✗ 不在（靶基因未测到或TRRUST未收录，正常）'}  [{note}]")

# ============================================================
# 步骤 6：保存
# ============================================================
np.save(MASK_OUT, mask)
np.save(DIR_OUT, dirmask)
pd.DataFrame(rows).to_csv(EDGE_CSV, index=False, encoding="utf-8-sig")
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log(f"\n已保存: {MASK_OUT} ({os.path.getsize(MASK_OUT)/1e6:.0f}MB)")
log(f"已保存: {DIR_OUT}, {EDGE_CSV}（{len(rows)}行）, {LOG_FILE}")
log("完成。注意：掩码为int8格式，训练代码加载后需 .astype(np.float32)。")