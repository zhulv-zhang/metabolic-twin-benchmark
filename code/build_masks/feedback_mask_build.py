# -*- coding: utf-8 -*-
# ============================================================
# mask_feedback（代谢物→蛋白 反馈掩码）构建脚本
# ============================================================
#
# 【功能】构建 代谢物→蛋白 的 0/1 掩码，作为 ODE 中
#        "代谢物反馈调节信号蛋白" 的硬约束（如 AMP→AMPK、NAD+→SIRT1、
#        谷氨酰胺→mTOR 这类"代谢物作为信号分子被感知"的边）。
#
# 【设计依据】（2026-08-31 定型，见进度文档）
#   源头（行）：信号代谢物白名单 ∩ 实际测到的194节点
#     白名单11个：ATP/AMP/乙酰CoA/NAD+/NADH/葡萄糖/谷氨酸/谷氨酰胺/柠檬酸/丙酮酸/甘油
#     实测仅8个在节点列表：C00002(ATP) C00020(AMP) C00024(乙酰CoA) C00003(NAD+)
#                          C00004(NADH) C00025(谷氨酸) C00064(谷氨酰胺) C00158(柠檬酸)
#     缺3个：C00031(葡萄糖) C00022(丙酮酸) C00116(甘油)——未测到，不能作反馈源，
#            论文口径：数据决定网络边界，诚实说明即可。
#   靶点（列）：FEEDBACK_SCOPE 控制
#     "signaling"（默认/推荐）：13个KGML中信号通路(04xxx系列)的全部基因
#        ——代谢物反馈的生物学本质就是被信号通路感知，收窄合理
#     "all"：全部27999个基因/蛋白——原设计的宽口径，参数多22万，慎用
#
# 【掩码方向约定】行=源（代谢物, 194维），列=靶（蛋白/基因, 27999维）
#   与 ECrel 一致：行→列 = 作用方向。本掩码不带方向（反馈调节符号由数据学）。
#
# 【防坑设计】同前几版：基因/代谢物索引与 ECrel 主脚本同规则重建 +
#   双对齐校验（gene_index_map.json / metabolite_index_map.json），
#   不一致立即报错停止；多ID拆分；旧文件自动备份。
#
# 【输入】
#   all_kegg_nodes.txt / gene_index_map.json / metabolite_index_map.json
#   kegg_kgml/*.kgml（取 04xxx 信号通路的基因集合）
#
# 【输出】
#   mask_feedback.npy        (194×27999) int8
#   mask_feedback_edges.csv  边表（metabolite, protein, 所属信号通路）
#   build_feedback_log.txt
#
# 【运行】python feedback掩码构建_代谢物到蛋白_带注释.py
# ============================================================

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# ----------------- 配置区 -----------------
FEEDBACK_SCOPE = "signaling"   # "signaling"（默认，推荐）或 "all"（全蛋白，原宽设计）

# 信号代谢物白名单：名称 → KEGG C编号（脚本自动检查哪些在194节点内）
WHITELIST = {
    "ATP":      "C00002",
    "AMP":      "C00020",
    "乙酰CoA":  "C00024",
    "NAD+":     "C00003",
    "NADH":     "C00004",
    "葡萄糖":   "C00031",
    "谷氨酸":   "C00025",
    "谷氨酰胺": "C00064",
    "柠檬酸":   "C00158",
    "丙酮酸":   "C00022",
    "甘油":     "C00116",
}

DATA_DIR = "GSE166336_小鼠组学数据"

def locate(fname):
    for cand in [fname, os.path.join(DATA_DIR, fname)]:
        if os.path.exists(cand):
            return cand
    return None

NODE_FILE = locate("all_kegg_nodes.txt")
GENE_MAP  = locate("gene_index_map.json")
METAB_MAP = locate("metabolite_index_map.json")
KGML_DIR  = None
for cand in ["kegg_kgml", os.path.join(DATA_DIR, "kegg_kgml")]:
    if os.path.isdir(cand):
        KGML_DIR = cand
        break

MASK_OUT = os.path.join(DATA_DIR, "mask_feedback.npy")
EDGE_CSV = os.path.join(DATA_DIR, "mask_feedback_edges.csv")
LOG_FILE = os.path.join(DATA_DIR, "build_feedback_log.txt")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=" * 60)
log(f"mask_feedback 构建（靶点范围: {FEEDBACK_SCOPE}）")
log("=" * 60)

# ============================================================
# 步骤 0：索引重建 + 双对齐校验
# ============================================================
if not NODE_FILE or not GENE_MAP or not METAB_MAP:
    raise SystemExit("错误：缺节点文件或索引映射（先跑 ECrel 补边脚本矫正索引）")

with open(NODE_FILE, "r", encoding="utf-8") as f:
    all_nodes = [l.strip() for l in f if l.strip()]

gene_nodes  = [n for n in all_nodes if n.startswith("mmu:")]
metab_nodes = [n for n in all_nodes if n.startswith("C")]
gene_to_idx  = {g: i for i, g in enumerate(gene_nodes)}
metab_to_idx = {m: i for i, m in enumerate(metab_nodes)}

with open(GENE_MAP, "r", encoding="utf-8") as f:
    ecrel_gene_map = json.load(f)
with open(METAB_MAP, "r", encoding="utf-8") as f:
    ecrel_metab_map = json.load(f)

bad_g = [g for g in gene_to_idx if ecrel_gene_map.get(g) != gene_to_idx[g]]
bad_m = [m for m in metab_to_idx if ecrel_metab_map.get(m) != metab_to_idx[m]]
if bad_g or bad_m:
    raise SystemExit(f"错误：索引与ECrel不一致（基因{len(bad_g)}个/代谢物{len(bad_m)}个），已停止")
log(f"对齐校验通过: 基因 {len(gene_to_idx)} 唯一 / {len(gene_nodes)} 行, "
    f"代谢物 {len(metab_to_idx)} 唯一 / {len(metab_nodes)} 行")

# ============================================================
# 步骤 1：白名单 ∩ 实际测到的节点
# ============================================================
log("\n" + "=" * 60)
log("步骤1: 信号代谢物白名单检查")
log("=" * 60)
active_sources = {}   # 名称 -> C编号（仅保留在节点列表中的）
for name, cid in WHITELIST.items():
    if cid in metab_to_idx:
        active_sources[name] = cid
        log(f"  {name}({cid}): ✓ 在节点，作为反馈源")
    else:
        log(f"  {name}({cid}): ✗ 未测到，跳过（论文口径：数据决定网络边界）")

if not active_sources:
    raise SystemExit("错误：白名单无一在节点列表，检查节点文件")

# ============================================================
# 步骤 2：确定靶蛋白集合
# ============================================================
log("\n" + "=" * 60)
log("步骤2: 确定靶蛋白集合")
log("=" * 60)

if FEEDBACK_SCOPE == "all":
    target_genes = set(gene_to_idx.keys())
    log(f"靶点=全部基因/蛋白: {len(target_genes)} 个")
else:
    # signaling：从 04xxx 信号通路 KGML 收集基因
    if not KGML_DIR:
        raise SystemExit("错误：signaling 模式需要 kegg_kgml 目录")
    sig_genes = {}   # gene -> set(通路名)
    kgml_files = [f for f in os.listdir(KGML_DIR) if f.endswith((".xml", ".kgml"))]
    for xf in kgml_files:
        # 通路编号在文件名里（如 mmu04150.kgml）：04xxx = 信号转导/信号分子类
        m = re_match = __import__("re").search(r"mmu(\d{5})", xf)
        if not m or not m.group(1).startswith("04"):
            continue
        pathway_id = m.group(0)
        root = ET.parse(os.path.join(KGML_DIR, xf)).getroot()
        for entry in root.findall("entry"):
            if entry.get("type") != "gene":
                continue
            for g in entry.get("name", "").split():   # 多ID拆分
                if g.startswith("mmu:") and g in gene_to_idx:
                    sig_genes.setdefault(g, set()).add(pathway_id)
        log(f"  信号通路 {pathway_id}: 累计信号基因 {len(sig_genes)} 个")
    target_genes = set(sig_genes.keys())
    log(f"靶点=信号通路基因: {len(target_genes)} 个（来自04xxx系列KGML）")

# ============================================================
# 步骤 3：建掩码（行=代谢物 194, 列=蛋白 27999）
# ============================================================
log("\n" + "=" * 60)
log("步骤3: 构建掩码")
log("=" * 60)

n_metabs = len(metab_nodes)
n_genes  = len(gene_nodes)
mask = np.zeros((n_metabs, n_genes), dtype=np.int8)

rows = []
for name, cid in active_sources.items():
    i = metab_to_idx[cid]
    for g in sorted(target_genes):
        j = gene_to_idx[g]
        mask[i, j] = 1
        rows.append({"metabolite": cid, "metabolite_name": name,
                     "protein": g, "scope": FEEDBACK_SCOPE})

n_total = int(mask.sum())
log(f"反馈边总数: {n_total}（{len(active_sources)} 个源 × {len(target_genes)} 个靶）")
log(f"掩码形状: {mask.shape}（代谢物×蛋白），密度: {n_total / mask.size:.4%}")

# ============================================================
# 步骤 4：生物学抽查（已知代谢物感知对）
# ============================================================
log("\n" + "=" * 60)
log("步骤4: 生物学抽查")
log("=" * 60)
# symbol→mmu 简易映射（从节点文件无法拿symbol，这里硬编码几个关键的，来自KEGG注释）
spot = [
    ("C00020", "AMP",  ["mmu:105787", "mmu:108079"], "AMP→AMPK(Prkaa1/2)"),
    ("C00003", "NAD+", ["mmu:93759"],                 "NAD+→SIRT1"),
    ("C00064", "谷氨酰胺", ["mmu:56717"],              "谷氨酰胺→mTOR"),
    ("C00002", "ATP",  ["mmu:56717"],                 "ATP→mTOR（能量充足信号）"),  # noqa
]
for cid, name, gene_ids, note in spot:
    if cid not in metab_to_idx:
        log(f"  {name}: 不在节点，跳过")
        continue
    i = metab_to_idx[cid]
    hits = [g for g in gene_ids if g in gene_to_idx and mask[i, gene_to_idx[g]] == 1]
    log(f"  {name}({cid})→{note}: {'✓ 连上 '+str(len(hits))+'/'+str(len(gene_ids)) if hits else '✗ 未连上（靶基因不在信号基因集或节点内）'}")

# ============================================================
# 步骤 5：保存
# ============================================================
for fp in [MASK_OUT, EDGE_CSV]:
    if os.path.exists(fp):
        bak = fp.replace(".npy", "_备份.npy").replace(".csv", "_备份.csv")
        if not os.path.exists(bak):
            os.rename(fp, bak)
            log(f"旧文件备份: {bak}")

np.save(MASK_OUT, mask)
pd.DataFrame(rows).to_csv(EDGE_CSV, index=False, encoding="utf-8-sig")
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log(f"\n已保存: {MASK_OUT} ({os.path.getsize(MASK_OUT)/1e6:.1f}MB)")
log(f"已保存: {EDGE_CSV}（{len(rows)}行）, {LOG_FILE}")
log("完成。注意：int8格式，训练时 .astype(np.float32)；掩码方向为 行=代谢物→列=蛋白。")