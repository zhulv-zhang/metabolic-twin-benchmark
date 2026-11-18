#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECrel掩码构建完整修正版
======================
解决之前代码中的4个核心问题，构建酶(mmu:xxxxx) → 代谢物(Cxxxxx)的掩码矩阵。

【问题1-多ID拆分Bug】★致命★
  原代码: name_attr.split()[0] 只取第一个ID
  后果: 如 entry name="mmu:11651 mmu:11652 mmu:11653" 只保留mmu:11651(Aldoa)，
        丢弃mmu:11652(Aldob)和mmu:11653(Aldoc)，导致大量有效边丢失。
  修复: 拆分所有ID，每个ID独立建立entry→kegg_id映射。

【问题2-反应方向缺失】★重要★
  原代码: 只提取ECrel关系，不查代谢物是底物还是产物。
  后果: 模型不知道代谢物是被消耗(-1)还是生成(+1)，训练时权重符号自由学。
  修复: 从KGML的<reaction>标签解析底物/产物；同时用KEGG API查EC→反应方向补充。

【问题3-无酶复合物标记】
  原代码: 不识别同EC编号的基因家族。
  后果: Aldoa/Aldob/Aldoc催化同一反应但参数独立学习，增加参数量且生物学不合理。
  修复: 通过KEGG API查询每个基因的EC编号，同EC的基因标记为共享参数家族。

【问题4-无KEGG API方向补充】
  原代码: 仅依赖KGML本地文件。
  后果: KGML中部分反应方向标注不完整（如可逆反应标为"reversible"）。
  修复: 调用KEGG REST API (link/ec, link/reaction) 查询基因→EC→反应→方向。

输出文件:
  - mask_ECrel.npy          : (n_proteins × n_metabolites) 0/1掩码矩阵
  - direction_mask.npy      : (n_proteins × n_metabolites) -1/0/+1方向矩阵
  - enzyme_families.json    : 酶复合物家族分组（参数共享标记）
  - mask_ECrel_edges.csv    : 边列表（便于人工审查）
  - build_log.txt           : 构建日志

作者: 修正版
日期: 2026-08-30
"""

import os
import sys
import time
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import requests
import pandas as pd
import numpy as np

# ==================== 配置区 ====================
DATA_DIR = "GSE166336_小鼠组学数据"       # 数据主目录
KGML_DIR = "kegg_kgml"                     # KGML文件目录
NODE_FILE = os.path.join(DATA_DIR, "all_kegg_nodes.txt")  # 节点列表
OUTPUT_DIR = DATA_DIR                      # 输出目录

# KEGG REST API 基础URL（用于查询EC编号、反应方向）
KEGG_API_BASE = "https://rest.kegg.jp"
API_DELAY = 0.5  # API请求间隔(秒)，避免被封

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(KGML_DIR, exist_ok=True)

# 日志文件
LOG_FILE = os.path.join(OUTPUT_DIR, "build_log.txt")
log_lines = []

def log(msg: str, print_console: bool = True):
    """记录日志到文件和控制台"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    log_lines.append(line)
    if print_console:
        print(line)

# ==================== 步骤1: 读取节点列表 ====================
log("=" * 60)
log("步骤1: 读取节点列表")
log("=" * 60)

with open(NODE_FILE, "r", encoding="utf-8") as f:
    all_nodes = [line.strip() for line in f if line.strip()]

# 分离基因节点和代谢物节点
gene_nodes = [n for n in all_nodes if n.startswith("mmu:")]
metab_nodes = [n for n in all_nodes if n.startswith("C")]

# 建立索引映射
gene_to_idx = {g: i for i, g in enumerate(gene_nodes)}
metab_to_idx = {m: i for i, m in enumerate(metab_nodes)}

log(f"总节点数: {len(all_nodes)}")
log(f"基因节点数: {len(gene_nodes)}")
log(f"代谢物节点数: {len(metab_nodes)}")

if len(gene_nodes) == 0 or len(metab_nodes) == 0:
    log("错误: 基因或代谢物节点数为0，请检查节点列表格式！", True)
    sys.exit(1)

# ==================== 步骤2: 解析KGML - 修正多ID拆分Bug ====================
log("")
log("=" * 60)
log("步骤2: 解析KGML文件（修正多ID拆分Bug）")
log("=" * 60)

kgml_files = [f for f in os.listdir(KGML_DIR) if f.endswith(".kgml")]
log(f"找到 {len(kgml_files)} 个KGML文件")

# 数据结构
# entry_to_kegg: entry_id -> [kegg_id1, kegg_id2, ...]  ★修正1: 存列表而非单个字符串
entry_to_kegg: Dict[str, List[str]] = defaultdict(list)

# reaction_info: reaction_id -> {"substrates": [Cxxxxx, ...], "products": [Cxxxxx, ...], "type": "irreversible|reversible"}
reaction_info: Dict[str, Dict] = {}

# ec_to_genes: ec_number -> [mmu:xxxxx, ...]  （用于酶复合物检测）
ec_to_genes: Dict[str, List[str]] = defaultdict(list)

# 统计
total_entries_parsed = 0
total_multi_id_entries = 0
total_reactions_parsed = 0

for fname in kgml_files:
    filepath = os.path.join(KGML_DIR, fname)
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception as e:
        log(f"  警告: 解析 {fname} 失败: {e}")
        continue

    # --- 2.1 解析entry节点（修正多ID拆分）---
    for entry in root.findall("entry"):
        entry_id = entry.get("id")
        name_attr = entry.get("name", "")
        entry_type = entry.get("type", "")

        if not name_attr or not entry_id:
            continue

        total_entries_parsed += 1

        # ★修正1核心: 拆分所有ID，而非只取第一个
        # 原bug: kegg_id = name_attr.split()[0]  ← 只取第一个！
        # 修正后: 遍历所有ID，每个有效ID都存入映射
        all_ids = name_attr.strip().split()
        valid_ids = [k for k in all_ids if k.startswith(("mmu:", "C", "ec:"))]

        if len(valid_ids) > 1:
            total_multi_id_entries += 1

        for kegg_id in valid_ids:
            entry_to_kegg[entry_id].append(kegg_id)

            # 收集EC编号（用于后续酶复合物检测）
            if kegg_id.startswith("ec:"):
                # EC编号通常出现在type="enzyme"的entry中，但我们也从name里收集
                pass

    # --- 2.2 解析reaction标签（提取底物/产物方向）---
    for reaction in root.findall("reaction"):
        reaction_name = reaction.get("name", "")  # 如 "rn:R00710"
        reaction_type = reaction.get("type", "reversible")  # irreversible | reversible

        substrates = []
        products = []

        for substrate in reaction.findall("substrate"):
            sub_name = substrate.get("name", "")
            # 格式通常是 "cpd:C00031"，提取C编号
            if sub_name.startswith("cpd:"):
                substrates.append(sub_name.replace("cpd:", ""))
            elif sub_name.startswith("C"):
                substrates.append(sub_name)

        for product in reaction.findall("product"):
            prod_name = product.get("name", "")
            if prod_name.startswith("cpd:"):
                products.append(prod_name.replace("cpd:", ""))
            elif prod_name.startswith("C"):
                products.append(prod_name)

        if reaction_name:
            reaction_info[reaction_name] = {
                "substrates": substrates,
                "products": products,
                "type": reaction_type,
                "source_file": fname
            }
            total_reactions_parsed += 1

log(f"  解析entry节点: {total_entries_parsed} 个")
log(f"  其中多ID entry: {total_multi_id_entries} 个（之前这些被错误丢弃了！）")
log(f"  解析reaction: {total_reactions_parsed} 个")
log(f"  唯一entry_id映射: {len(entry_to_kegg)} 个")

# ==================== 步骤3: 提取ECrel关系 ====================
log("")
log("=" * 60)
log("步骤3: 提取ECrel关系（酶→代谢物）")
log("=" * 60)

# 存储提取的边: (gene_kegg_id, metab_kegg_id, reaction_id, source_file)
raw_edges: List[Tuple[str, str, str, str]] = []

for fname in kgml_files:
    filepath = os.path.join(KGML_DIR, fname)
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception:
        continue

    for rel in root.findall("relation"):
        rel_type = rel.get("type")
        if rel_type != "ECrel":
            continue

        src_entry = rel.get("entry1")
        tgt_entry = rel.get("entry2")

        # 获取两端的所有KEGG ID（★修正1: 遍历所有组合）
        src_ids = entry_to_kegg.get(src_entry, [])
        tgt_ids = entry_to_kegg.get(tgt_entry, [])

        if not src_ids or not tgt_ids:
            continue

        # 遍历所有组合（笛卡尔积），确保不遗漏任何可能的连接
        for src_kegg in src_ids:
            for tgt_kegg in tgt_ids:
                # ECrel方向: 基因(mmu:) → 代谢物(C) 或 代谢物(C) → 基因(mmu:)
                # 我们双向都记录，后续根据reaction方向确定最终方向
                if src_kegg.startswith("mmu:") and tgt_kegg.startswith("C"):
                    raw_edges.append((src_kegg, tgt_kegg, "", fname))
                elif src_kegg.startswith("C") and tgt_kegg.startswith("mmu:"):
                    raw_edges.append((tgt_kegg, src_kegg, "", fname))

log(f"  原始ECrel边（去重前）: {len(raw_edges)} 条")

# 去重
unique_edges = list(set((s, t) for s, t, _, _ in raw_edges))
log(f"  去重后ECrel边: {len(unique_edges)} 条")

# ==================== 步骤4: 从reaction标签推断方向 ====================
log("")
log("=" * 60)
log("步骤4: 从KGML <reaction>标签推断代谢物方向")
log("=" * 60)

# direction_map: (gene_id, metab_id) -> {-1, 0, +1}
# -1: 底物（被酶消耗）  +1: 产物（被酶生成）  0: 未知/可逆
direction_map: Dict[Tuple[str, str], int] = {}

# 建立 reaction_id -> [关联的entry_id] 映射（用于关联ECrel和reaction）
reaction_to_entries: Dict[str, List[str]] = defaultdict(list)
for fname in kgml_files:
    filepath = os.path.join(KGML_DIR, fname)
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception:
        continue

    # 找到reaction对应的entry id
    for entry in root.findall("entry"):
        entry_id = entry.get("id")
        name_attr = entry.get("name", "")
        entry_type = entry.get("type", "")

        if entry_type == "reaction" and name_attr:
            # reaction entry的name可能是 "rn:R00710" 或 "rn:R00710 rn:R00711"
            for rn in name_attr.split():
                if rn.startswith("rn:"):
                    reaction_to_entries[rn].append(entry_id)

# 关联ECrel关系与reaction方向
for fname in kgml_files:
    filepath = os.path.join(KGML_DIR, fname)
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception:
        continue

    for rel in root.findall("relation"):
        if rel.get("type") != "ECrel":
            continue

        src_entry = rel.get("entry1")
        tgt_entry = rel.get("entry2")

        # 查找关联的reaction
        # ECrel通常连接 gene entry 和 reaction entry
        # 我们需要找到这个reaction的底物/产物信息

        # 确定哪个entry是reaction
        reaction_entries = []
        gene_entries = []

        for eid in [src_entry, tgt_entry]:
            ids = entry_to_kegg.get(eid, [])
            for k in ids:
                if k.startswith("rn:"):
                    reaction_entries.append(k)
                elif k.startswith("mmu:"):
                    gene_entries.extend([k])

        if not reaction_entries or not gene_entries:
            continue

        for rn in reaction_entries:
            if rn not in reaction_info:
                continue
            rinfo = reaction_info[rn]
            substrates = rinfo["substrates"]
            products = rinfo["products"]
            rtype = rinfo["type"]

            # 遍历该ECrel涉及的所有基因和代谢物
            for gene_id in gene_entries:
                # 查找该基因对应的entry中的代谢物
                for eid in [src_entry, tgt_entry]:
                    for kegg_id in entry_to_kegg.get(eid, []):
                        if kegg_id.startswith("C"):
                            # 判断代谢物是底物还是产物
                            if kegg_id in substrates:
                                direction_map[(gene_id, kegg_id)] = -1
                            elif kegg_id in products:
                                direction_map[(gene_id, kegg_id)] = +1
                            elif rtype == "reversible":
                                direction_map[(gene_id, kegg_id)] = 0

log(f"  从reaction标签推断出方向: {len(direction_map)} 条")
direction_stats = defaultdict(int)
for d in direction_map.values():
    direction_stats[d] += 1
log(f"    底物(-1): {direction_stats.get(-1, 0)} 条")
log(f"    产物(+1): {direction_stats.get(1, 0)} 条")
log(f"    可逆/未知(0): {direction_stats.get(0, 0)} 条")

# ==================== 步骤5: KEGG API查询补充（EC编号 + 反应方向）====================
log("")
log("=" * 60)
log("步骤5: 通过KEGG REST API查询补充EC编号和方向")
log("=" * 60)

# 缓存文件（避免重复查询）
CACHE_FILE = os.path.join(OUTPUT_DIR, "kegg_api_cache.json")
kegg_cache: Dict = {}

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        kegg_cache = json.load(f)
    log(f"  加载缓存: {len(kegg_cache)} 条")

def kegg_api_query(url: str, max_retries: int = 3) -> Optional[str]:
    """带缓存和重试的KEGG API查询"""
    if url in kegg_cache:
        return kegg_cache[url]

    for attempt in range(max_retries):
        try:
            time.sleep(API_DELAY)  # 礼貌间隔
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                kegg_cache[url] = resp.text
                return resp.text
            elif resp.status_code == 404:
                kegg_cache[url] = None
                return None
            else:
                log(f"    API返回状态码 {resp.status_code}, 重试 {attempt+1}/{max_retries}")
                time.sleep(API_DELAY * 2)
        except Exception as e:
            log(f"    API请求失败: {e}, 重试 {attempt+1}/{max_retries}")
            time.sleep(API_DELAY * 2)

    kegg_cache[url] = None
    return None

def save_cache():
    """保存API缓存"""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(kegg_cache, f, ensure_ascii=False, indent=2)

# 5.1 查询每个基因的EC编号（用于酶复合物检测）
log("  5.1 查询基因→EC编号...")

# 收集所有涉及的基因（去重）
all_genes_in_edges = list(set([g for g, _ in unique_edges]))
log(f"    涉及基因数: {len(all_genes_in_edges)}")

gene_to_ec: Dict[str, List[str]] = defaultdict(list)

# 批量查询（KEGG API支持批量，每批最多10个）
batch_size = 10
for i in range(0, len(all_genes_in_edges), batch_size):
    batch = all_genes_in_edges[i:i+batch_size]
    batch_str = "+".join(batch)
    url = f"{KEGG_API_BASE}/link/ec/{batch_str}"

    result = kegg_api_query(url)
    if result:
        # 解析返回: "mmu:11651	ec:4.1.2.13"
        for line in result.strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                gene_id, ec_num = parts[0], parts[1]
                if gene_id in gene_to_idx and ec_num.startswith("ec:"):
                    gene_to_ec[gene_id].append(ec_num)
                    ec_to_genes[ec_num].append(gene_id)

    if (i // batch_size + 1) % 5 == 0:
        save_cache()
        log(f"    进度: {min(i+batch_size, len(all_genes_in_edges))}/{len(all_genes_in_edges)}")

save_cache()
log(f"    查询到EC编号的基因: {len(gene_to_ec)} 个")

# 5.2 查询EC→反应方向补充
log("  5.2 查询EC→反应方向补充...")

# 收集所有已知的EC编号
all_ecs = list(ec_to_genes.keys())
log(f"    涉及EC编号: {len(all_ecs)} 个")

# EC→反应→方向映射
ec_reaction_direction: Dict[str, Dict[str, Dict]] = {}

for ec_num in all_ecs:
    # 查询该EC催化的所有反应
    url = f"{KEGG_API_BASE}/link/reaction/{ec_num}"
    result = kegg_api_query(url)

    if not result:
        continue

    for line in result.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) == 2:
            _, rn = parts[0], parts[1]
            if not rn.startswith("rn:"):
                continue

            # 查询反应详情（获取底物/产物）
            url2 = f"{KEGG_API_BASE}/get/{rn}"
            result2 = kegg_api_query(url2)

            if not result2:
                continue

            # 解析反应方程: "EQUATION    C00031 + C00002 <=> C00092 + C00008"
            substrates = []
            products = []
            equation_found = False

            for rline in result2.split("\n"):
                if rline.startswith("EQUATION"):
                    equation = rline.replace("EQUATION", "").strip()
                    equation_found = True

                    # 分离左右两边
                    if " <=> " in equation:
                        left, right = equation.split(" <=> ")
                        reversible = True
                    elif " => " in equation:
                        left, right = equation.split(" => ")
                        reversible = False
                    elif " <= " in equation:
                        right, left = equation.split(" <= ")
                        reversible = False
                    else:
                        continue

                    # 提取C编号（简单正则）
                    import re
                    substrates = re.findall(r"C\d{5}", left)
                    products = re.findall(r"C\d{5}", right)

                    # 记录到ec_reaction_direction
                    if ec_num not in ec_reaction_direction:
                        ec_reaction_direction[ec_num] = {}
                    ec_reaction_direction[ec_num][rn] = {
                        "substrates": substrates,
                        "products": products,
                        "reversible": reversible
                    }
                    break

save_cache()
log(f"    从API补充反应方向: {sum(len(v) for v in ec_reaction_direction.values())} 条")

# 5.3 用API结果补充direction_map
api_direction_added = 0
for ec_num, reactions in ec_reaction_direction.items():
    genes = ec_to_genes.get(ec_num, [])
    for rn, rinfo in reactions.items():
        for gene_id in genes:
            for metab_id in rinfo["substrates"]:
                if metab_id in metab_to_idx:
                    key = (gene_id, metab_id)
                    if key not in direction_map:
                        direction_map[key] = -1
                        api_direction_added += 1
            for metab_id in rinfo["products"]:
                if metab_id in metab_to_idx:
                    key = (gene_id, metab_id)
                    if key not in direction_map:
                        direction_map[key] = +1
                        api_direction_added += 1

log(f"    API补充新方向: {api_direction_added} 条")

# ==================== 步骤6: 过滤有效边并构建掩码矩阵 ====================
log("")
log("=" * 60)
log("步骤6: 过滤有效边并构建掩码矩阵")
log("=" * 60)

# 只保留两端都在节点列表中的边
valid_edges = []
invalid_reasons = defaultdict(int)

for gene_id, metab_id in unique_edges:
    if gene_id not in gene_to_idx:
        invalid_reasons["gene_not_in_list"] += 1
        continue
    if metab_id not in metab_to_idx:
        invalid_reasons["metab_not_in_list"] += 1
        continue
    valid_edges.append((gene_id, metab_id))

log(f"  有效边（两端都在节点列表中）: {len(valid_edges)} 条")
log(f"  过滤原因统计:")
for reason, count in invalid_reasons.items():
    log(f"    {reason}: {count}")

if len(valid_edges) == 0:
    log("错误: 有效边数为0！请检查节点列表与KGML文件的ID格式是否匹配。", True)
    log("常见原因: 节点列表用symbol(Akt1)但KGML用mmu:11651，或代谢物编号不一致。", True)
    sys.exit(1)

# 初始化掩码矩阵
n_genes = len(gene_nodes)
n_metabs = len(metab_nodes)

mask_ECrel = np.zeros((n_genes, n_metabs), dtype=np.float32)
direction_mask = np.zeros((n_genes, n_metabs), dtype=np.int8)

for gene_id, metab_id in valid_edges:
    i = gene_to_idx[gene_id]
    j = metab_to_idx[metab_id]
    mask_ECrel[i, j] = 1.0

    # 查找方向
    key = (gene_id, metab_id)
    if key in direction_map:
        direction_mask[i, j] = direction_map[key]
    else:
        direction_mask[i, j] = 0  # 未知

# 统计
log(f"  掩码矩阵形状: {mask_ECrel.shape} (基因 × 代谢物)")
log(f"  非零元素数: {int(mask_ECrel.sum())}")
log(f"  掩码密度: {mask_ECrel.sum() / (n_genes * n_metabs):.6%}")

direction_counts = defaultdict(int)
for i in range(n_genes):
    for j in range(n_metabs):
        if mask_ECrel[i, j] > 0:
            direction_counts[int(direction_mask[i, j])] += 1

log(f"  方向统计:")
log(f"    底物(-1): {direction_counts.get(-1, 0)} 条")
log(f"    产物(+1): {direction_counts.get(1, 0)} 条")
log(f"    未知(0):  {direction_counts.get(0, 0)} 条")

# ==================== 步骤7: 酶复合物检测（参数共享标记）====================
log("")
log("=" * 60)
log("步骤7: 酶复合物检测（同EC基因家族 → 参数共享）")
log("=" * 60)

enzyme_families = []
family_id = 0

for ec_num, genes in ec_to_genes.items():
    # 只保留在节点列表中的基因
    valid_genes = [g for g in genes if g in gene_to_idx]
    if len(valid_genes) >= 2:
        family_id += 1
        family = {
            "family_id": family_id,
            "ec_number": ec_num,
            "genes": valid_genes,
            "gene_symbols": [g.replace("mmu:", "") for g in valid_genes],
            "n_genes": len(valid_genes),
            "description": f"EC {ec_num} 催化的酶复合物/同工酶家族"
        }
        enzyme_families.append(family)

log(f"  检测到酶复合物家族: {len(enzyme_families)} 个")
for fam in enzyme_families[:10]:  # 只打印前10个
    log(f"    家族{fam['family_id']}: {fam['ec_number']} -> {', '.join(fam['gene_symbols'])} ({fam['n_genes']}个基因)")

if len(enzyme_families) > 10:
    log(f"    ... 还有 {len(enzyme_families)-10} 个家族")

# ==================== 步骤8: 保存结果 ====================
log("")
log("=" * 60)
log("步骤8: 保存结果文件")
log("=" * 60)

# 8.1 保存掩码矩阵
np.save(os.path.join(OUTPUT_DIR, "mask_ECrel.npy"), mask_ECrel)
log(f"  ✅ mask_ECrel.npy 已保存: {mask_ECrel.shape}")

np.save(os.path.join(OUTPUT_DIR, "direction_mask.npy"), direction_mask)
log(f"  ✅ direction_mask.npy 已保存: {direction_mask.shape}")

# 8.2 保存酶复合物信息
with open(os.path.join(OUTPUT_DIR, "enzyme_families.json"), "w", encoding="utf-8") as f:
    json.dump(enzyme_families, f, ensure_ascii=False, indent=2)
log(f"  ✅ enzyme_families.json 已保存: {len(enzyme_families)} 个家族")

# 8.3 保存边列表CSV
edge_records = []
for gene_id, metab_id in valid_edges:
    i = gene_to_idx[gene_id]
    j = metab_to_idx[metab_id]
    direction = int(direction_mask[i, j])
    direction_str = { -1: "substrate", 0: "unknown", 1: "product" }.get(direction, "unknown")

    edge_records.append({
        "source_gene": gene_id,
        "source_symbol": gene_id.replace("mmu:", ""),
        "target_metabolite": metab_id,
        "direction_code": direction,
        "direction_label": direction_str
    })

edges_df = pd.DataFrame(edge_records)
edges_df.to_csv(os.path.join(OUTPUT_DIR, "mask_ECrel_edges.csv"), index=False, encoding="utf-8")
log(f"  ✅ mask_ECrel_edges.csv 已保存: {len(edge_records)} 条")

# 8.4 保存构建日志
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"  ✅ build_log.txt 已保存")

# 8.5 保存节点索引映射（方便后续使用）
with open(os.path.join(OUTPUT_DIR, "gene_index_map.json"), "w", encoding="utf-8") as f:
    json.dump(gene_to_idx, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, "metabolite_index_map.json"), "w", encoding="utf-8") as f:
    json.dump(metab_to_idx, f, ensure_ascii=False, indent=2)
log(f"  ✅ 节点索引映射已保存")

# ==================== 最终统计 ====================
log("")
log("=" * 60)
log("构建完成！最终统计")
log("=" * 60)
log(f"基因节点数: {n_genes}")
log(f"代谢物节点数: {n_metabs}")
log(f"ECrel有效边数: {len(valid_edges)}")
log(f"掩码密度: {len(valid_edges) / (n_genes * n_metabs):.6%}")
log(f"方向已知率: {(direction_counts.get(-1,0) + direction_counts.get(1,0)) / len(valid_edges) * 100:.1f}%")
log(f"酶复合物家族: {len(enzyme_families)} 个")
log("")
log("输出文件列表:")
log(f"  {OUTPUT_DIR}/mask_ECrel.npy")
log(f"  {OUTPUT_DIR}/direction_mask.npy")
log(f"  {OUTPUT_DIR}/enzyme_families.json")
log(f"  {OUTPUT_DIR}/mask_ECrel_edges.csv")
log(f"  {OUTPUT_DIR}/build_log.txt")
log(f"  {OUTPUT_DIR}/kegg_api_cache.json")
log("")
log("【使用说明】")
log("1. mask_ECrel[i,j] = 1 表示基因i（酶）与代谢物j存在催化关系")
log("2. direction_mask[i,j] = -1(底物/消耗) / +1(产物/生成) / 0(未知)")
log("3. enzyme_families.json中同家族的基因应共享ODE权重参数")
log("4. 若有效边数仍为0，请检查节点列表格式是否与KGML一致（mmu: vs C编号）")
log("=" * 60)