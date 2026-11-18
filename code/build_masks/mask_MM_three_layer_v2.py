#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_MM（代谢物→代谢物掩码）三层构建版
========================================
【功能】构建 代谢物×代谢物 的 0/1 掩码矩阵，作为 ODE 代谢层
        W_MM ⊙ mask_MM 的硬约束：掩码=1 的位置才允许学习权重，
        掩码=0 的位置永远为零。目的是把 194×194=37636 个自由参数
        压缩到 1500-3000 个，使 10 样本可训练、不过拟合。

【为什么不用 KEGG 反应方程建边】
        实测的 194 个代谢物多为通路"终点"（G6P、乳酸、柠檬酸等），
        它们之间隔着大量未测量的中间产物（ATP、NADH、烯醇式丙酮酸等）。
        要求"底物和产物都在 194 个节点内"才能连边 → 结构性连不上（0 条边）。
        因此放弃精确化学计量（S 矩阵同步放弃），改用三层轻量证据。

【三层证据（取并集，任一层认为有关即允许连接）】
  层A 通路共现掩码（必须做）：查 KEGG REST API，两个代谢物同属
      至少一条 KEGG 通路 → mask=1。生物学含义：同通路代谢物经多步
      反应耦合，在小时级时间尺度上"模块内协同变化"比精确化学计量更稳。
  层B 数据相关掩码（辅助，可选）：用代谢组时序数据算两两 Pearson
      相关，|r|>0.5 → mask=1。只用于压缩参数空间，不作因果证据。
      需要代谢组数据文件；文件不存在时自动跳过并提示。
  层C 模块先验掩码（审稿人保险）：手动标注糖酵解/TCA/磷酸戊糖
      核心代谢物（KEGG C编号），模块内全连接。防止 KEGG 注释不全
      或数据噪声把常识性连接漏掉。

【输入】
  GSE166336_小鼠组学数据/all_kegg_nodes.txt  节点列表（代谢物为 Cxxxxx 格式）
      ※该文件含重复行和 "-" 分隔行，本脚本 v2 已兼容（详见 Step 0 注释）：
        保留重复以与 ECrel 掩码(27999×194)列顺序对齐，并用
        metabolite_index_map.json 做对齐校验，不一致会报错停止。
  （可选）GSE166336_小鼠组学数据/metabolite_data.csv
      行=样本（10样本×5时间点可全部展开为行），列=代谢物C编号，值=浓度/丰度

【输出】（均保存在 DATA_DIR 下）
  mask_metabolite2metabolite.npy   (194×194) 0/1 掩码，ODE 直接使用
  mask_MM_edges.csv                边列表（source, target, layer），人工审查用
  kegg_pathway_cache.json          KEGG 查询缓存，重复运行不重复请求
  build_MM_log.txt                 构建日志（每层边数、总数、密度）

【ODE 接入方式】（供主模型代码参考）
  dz_M/dt = -α_M * z_M + (W_MP ⊙ mask_MP) @ σ(z_P*)
                         + (W_MM ⊙ mask_MM) @ σ(z_M)   ← 本脚本产出 mask_MM

作者: mask_MM 三层构建版
日期: 2026-08-31
"""

import os
import re
import sys
import time
import json
import requests
import numpy as np
import pandas as pd

# ==================== 配置区 ====================
DATA_DIR = "GSE166336_小鼠组学数据"          # 数据主目录
NODE_FILE = os.path.join(DATA_DIR, "all_kegg_nodes.txt")   # 节点列表
METAB_DATA_FILE = os.path.join(DATA_DIR, "metabolite_data.csv")  # 可选：代谢组数据
CACHE_FILE = os.path.join(DATA_DIR, "kegg_pathway_cache.json")   # KEGG缓存
LOG_FILE = os.path.join(DATA_DIR, "build_MM_log.txt")            # 日志

CORR_THRESHOLD = 0.5   # 层B：相关性阈值 |r|>0.5 才允许连接
API_DELAY = 0.2        # KEGG 请求间隔（秒），防止限流
MIN_SHARED_PATHWAYS = 1  # 层A：至少共享几条"具体"通路才连边（太密可调大到2）

# 【v3 新增】KEGG 全局/概览图排除规则
# link/pathway 返回的通路里混有"全局图/概览图"（global & overview maps）：
#   mmu01100 Metabolic pathways（一张图含数百代谢物）
#   mmu01110 Biosynthesis of secondary metabolites
#   mmu01120 Microbial metabolism in diverse environments
#   mmu01200 Carbon metabolism 等 012xx 系列概览图
# 它们会把"同通路共现"炸成全连接（实测：不排除时层A密度69%，失去约束意义）。
# 因此层A只使用具体通路（排除 mmu011xx / mmu012xx）。
def is_global_map(pathway_id):
    """判断是否为KEGG全局/概览图（这类图不含具体机制信息）
    注意：代谢物查询返回的通路带 map 前缀（map01100），基因查询才带 mmu 前缀，
    因此取数字部分判断，兼容两种前缀。"""
    m = re.search(r"(\d{5})", pathway_id)
    if not m:
        return False
    num = m.group(1)
    return num.startswith("011") or num.startswith("012")

# ---------- 层C：模块先验（KEGG C编号，手动标注，可按数据实际增删） ----------
MODULES = {
    # 糖酵解：G6P, F6P, DHAP, 3PG, 2PG, PEP, 乳酸
    "糖酵解": ["C00092", "C00085", "C00111", "C00118", "C00197", "C00631", "C00186"],
    # TCA循环：柠檬酸, 异柠檬酸, α-酮戊二酸, 琥珀酰CoA, 琥珀酸, 延胡索酸, 苹果酸
    "TCA循环": ["C00158", "C00311", "C00026", "C00091", "C00042", "C05381", "C00149"],
    # 磷酸戊糖：G6P, 6-磷酸葡萄糖酸, 核酮糖5P, 核糖5P, 木酮糖5P, 景天庚酮糖7P, 赤藓糖4P, NADPH
    "磷酸戊糖": ["C00092", "C00345", "C00199", "C00117", "C00231", "C05382", "C00279", "C00005"],
}

# ==================== 日志工具 ====================
log_lines = []

def log(msg):
    """同时写日志文件并打印到控制台"""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    log_lines.append(line)
    print(line)

os.makedirs(DATA_DIR, exist_ok=True)

# ==================== Step 0: 读取节点列表 ====================
log("=" * 60)
log("Step 0: 读取节点列表")
log("=" * 60)

with open(NODE_FILE, "r", encoding="utf-8") as f:
    all_nodes = [line.strip() for line in f if line.strip()]

# 代谢物节点：KEGG 化合物编号，以 "C" 开头（如 C00031=葡萄糖）
# 【v2 修改说明】节点文件 all_kegg_nodes.txt 存在两个问题（2026-08-31 发现）：
#   1) 含 8 行 "-" 分隔符（疑为4个组学映射表拼接时分段标记）——不以 C 开头，
#      下面的 startswith("C") 过滤会自动跳过，无需特殊处理；
#   2) 含重复行（194 行代谢物只有 193 个唯一ID）——此处【保留重复】，
#      因为 ECrel 掩码(27999×194)的列顺序就是按这个文件顺序建的，
#      MM 掩码必须保持同样的 194 列顺序才能与 ECrel 列对齐；
#      索引规则与主脚本一致：字典保留最后出现位置。
#   → 是否去重，等六张掩码齐后统一决策（见 方案与进度文档 八-注意点B）。
metab_nodes = [n for n in all_nodes if n.startswith("C")]
metab_to_idx = {m: i for i, m in enumerate(metab_nodes)}
n = len(metab_nodes)

log(f"总节点数: {len(all_nodes)}")
log(f"代谢物节点数: {n}（唯一ID: {len(metab_to_idx)}）")

# 重复诊断：报告哪些代谢物编号出现了多次
from collections import Counter
dup = [c for c, k in Counter(metab_nodes).items() if k > 1]
if dup:
    log(f"  [诊断] 重复代谢物编号({len(dup)}个): {dup} —— 保留以对齐ECrel掩码列")

# 与 ECrel 掩码列索引对齐校验：
# 读取补边脚本矫正后重新保存的 metabolite_index_map.json（193个唯一ID→列号），
# 要求本脚本的索引与它完全一致——不一致说明列顺序错位，必须停下检查
_map_file = os.path.join(DATA_DIR, "metabolite_index_map.json")
if os.path.exists(_map_file):
    with open(_map_file, "r", encoding="utf-8") as f:
        ecrel_metab_map = json.load(f)
    mismatch = [c for c in metab_to_idx
                if c not in ecrel_metab_map or ecrel_metab_map[c] != metab_to_idx[c]]
    if mismatch:
        log(f"  错误: {len(mismatch)} 个代谢物与ECrel列索引不一致: {mismatch[:5]}")
        log("  已停止：请先检查 metabolite_index_map.json 是否为ECrel补边脚本矫正后的版本。")
        sys.exit(1)
    log(f"  与ECrel掩码列索引对齐校验: 通过（{len(metab_to_idx)} 个唯一代谢物全部一致）")
else:
    log("  警告: 未找到 metabolite_index_map.json，跳过对齐校验（建议先跑ECrel补边脚本）")

if n == 0:
    log("错误: 代谢物节点数为0，请检查节点列表格式（应为 Cxxxxx）！")
    sys.exit(1)

# ==================== Step 1: 查询每个代谢物所属的 KEGG 通路（层A原料） ====================
# 用 KEGG REST API: http://rest.kegg.jp/link/pathway/{C编号}
# 返回该代谢物参与的所有通路（如 path:mmu00010=糖酵解）
# 带本地缓存：第二次运行直接读缓存，不重复请求（KEGG限流友好）
log("")
log("=" * 60)
log("Step 1: 查询每个代谢物的 KEGG 通路归属（带缓存）")
log("=" * 60)

cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    log(f"已加载缓存: {len(cache)} 条")

def get_pathways(compound_id):
    """查询某代谢物属于哪些KEGG通路，返回通路编号集合（如 {'mmu00010', ...}）"""
    if compound_id in cache:                      # 命中缓存直接返回
        return set(cache[compound_id])
    url = f"http://rest.kegg.jp/link/pathway/{compound_id}"
    for attempt in range(3):                      # 失败重试3次
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                pathways = set()
                for line in r.text.strip().split("\n"):
                    if line and "\t" in line:
                        # 返回格式: "cpd:C00092\tpath:mmu00010"
                        pathways.add(line.split("\t")[1].replace("path:", ""))
                cache[compound_id] = sorted(pathways)   # 写入缓存（list便于json存）
                time.sleep(API_DELAY)
                return pathways
        except Exception as e:
            log(f"  查询 {compound_id} 第{attempt+1}次失败: {e}")
            time.sleep(1)
    cache[compound_id] = []                       # 彻底失败记空，避免反复重试
    return set()

metab_pathways = {}
for i, cid in enumerate(metab_nodes):
    metab_pathways[cid] = get_pathways(cid)
    if (i + 1) % 50 == 0:                         # 每50个存一次缓存+报进度
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        log(f"  进度 {i+1}/{n}（已缓存 {len(cache)} 条）")

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f)
log(f"通路查询完成，有效通路归属的代谢物: {sum(1 for v in metab_pathways.values() if v)}/{n}")

# 【v3】剔除全局/概览图，只保留具体通路（不过滤原始缓存，只在内存中过滤）
metab_pathways_specific = {
    cid: {p for p in pws if not is_global_map(p)}
    for cid, pws in metab_pathways.items()
}
n_specific = sum(1 for v in metab_pathways_specific.values() if v)
log(f"剔除全局/概览图(mmu011xx/mmu012xx)后，仍有具体通路归属的代谢物: {n_specific}/{n}")
# 诊断：显示被剔除前后的通路数变化最大的几个（帮助确认过滤生效）
_shrunk = sorted(((len(metab_pathways[c]), len(metab_pathways_specific[c]), c)
                  for c in metab_pathways), reverse=True)[:5]
for before, after, c in _shrunk:
    log(f"  [诊断] {c}: 通路数 {before} → {after}")

# ==================== Step 2: 层A——通路共现掩码 ====================
# 规则：两个代谢物只要共享至少一条KEGG通路 → mask=1（对称，模块内双向耦合）
log("")
log("=" * 60)
log("Step 2: 构建层A 通路共现掩码")
log("=" * 60)

mask_pathway = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    for j in range(i + 1, n):
        # 【v3】用剔除全局图后的具体通路计算共现，且要求共享数≥MIN_SHARED_PATHWAYS
        shared = metab_pathways_specific[metab_nodes[i]] & metab_pathways_specific[metab_nodes[j]]
        if len(shared) >= MIN_SHARED_PATHWAYS:
            mask_pathway[i, j] = 1.0
            mask_pathway[j, i] = 1.0

n_pathway = int(mask_pathway.sum())
log(f"层A 通路共现边数（含双向）: {n_pathway}")

# ==================== Step 3: 层B——数据相关性掩码（可选） ====================
# 规则：用代谢组时序数据算两两Pearson相关，|r|>0.5 → mask=1
# 注意：只用于压缩参数空间，不作为因果证据；数据文件缺失时自动跳过
log("")
log("=" * 60)
log("Step 3: 构建层B 数据相关性掩码（可选）")
log("=" * 60)

mask_corr = np.zeros((n, n), dtype=np.float32)
if os.path.exists(METAB_DATA_FILE):
    df = pd.read_csv(METAB_DATA_FILE, index_col=0)
    # 列名需与节点列表的C编号对齐；只保留在节点列表中的列，缺失列补说明
    cols = [c for c in df.columns if c in metab_to_idx]
    log(f"代谢组数据: {df.shape[0]} 样本 × {len(cols)} 个匹配代谢物列")
    if len(cols) >= 2:
        corr = np.corrcoef(df[cols].values.T)     # 代谢物×代谢物 相关矩阵
        corr = np.nan_to_num(corr, nan=0.0)       # 常数列会产生NaN，置0
        for a, ca in enumerate(cols):
            for b, cb in enumerate(cols):
                if a != b and abs(corr[a, b]) > CORR_THRESHOLD:
                    mask_corr[metab_to_idx[ca], metab_to_idx[cb]] = 1.0
    else:
        log("警告: 匹配到的代谢物列不足2个，层B跳过")
else:
    log(f"未找到 {METAB_DATA_FILE}，层B跳过（稍后可把数据放到该路径重跑）")

n_corr = int(mask_corr.sum())
log(f"层B 数据相关边数（含双向）: {n_corr}")

# ==================== Step 4: 层C——模块先验掩码 ====================
# 规则：手动标注的核心模块（糖酵解/TCA/磷酸戊糖）内部两两全连接
# 作用：审稿人保险——即使KEGG注释不全、数据噪声大，常识性连接也必须在
log("")
log("=" * 60)
log("Step 4: 构建层C 模块先验掩码")
log("=" * 60)

mask_module = np.zeros((n, n), dtype=np.float32)
for module_name, cids in MODULES.items():
    idxs = [metab_to_idx[c] for c in cids if c in metab_to_idx]  # 只取在194节点里的
    missing = [c for c in cids if c not in metab_to_idx]
    for i in idxs:
        for j in idxs:
            if i != j:
                mask_module[i, j] = 1.0
    log(f"  模块[{module_name}]: 命中 {len(idxs)}/{len(cids)} 个节点"
        + (f"，未命中 {missing}" if missing else ""))

n_module = int(mask_module.sum())
log(f"层C 模块先验边数（含双向）: {n_module}")

# ==================== Step 5: 三层合并（取并集） ====================
# 为什么是并集而不是交集：三层证据各有盲区（KEGG注释不全/10样本噪声大/
# 只标注了3个模块），并集最大化保留真实耦合；且三层都有依据，噪声可控
log("")
log("=" * 60)
log("Step 5: 合并三层掩码（并集）")
log("=" * 60)

mask_MM = np.clip(mask_pathway + mask_corr + mask_module, 0, 1)
np.fill_diagonal(mask_MM, 0)                      # 去掉自环
n_total = int(mask_MM.sum())
density = n_total / (n * (n - 1)) if n > 1 else 0
log(f"合并后总边数（含双向）: {n_total}")
log(f"掩码密度: {density:.4%}（全连接为100%）")
log(f"参数量压缩: {n*(n-1)} → {n_total}")

# ==================== Step 6: 保存输出 ====================
log("")
log("=" * 60)
log("Step 6: 保存")
log("=" * 60)

np.save(os.path.join(DATA_DIR, "mask_metabolite2metabolite.npy"), mask_MM)

# 边列表（标注每条边来自哪几层，便于人工审查）
edges = []
for i in range(n):
    for j in range(n):
        if mask_MM[i, j] > 0:
            layers = []
            if mask_pathway[i, j] > 0: layers.append("通路共现")
            if mask_corr[i, j] > 0:    layers.append("数据相关")
            if mask_module[i, j] > 0:  layers.append("模块先验")
            edges.append((metab_nodes[i], metab_nodes[j], "+".join(layers)))
pd.DataFrame(edges, columns=["source", "target", "layer"]).to_csv(
    os.path.join(DATA_DIR, "mask_MM_edges.csv"), index=False)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log(f"✅ 掩码矩阵: {DATA_DIR}/mask_metabolite2metabolite.npy  形状 {mask_MM.shape}")
log(f"✅ 边列表:   {DATA_DIR}/mask_MM_edges.csv  共 {len(edges)} 条")
log(f"✅ 日志:     {LOG_FILE}")
log("")
log("【下一步】把 总边数/密度 发回核对：预期总边数 1500-3000（双向计数），"
    "密度 4%-8%。若仍远超此范围，把配置区 MIN_SHARED_PATHWAYS 调为 2 重跑（有缓存，秒级）。")