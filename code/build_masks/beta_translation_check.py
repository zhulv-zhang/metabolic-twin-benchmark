# -*- coding: utf-8 -*-
# ============================================================
# β_翻译（基因→蛋白）对齐检查脚本（只检查不改数据，~5分钟）
# ============================================================
#
# 【功能】β_翻译是"基因→蛋白"的对角映射（中心法则：每个基因的mRNA
#   翻译成对应蛋白）。它不需要建大掩码，但必须验证一件事：
#   三个组学的基因身份能对上——
#     ① 转录组测到的基因 有多少在节点列表里
#     ② 蛋白组测到的蛋白 对应的基因 是否在转录组里（孤儿蛋白检查）
#     ③ 磷酸化组测到的蛋白 对应的基因 是否在蛋白组/转录组里
#   对不上的是"孤儿"：有蛋白数据但无基因数据（或反过来）的节点，
#   训练时这些边的输入恒为0，需要知道它们有多少、是谁。
#
# 【输出】控制台报告 + build_beta_check_log.txt（同目录）
# 【运行】python β翻译_对齐检查_带注释.py
# ============================================================

import os
import json
import pandas as pd

DATA_DIR = "GSE166336_小鼠组学数据"

def locate(fname):
    for cand in [fname, os.path.join(DATA_DIR, fname)]:
        if os.path.exists(cand):
            return cand
    return None

NODE_FILE = locate("all_kegg_nodes.txt")
FILES = {
    "转录组":   locate("transcriptome_kegg_map.csv"),
    "蛋白组":   locate("proteome_kegg_map.csv"),
    "磷酸化组": locate("phosphoproteome_kegg_map.csv"),
}
LOG_FILE = os.path.join(DATA_DIR, "build_beta_check_log.txt")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=" * 60)
log("β_翻译（基因→蛋白）对齐检查")
log("=" * 60)

# ---- 节点列表（基因部分，与ECrel同规则） ----
with open(NODE_FILE, "r", encoding="utf-8") as f:
    all_nodes = [l.strip() for l in f if l.strip()]
gene_nodes = [n for n in all_nodes if n.startswith("mmu:")]
node_genes = set(gene_nodes)                     # 唯一 mmu 编号集合
log(f"节点列表基因: {len(node_genes)} 个唯一 mmu 编号")

def load_id_set(fp, label):
    """从映射表CSV里自动找ID列，返回 (mmu编号集合, symbol集合)。
    自动适配两种情形：列里带 mmu: 前缀的ID，或基因symbol。"""
    df = pd.read_csv(fp)
    log(f"\n[{label}] {os.path.basename(fp)}: {df.shape[0]} 行 × {df.shape[1]} 列")
    log(f"  列名: {list(df.columns)[:8]}")
    mmu_ids, symbols = set(), set()
    for col in df.columns:
        vals = df[col].dropna().astype(str)
        hit_mmu = vals[vals.str.startswith("mmu:")]
        if len(hit_mmu) > 10:                    # 该列是 mmu ID 列
            mmu_ids |= set(hit_mmu)
            log(f"  → ID列: '{col}'（mmu: 格式，{len(hit_mmu)} 个）")
        else:
            # 试探是否为symbol列：抽样20个值，多数是"字母开头短词"则认为是
            sample = vals.head(20).tolist()
            if sample and all(2 <= len(v) <= 20 and v[0].isalpha() for v in sample):
                symbols |= set(vals)
                log(f"  → 候选symbol列: '{col}'（{len(set(vals))} 个唯一值）")
    return mmu_ids, symbols, df

# ---- 读取三个组学映射表 ----
omics = {}
for label, fp in FILES.items():
    if not fp:
        log(f"\n[{label}] ✗ 文件未找到！")
        continue
    omics[label] = load_id_set(fp, label)

# ---- 若只有symbol列，借 KEGG list/mmu 缓存做 symbol→mmu 转换 ----
CACHE_FILE = locate("kegg_api_cache.json")
symbol_to_mmu = {}
if CACHE_FILE:
    cache = json.load(open(CACHE_FILE, encoding="utf-8"))
    list_text = cache.get("https://rest.kegg.jp/list/mmu", "")
    for line in list_text.strip().split("\n"):
        parts = line.split("\t")
        if not parts[0].startswith("mmu:"):
            continue
        desc = parts[3] if (len(parts) >= 4 and parts[1] == "CDS") else (parts[1] if len(parts) >= 2 else "")
        for sym in desc.split(";")[0].split(","):
            sym = sym.strip()
            if sym:
                symbol_to_mmu.setdefault(sym, parts[0])
    log(f"\nsymbol→mmu 映射（来自缓存）: {len(symbol_to_mmu)} 个")

def to_mmu(mmu_ids, symbols):
    """统一成 mmu 编号集合"""
    out = set(mmu_ids)
    out |= {symbol_to_mmu[s] for s in symbols if s in symbol_to_mmu}
    return out

sets = {label: to_mmu(m, s) for label, (m, s, _) in omics.items()}

# ---- 对齐报告 ----
log("\n" + "=" * 60)
log("对齐报告")
log("=" * 60)
for label, gs in sets.items():
    log(f"[{label}] 测到基因: {len(gs)}，其中在节点列表: {len(gs & node_genes)} "
        f"({len(gs & node_genes) / max(len(gs),1) * 100:.1f}%)")

if "蛋白组" in sets and "转录组" in sets:
    orphan_prot = (sets["蛋白组"] & node_genes) - sets["转录组"]
    log(f"\n孤儿蛋白（有蛋白数据但转录组无对应基因）: {len(orphan_prot)} 个")
    if orphan_prot:
        log(f"  示例: {sorted(orphan_prot)[:10]}")
if "磷酸化组" in sets:
    ref = sets.get("蛋白组", set()) | sets.get("转录组", set())
    orphan_phos = (sets["磷酸化组"] & node_genes) - ref
    log(f"孤儿磷酸化蛋白（磷酸化组有但蛋白组/转录组都无）: {len(orphan_phos)} 个")
    if orphan_phos:
        log(f"  示例: {sorted(orphan_phos)[:10]}")

# 三方交集 = β_翻译对角映射的实际有效规模
common = sets.get("转录组", set()) & sets.get("蛋白组", set()) & node_genes
log(f"\nβ_翻译有效对角（转录组∩蛋白组∩节点）: {len(common)} 个基因")
if "磷酸化组" in sets:
    common3 = common & sets["磷酸化组"]
    log(f"三层齐全（转录∩蛋白∩磷酸化∩节点）: {len(common3)} 个基因")

log("\n判读标准：孤儿占比<10%为健康；10-30%可接受（论文注明）；>30%需查原因。")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"\n日志已保存: {LOG_FILE}")