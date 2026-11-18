# -*- coding: utf-8 -*-
# ============================================================
# ECrel 掩码补边脚本（离线版，不联网）v3
# ============================================================
#
# 【功能】
#   利用 ECrel 主脚本运行时留下的 KEGG API 缓存（kegg_api_cache.json），
#   解析其中已下载的"反应方程式"数据（基因→EC→反应→底物/产物），
#   补充构建"反应级精确"的酶→代谢物边，合并进现有掩码。
#
# 【为什么需要补边】
#   原掩码的边只来自 KGML 通路图中的 ECrel 连接关系（步骤3），
#   KGML 的 ECrel 是"通路邻居"粒度，会漏掉真实反应边。
#   已确认案例：丙酮酸激酶(mmu:18746 Pklr / mmu:18770 Pkm) → PEP(C00074)。
#
# 【补边规则（方向）】
#     酶→产物：方向 +1（产物是该酶正向生成的）
#     酶→底物：方向 -1（底物是该酶正向消耗的）
#   掩码允许学习，方向是软先验（L_direction），个别标反不会硬阻断。
#
# 【v3 重要修改：索引重建 + 完整性验证】
#   发现文件夹里的 gene_index_map.json (14076条) 与掩码(27999行)不一致，
#   内容不可信。但节点文件 all_kegg_nodes.txt 自 8/28 起未被修改，
#   ECrel 主脚本(8/31)读的就是它，因此用主脚本相同的过滤规则
#   （行首 mmu: = 基因，行首 C = 代谢物，保持文件顺序）重建索引，
#   必然与掩码行列一一对应。
#   重建后用现有 1285 条边逐条回验：每条边在掩码对应位置必须=1，
#   全部通过才继续，否则报错退出（不会写坏数据）。
#   验证通过后会重新保存正确的索引映射，覆盖异常文件。
#
# 【使用方法】
#   放在工作目录直接运行（自动在 当前目录 和 数据子目录 两处找文件）：
#       python ECrel补边_从反应方程补充_带注释.py
# ============================================================

import json
import os
import re
import shutil
import numpy as np
import pandas as pd

# ----------------- 路径配置（自动定位：当前目录 或 数据子目录） -----------------
DATA_DIR = "GSE166336_小鼠组学数据"

def locate(fname):
    """文件可能在工作目录，也可能在 DATA_DIR 子目录里，两处都找"""
    for cand in [fname, os.path.join(DATA_DIR, fname)]:
        if os.path.exists(cand):
            return cand
    return fname

NODE_FILE  = locate("all_kegg_nodes.txt")
CACHE_FILE = locate("kegg_api_cache.json")
MASK_FILE  = locate("mask_ECrel.npy")
DIR_FILE   = locate("direction_mask.npy")
EDGE_CSV   = locate("mask_ECrel_edges.csv")
GENE_MAP_FILE  = locate("gene_index_map.json")
METAB_MAP_FILE = locate("metabolite_index_map.json")
LOG_FILE = os.path.join(os.path.dirname(MASK_FILE) or ".", "ECrel补边日志.txt")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=" * 60)
log("ECrel 补边 v3：从反应方程缓存补充精确边")
log("=" * 60)
for name, fp in [("节点文件", NODE_FILE), ("缓存文件", CACHE_FILE),
                 ("掩码文件", MASK_FILE), ("边表文件", EDGE_CSV)]:
    log(f"[路径] {name}: {fp}")

for fp in [NODE_FILE, CACHE_FILE, MASK_FILE, DIR_FILE, EDGE_CSV]:
    if not os.path.exists(fp):
        raise SystemExit(f"错误：找不到 {fp}")

# ============================================================
# 步骤 0：从节点文件重建索引（与 ECrel 主脚本完全相同的规则）
# ============================================================
with open(NODE_FILE, "r", encoding="utf-8") as f:
    all_nodes = [l.strip() for l in f if l.strip()]

# 主脚本规则：startswith("mmu:") 为基因，startswith("C") 为代谢物，保持文件顺序
gene_nodes  = [n for n in all_nodes if n.startswith("mmu:")]
metab_nodes = [n for n in all_nodes if n.startswith("C")]
gene_to_idx  = {g: i for i, g in enumerate(gene_nodes)}
metab_to_idx = {m: i for i, m in enumerate(metab_nodes)}
idx_to_metab = {v: k for k, v in metab_to_idx.items()}

log(f"\n节点文件: 总 {len(all_nodes)} 行")
log(f"  基因节点(mmu:): {len(gene_nodes)}")
log(f"  代谢物节点(C):  {len(metab_nodes)}")

# 顺便诊断：长度≠6的C编号 和 既不是mmu也不是C的行（弄清28201=27999+194+8里的"8"）
odd_c   = [n for n in metab_nodes if len(n) != 6]
odd_oth = [n for n in all_nodes if not n.startswith(("mmu:", "C"))]
if odd_c:
    log(f"  [诊断] 长度≠6的代谢物编号: {odd_c}")
if odd_oth:
    log(f"  [诊断] 既非基因也非代谢物的行({len(odd_oth)}): {odd_oth[:10]}")

# ============================================================
# 步骤 1：加载掩码，形状校验 + 用现有边表做完整性回验
# ============================================================
mask = np.load(MASK_FILE)
dirmask = np.load(DIR_FILE)
log(f"\n现有掩码: 形状 {mask.shape}, 边数 {int(mask.sum())}")
log(f"现有方向已知边数: {int((dirmask != 0).sum())}")

if mask.shape != (len(gene_nodes), len(metab_nodes)):
    raise SystemExit(
        f"错误：重建索引 ({len(gene_nodes)}, {len(metab_nodes)}) "
        f"与掩码形状 {mask.shape} 不一致！\n"
        f"说明节点文件在掩码构建后被改过，不能用当前文件重建索引。"
    )
log("形状校验: 通过")

# --- 完整性回验：现有每条边在掩码对应位置必须=1 ---
edges_old = pd.read_csv(EDGE_CSV)
# 兼容两种列名（主脚本版本 / 补边脚本版本）
gc = "source_gene" if "source_gene" in edges_old.columns else "enzyme"
mc = "target_metabolite" if "target_metabolite" in edges_old.columns else "metabolite"
n_ok, n_bad = 0, 0
for _, row in edges_old.iterrows():
    g, m = row[gc], row[mc]
    if g in gene_to_idx and m in metab_to_idx \
       and mask[gene_to_idx[g], metab_to_idx[m]] == 1:
        n_ok += 1
    else:
        n_bad += 1
log(f"完整性回验: {n_ok}/{len(edges_old)} 条现有边与掩码位置吻合")
if n_bad > 0:
    raise SystemExit(
        f"错误：{n_bad} 条现有边在重建索引下对不上掩码！\n"
        f"索引顺序与掩码构建时不一致，已停止，未修改任何文件。"
    )
log("完整性回验: 全部通过，重建索引与掩码完全一致")

# 备份
for fp in [MASK_FILE, DIR_FILE, EDGE_CSV]:
    bak = fp.replace(".npy", "_备份_补边前.npy").replace(".csv", "_备份_补边前.csv")
    if not os.path.exists(bak):
        shutil.copy(fp, bak)
        log(f"已备份: {bak}")

# ============================================================
# 步骤 2：解析缓存 —— 重建 基因→EC→反应→底物/产物
# ============================================================
with open(CACHE_FILE, "r", encoding="utf-8") as f:
    cache = json.load(f)
log(f"\n缓存条目数: {len(cache)}")

# --- 2a) link/ec/{基因批次} → 基因→EC，值每行: "mmu:18746\tec:2.7.1.40" ---
gene_to_ecs = {}
for key, val in cache.items():
    if "/link/ec/" in key and val:
        for line in val.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 2 and parts[0].startswith("mmu:"):
                gene_to_ecs.setdefault(parts[0], set()).add(parts[1])
log(f"缓存中基因→EC: {len(gene_to_ecs)} 个基因有 EC 注释")

# --- 2b) link/reaction/{ec} → EC→反应，值每行: "ec:2.7.1.40\trn:R00200" ---
ec_to_rxns = {}
for key, val in cache.items():
    if "/link/reaction/" in key and val:
        for line in val.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 2 and parts[1].startswith("rn:"):
                ec_to_rxns.setdefault(parts[0], []).append(parts[1][3:])  # 去 "rn:" 前缀
log(f"缓存中EC→反应: {len(ec_to_rxns)} 个 EC 有反应注释")

# --- 2c) get/rn:{R编号} → 反应方程式 → 底物/产物 ---
# 缓存键形如 "https://rest.kegg.jp/get/rn:R00200"（带 rn: 前缀）
# EQUATION 行: "EQUATION    C00031 + C00002 <=> C00092 + C00008"
rxn_equation = {}
for key, val in cache.items():
    if "/get/" not in key or not val:
        continue
    m = re.search(r"/get/(?:rn:)?(R\d{5})", key)   # 兼容带不带 rn: 前缀
    if not m:
        continue
    rxn_id = m.group(1)
    eq = None
    for line in val.split("\n"):
        if line.startswith("EQUATION"):
            eq = line.replace("EQUATION", "").strip()
            break
    if not eq:
        continue
    if " <=> " in eq:
        left, right = eq.split(" <=> ", 1)
    elif " => " in eq:
        left, right = eq.split(" => ", 1)
    elif " <= " in eq:
        right, left = eq.split(" <= ", 1)
    else:
        continue
    rxn_equation[rxn_id] = {
        "substrates": re.findall(r"C\d{5}", left),
        "products":   re.findall(r"C\d{5}", right),
    }
log(f"缓存中反应方程: {len(rxn_equation)} 个反应解析成功")

# ============================================================
# 步骤 3：生成反应级精确边
# ============================================================
new_edges = {}   # (gene, metab) -> direction
n_rxn_used = 0
skipped_no_rxn = 0

for gene, ecs in gene_to_ecs.items():
    if gene not in gene_to_idx:
        continue
    for ec in sorted(ecs):
        for rxn_id in ec_to_rxns.get(ec, ()):
            eq = rxn_equation.get(rxn_id)
            if not eq:
                skipped_no_rxn += 1
                continue
            n_rxn_used += 1
            for cpd in eq["substrates"]:
                if cpd in metab_to_idx and (gene, cpd) not in new_edges:
                    new_edges[(gene, cpd)] = -1     # 底物：消耗
            for cpd in eq["products"]:
                if cpd in metab_to_idx and (gene, cpd) not in new_edges:
                    new_edges[(gene, cpd)] = +1     # 产物：生成

log(f"\n从缓存反应方程生成候选边: {len(new_edges)} 条"
    f"（{n_rxn_used} 个反应被使用, {skipped_no_rxn} 次反应无方程跳过）")

# ============================================================
# 步骤 4：合并进掩码（只新增，不删除原有边）
# ============================================================
n_added = 0
n_dir_filled = 0
rows_new = []

for (gene, cpd), d in sorted(new_edges.items()):
    i, j = gene_to_idx[gene], metab_to_idx[cpd]
    if mask[i, j] == 0:
        mask[i, j] = 1
        dirmask[i, j] = d
        n_added += 1
        rows_new.append({
            "source_gene": gene, "target_metabolite": cpd,
            "direction_code": d,
            "direction_label": {1: "product", -1: "substrate", 0: "unknown"}[d],
            "source": "补边_反应方程",
        })
    else:
        if dirmask[i, j] == 0:      # 已有边但方向未知 → 补方向，不覆盖已有方向
            dirmask[i, j] = d
            n_dir_filled += 1

log(f"\n===== 补边结果 =====")
log(f"新增边: {n_added} 条")
log(f"原有边补到方向: {n_dir_filled} 条")
log(f"合并后总边数: {int(mask.sum())}")
log(f"合并后方向已知: {int((dirmask != 0).sum())} "
    f"({(dirmask != 0).sum() / max(mask.sum(), 1) * 100:.1f}%)")

# ============================================================
# 步骤 5：终验 —— 丙酮酸激酶 → PEP 这条边必须在
# ============================================================
log(f"\n===== 终验（上轮遗留问题）=====")
for gene, gname in [("mmu:18746", "Pklr"), ("mmu:18770", "Pkm")]:
    if gene not in gene_to_idx:
        log(f"{gname}({gene}): 不在掩码索引里")
        continue
    i = gene_to_idx[gene]
    tgts = [idx_to_metab[j] for j in range(mask.shape[1]) if mask[i, j] == 1]
    has_pep = "C00074" in tgts
    log(f"{gname}({gene}) 连接 {len(tgts)} 个代谢物, "
        f"含PEP(C00074): {'✓ 已修复' if has_pep else '✗ 仍缺失!'}")
    if has_pep:
        d = int(dirmask[i, metab_to_idx["C00074"]])
        log(f"    PEP方向: {d} (-1=底物消耗，符合PK反应 PEP→丙酮酸)")

# ============================================================
# 步骤 6：保存（含重新生成正确的索引映射，覆盖异常文件）
# ============================================================
np.save(MASK_FILE, mask)
np.save(DIR_FILE, dirmask)

edges_all = pd.concat([edges_old, pd.DataFrame(rows_new)], ignore_index=True)
edges_all.to_csv(EDGE_CSV, index=False, encoding="utf-8-sig")

# 覆盖之前那对条数对不上的索引映射（先备份）
for fp, mapping in [(GENE_MAP_FILE, gene_to_idx), (METAB_MAP_FILE, metab_to_idx)]:
    if os.path.exists(fp):
        bak = fp.replace(".json", "_备份_条数异常.json")
        if not os.path.exists(bak):
            shutil.copy(fp, bak)
            log(f"已备份异常索引文件: {bak}")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
log(f"已重新保存正确索引映射: 基因 {len(gene_to_idx)} 条, 代谢物 {len(metab_to_idx)} 条")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log(f"\n已保存: {MASK_FILE}, {DIR_FILE}, {EDGE_CSV}(+{len(rows_new)}行), {LOG_FILE}")
log("完成。原文件备份为 *_备份_补边前.*，如需回滚直接改名即可。")