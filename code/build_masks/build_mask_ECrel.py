import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# ===== 配置 =====
DATA_DIR = "GSE166336_小鼠组学数据"
KGML_DIR = "kegg_kgml"
NODE_FILE = os.path.join(DATA_DIR, "all_kegg_nodes.txt")

# ===== 1. 读取节点列表 =====
with open(NODE_FILE, "r") as f:
    all_nodes = [line.strip() for line in f if line.strip()]

print(f"总节点数: {len(all_nodes)}")

# 分离基因节点和代谢物节点
gene_nodes = [n for n in all_nodes if n.startswith("mmu:")]
metab_nodes = [n for n in all_nodes if n.startswith("C")]
gene_to_idx = {g: i for i, g in enumerate(gene_nodes)}
metab_to_idx = {m: i for i, m in enumerate(metab_nodes)}

print(f"基因节点数: {len(gene_nodes)}")
print(f"代谢物节点数: {len(metab_nodes)}")

# ===== 2. 解析所有 KGML 文件，提取 ECrel 边 =====
kgml_files = [f for f in os.listdir(KGML_DIR) if f.endswith(".kgml")]
print(f"找到 {len(kgml_files)} 个 KGML 文件")

edges_ec = []  # 存储 (src_kegg, tgt_kegg)

for fname in kgml_files:
    filepath = os.path.join(KGML_DIR, fname)
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception as e:
        print(f"  ⚠️ 解析 {fname} 失败: {e}")
        continue

    # 构建 entry_id → kegg_id 映射
    entry_to_kegg = {}
    for entry in root.findall("entry"):
        entry_id = entry.get("id")
        name_attr = entry.get("name", "")
        if name_attr:
            kegg_id = name_attr.split()[0]
            # ECrel 边涉及基因（mmu:xxxxx）或代谢物（Cxxxxx）
            if kegg_id.startswith("mmu:") or kegg_id.startswith("C"):
                entry_to_kegg[entry_id] = kegg_id

    # 提取 ECrel 边
    for rel in root.findall("relation"):
        rel_type = rel.get("type")
        if rel_type != "ECrel":
            continue

        src_entry = rel.get("entry1")
        tgt_entry = rel.get("entry2")
        if src_entry in entry_to_kegg and tgt_entry in entry_to_kegg:
            src_kegg = entry_to_kegg[src_entry]
            tgt_kegg = entry_to_kegg[tgt_entry]
            # ECrel 方向：酶（mmu:xxxxx）→ 代谢物（Cxxxxx）
            if src_kegg.startswith("mmu:") and tgt_kegg.startswith("C"):
                edges_ec.append((src_kegg, tgt_kegg))

print(f"\n提取到 ECrel 边: {len(edges_ec)} 条（原始）")

# ===== 3. 过滤：只保留两端都在节点列表中的边 =====
valid_edges = []
for src, tgt in edges_ec:
    if src in gene_to_idx and tgt in metab_to_idx:
        valid_edges.append((src, tgt))

print(f"有效 ECrel 边: {len(valid_edges)} 条")

# ===== 4. 构建掩码矩阵（基因 × 代谢物） =====
mask_ec = np.zeros((len(gene_nodes), len(metab_nodes)), dtype=np.float32)

for src, tgt in valid_edges:
    i = gene_to_idx[src]
    j = metab_to_idx[tgt]
    mask_ec[i, j] = 1.0

print(f"ECrel 掩码矩阵非零元素数: {int(mask_ec.sum())}")
print(f"掩码矩阵形状: {mask_ec.shape} (基因 × 代谢物)")

# ===== 5. 保存 =====
# 保存 .npy 矩阵
np.save(os.path.join(DATA_DIR, "mask_ECrel.npy"), mask_ec)
print(f"✅ 掩码矩阵已保存到: {DATA_DIR}/mask_ECrel.npy")

# 保存边列表 CSV（便于人工查看）
edges_df = pd.DataFrame(valid_edges, columns=["source_gene", "target_metabolite"])
edges_df.to_csv(os.path.join(DATA_DIR, "mask_ECrel_edges.csv"), index=False)
print(f"✅ 边列表已保存到: {DATA_DIR}/mask_ECrel_edges.csv")

# ===== 6. 统计 =====
print(f"\n===== 统计 =====")
print(f"基因节点数: {len(gene_nodes)}")
print(f"代谢物节点数: {len(metab_nodes)}")
print(f"ECrel 边数: {len(valid_edges)}")
print(f"掩码密度: {len(valid_edges) / (len(gene_nodes) * len(metab_nodes)):.6%}")