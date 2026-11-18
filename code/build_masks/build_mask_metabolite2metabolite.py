import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re

DATA_DIR = "GSE166336_小鼠组学数据"
KGML_DIR = "kegg_kgml"
NODE_FILE = os.path.join(DATA_DIR, "all_kegg_nodes.txt")

# 读取节点列表，筛选代谢物节点
with open(NODE_FILE, "r") as f:
    all_nodes = [line.strip() for line in f if line.strip()]
metab_nodes = [n for n in all_nodes if n.startswith("C")]
metab_to_idx = {m: i for i, m in enumerate(metab_nodes)}

print(f"代谢物节点数: {len(metab_nodes)}")

n_metabs = len(metab_nodes)
mask_metab2metab = np.zeros((n_metabs, n_metabs), dtype=np.float32)
edges = []

kgml_files = [f for f in os.listdir(KGML_DIR) if f.endswith(".kgml")]
print(f"找到 {len(kgml_files)} 个 KGML 文件")

for fname in kgml_files:
    filepath = os.path.join(KGML_DIR, fname)
    tree = ET.parse(filepath)
    root = tree.getroot()

    for reaction in root.findall("reaction"):
        equation = reaction.get("equation", "")
        if not equation:
            continue

        # 解析底物和产物（箭头左侧是底物，右侧是产物）
        if "<=>" in equation or "=>" in equation or "<=" in equation:
            parts = re.split(r'<=>|=>|<=', equation)
            if len(parts) >= 2:
                substrates = re.findall(r'(C\d+)', parts[0])
                products = re.findall(r'(C\d+)', parts[1])

                for s in substrates:
                    for p in products:
                        if s in metab_to_idx and p in metab_to_idx:
                            edges.append((s, p))
                            mask_metab2metab[metab_to_idx[s], metab_to_idx[p]] = 1.0

# 去重边
edges = list(set(edges))
print(f"提取到代谢物→代谢物边数: {len(edges)}")
print(f"掩码矩阵非零元素数: {int(np.sum(mask_metab2metab))}")

# 保存
np.save(os.path.join(DATA_DIR, "mask_metabolite2metabolite.npy"), mask_metab2metab)
pd.DataFrame(edges, columns=["source", "target"]).to_csv(
    os.path.join(DATA_DIR, "mask_metabolite2metabolite_edges.csv"), index=False
)

print(f"✅ 代谢物→代谢物掩码已保存到: {DATA_DIR}/mask_metabolite2metabolite.npy")
print(f"   形状: {mask_metab2metab.shape}")