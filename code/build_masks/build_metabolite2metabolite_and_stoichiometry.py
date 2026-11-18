import os
import time
import requests
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = "GSE166336_小鼠组学数据"
NODE_FILE = os.path.join(DATA_DIR, "all_kegg_nodes.txt")

# 读取代谢物节点
with open(NODE_FILE, "r") as f:
    all_nodes = [line.strip() for line in f if line.strip()]
metab_nodes = [n for n in all_nodes if n.startswith("C")]
print(f"代谢物节点数: {len(metab_nodes)}")

metab_to_idx = {m: i for i, m in enumerate(metab_nodes)}
n_metabs = len(metab_nodes)

# ===== 1. 获取每个代谢物参与的反应列表 =====
print("Step 1: 获取代谢物→反应映射...")
metab_to_reactions = {}
all_reaction_ids = set()

for cid in tqdm(metab_nodes, desc="查询代谢物反应"):
    url = f"http://rest.kegg.jp/link/reaction/{cid}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                rxns = []
                for line in r.text.strip().split("\n"):
                    if line:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            rxn_id = parts[1].replace("rn:", "")
                            rxns.append(rxn_id)
                            all_reaction_ids.add(rxn_id)
                metab_to_reactions[cid] = rxns
                break
            else:
                print(f"  查询 {cid} 失败: HTTP {r.status_code}, 重试 {attempt+1}/3")
                time.sleep(2)
        except Exception as e:
            print(f"  查询 {cid} 异常: {e}, 重试 {attempt+1}/3")
            time.sleep(2)
    else:
        metab_to_reactions[cid] = []
    time.sleep(0.2)

print(f"共找到 {len(all_reaction_ids)} 个相关反应")

# ===== 2. 获取每个反应的EQUATION =====
print("\nStep 2: 获取反应方程...")
reaction_equations = {}  # rxn_id -> (substrates, products)
rxn_id_to_idx = {}
reaction_list = []  # 用于构建S矩阵的列

for rxn_id in tqdm(all_reaction_ids, desc="查询反应方程"):
    url = f"http://rest.kegg.jp/get/rn:{rxn_id}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                for line in r.text.split("\n"):
                    if line.startswith("EQUATION"):
                        equation = line.replace("EQUATION", "").strip()
                        if "<=>" in equation or "=>" in equation or "<=" in equation:
                            parts = re.split(r'<=>|=>|<=', equation)
                            if len(parts) >= 2:
                                substrates = re.findall(r'(C\d+)', parts[0])
                                products = re.findall(r'(C\d+)', parts[1])
                                reaction_equations[rxn_id] = (substrates, products)
                                # 只保留涉及当前代谢物节点列表的反应
                                all_metabs_in_rxn = set(substrates) | set(products)
                                if any(m in metab_to_idx for m in all_metabs_in_rxn):
                                    reaction_list.append(rxn_id)
                        break
                break
            else:
                print(f"  查询反应 {rxn_id} 失败: HTTP {r.status_code}, 重试 {attempt+1}/3")
                time.sleep(2)
        except Exception as e:
            print(f"  查询反应 {rxn_id} 异常: {e}, 重试 {attempt+1}/3")
            time.sleep(2)
    else:
        print(f"  跳过反应 {rxn_id}")
    time.sleep(0.2)

print(f"有效反应数: {len(reaction_list)}")

# 构建反应索引
rxn_to_idx = {rxn: i for i, rxn in enumerate(reaction_list)}
n_reactions = len(reaction_list)

# ===== 3. 构建两张矩阵 =====
print("\nStep 3: 构建掩码矩阵和S矩阵...")
mask_metab2metab = np.zeros((n_metabs, n_metabs), dtype=np.float32)
S = np.zeros((n_metabs, n_reactions), dtype=np.int8)
edges = []

for rxn_id in tqdm(reaction_list, desc="构建矩阵"):
    if rxn_id not in reaction_equations:
        continue
    substrates, products = reaction_equations[rxn_id]
    rxn_idx = rxn_to_idx[rxn_id]

    for s in substrates:
        if s in metab_to_idx:
            S[metab_to_idx[s], rxn_idx] = -1
    for p in products:
        if p in metab_to_idx:
            S[metab_to_idx[p], rxn_idx] = +1

    for s in substrates:
        for p in products:
            if s in metab_to_idx and p in metab_to_idx:
                idx_s = metab_to_idx[s]
                idx_p = metab_to_idx[p]
                mask_metab2metab[idx_s, idx_p] = 1.0
                edges.append((s, p))

edges = list(set(edges))
print(f"掩码非零元素数: {int(mask_metab2metab.sum())}")
print(f"去重边数: {len(edges)}")
print(f"S矩阵形状: {S.shape}")

# ===== 4. 保存 =====
print("\nStep 4: 保存文件...")
os.makedirs(DATA_DIR, exist_ok=True)
np.save(os.path.join(DATA_DIR, "mask_metabolite2metabolite.npy"), mask_metab2metab)
np.save(os.path.join(DATA_DIR, "S_stoichiometry.npy"), S)

pd.DataFrame(edges, columns=["source", "target"]).to_csv(
    os.path.join(DATA_DIR, "mask_metabolite2metabolite_edges.csv"), index=False
)

pd.DataFrame({
    "reaction_id": reaction_list,
    "substrates": [",".join(reaction_equations[r][0]) for r in reaction_list],
    "products": [",".join(reaction_equations[r][1]) for r in reaction_list]
}).to_csv(os.path.join(DATA_DIR, "reaction_list.csv"), index=False)

print(f"✅ 代谢物→代谢物掩码已保存到: {DATA_DIR}/mask_metabolite2metabolite.npy")
print(f"   形状: {mask_metab2metab.shape}")
print(f"✅ S_化学计量矩阵已保存到: {DATA_DIR}/S_stoichiometry.npy")
print(f"   形状: {S.shape}")
print(f"✅ 反应列表已保存到: {DATA_DIR}/reaction_list.csv")