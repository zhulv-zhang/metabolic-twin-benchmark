import pandas as pd
import os

# 统一数据目录
DATA_DIR = "GSE166336_小鼠组学数据"

# 读取四个映射文件
phospho = pd.read_csv(os.path.join(DATA_DIR, "phosphoproteome_kegg_map.csv"))
proteome = pd.read_csv(os.path.join(DATA_DIR, "proteome_kegg_map.csv"))
transcript = pd.read_csv(os.path.join(DATA_DIR, "transcriptome_kegg_map.csv"))
metabolome = pd.read_csv(os.path.join(DATA_DIR, "metabolome_kegg_map.csv"))

# 提取有效的 KEGG ID（非空）
phospho_ids = phospho[phospho['kegg_id'].notna()]['kegg_id'].tolist()
proteome_ids = proteome[proteome['kegg_id'].notna()]['kegg_id'].tolist()
transcript_ids = transcript[transcript['kegg_id'].notna()]['kegg_id'].tolist()
metabolome_ids = metabolome[metabolome['kegg_id'].notna()]['kegg_id'].tolist()

print(f"磷酸化蛋白组节点数: {len(phospho_ids)}")
print(f"蛋白组节点数: {len(proteome_ids)}")
print(f"转录组节点数: {len(transcript_ids)}")
print(f"代谢组节点数: {len(metabolome_ids)}")

# 统一节点列表（按顺序：磷酸化 → 转录 → 蛋白 → 代谢）
all_nodes = phospho_ids + transcript_ids + proteome_ids + metabolome_ids
print(f"总节点数: {len(all_nodes)}")

# 保存节点列表
with open(os.path.join(DATA_DIR, "all_kegg_nodes.txt"), "w") as f:
    for node in all_nodes:
        f.write(node + "\n")

print(f"节点列表已保存到 {DATA_DIR}/all_kegg_nodes.txt")