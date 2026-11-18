# -*- coding: utf-8 -*-
# 缓存纯化器：以8/31纯净缓存为唯一数据源，重写目标缓存中所有link/ec批URL的响应
import os, sys, json
from collections import defaultdict

if len(sys.argv) < 3:
    sys.exit("用法： python 缓存纯化器.py <纯净缓存路径> <复现根目录>")
纯净路径, 根目录 = sys.argv[1], sys.argv[2]
os.chdir(根目录)
目标路径 = os.path.join("GSE166336_小鼠组学数据", "kegg_api_cache.json")

with open(纯净路径, encoding="utf-8") as f:
    纯净 = json.load(f)
with open(目标路径, encoding="utf-8") as f:
    目标 = json.load(f)

# 从纯净缓存解析 基因→原始行（8/31数据）
基因行 = defaultdict(list)
for key, val in 纯净.items():
    if "/link/ec/" in key and val:
        for line in val.strip().split("\n"):
            p = line.split("\t")
            if len(p) == 2 and p[0].startswith("mmu:"):
                基因行[p[0]].append(line)
print("纯净库有数据的基因:", len(基因行))

# 重写目标缓存里所有link/ec批URL
重写 = 0
for key in list(目标.keys()):
    if not key.startswith("https://rest.kegg.jp/link/ec/mmu:"):
        continue
    成员 = key.split("/link/ec/")[1].split("+")
    行 = []
    for g in 成员:
        行.extend(基因行.get(g, []))
    新值 = "\n".join(行)
    if 目标.get(key) != 新值:
        目标[key] = 新值
        重写 += 1

with open(目标路径, "w", encoding="utf-8") as f:
    json.dump(目标, f, ensure_ascii=False, indent=2)
print("重写的批URL数:", 重写)
print("完成，接着重跑ECrel主脚本。")
