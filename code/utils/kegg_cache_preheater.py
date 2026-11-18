# -*- coding: utf-8 -*-
# 复现缓存预热器 v3：不解析KGML，直接挖原始批URL重建基因池
# 过关三数：基因池=517、批数=52、新增=52
import os, sys, json
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("用法： python 复现_cache预热器_v3.py <复现根目录>")
os.chdir(sys.argv[1])
print("[诊断] 工作目录:", os.getcwd())

缓存文件 = os.path.join("GSE166336_小鼠组学数据", "kegg_api_cache.json")
with open(缓存文件, encoding="utf-8") as f:
    cache = json.load(f)
print("[诊断] 缓存条数:", len(cache))

基因池 = []
批URL数 = 0
for key in cache:
    if key.startswith("https://rest.kegg.jp/link/ec/mmu:"):
        批URL数 += 1
        基因池.extend(key.split("/link/ec/")[1].split("+"))
print("[诊断] 原始批URL数:", 批URL数, "| 挖出基因:", len(基因池), "| 去重后:", len(set(基因池)))

全部基因 = list(set(基因池))
print("基因:", len(全部基因))
if len(全部基因) != 517:
    sys.exit("错误：基因池≠517，停下来把输出贴回")

基因行 = defaultdict(list)
for key, val in cache.items():
    if "/link/ec/" in key and val:
        for line in val.strip().split("\n"):
            p = line.split("\t")
            if len(p) == 2 and p[0].startswith("mmu:"):
                基因行[p[0]].append(line)
print("缓存中有link/ec数据的基因:", len(基因行))

新增 = 0
for i in range(0, len(全部基因), 10):
    批 = 全部基因[i:i+10]
    url = "https://rest.kegg.jp/link/ec/" + "+".join(批)
    行 = []
    for g in 批:
        行.extend(基因行.get(g, []))
    if url not in cache or cache[url] is None:
        cache[url] = "\n".join(行)
        新增 += 1

with open(缓存文件, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)
print("批数:", (len(全部基因)+9)//10, "| 新增:", 新增, "| 缓存总条数:", len(cache))
if 新增 != 52:
    print("注意：新增≠52（缓存已被预热过则正常；首次应为52）")
print("预热完成。同一窗口（PYTHONHASHSEED=0）接着跑ECrel主脚本。")
