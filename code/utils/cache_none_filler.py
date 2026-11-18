# -*- coding: utf-8 -*-
# 缓存None补齐器：主脚本上轮请求失败留None的link/ec批URL，用缓存内单基因数据补全
import os, sys, json
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("用法： python 缓存None补齐器.py <复现根目录>")
os.chdir(sys.argv[1])
缓存文件 = os.path.join("GSE166336_小鼠组学数据", "kegg_api_cache.json")
with open(缓存文件, encoding="utf-8") as f:
    cache = json.load(f)
print("[诊断] 缓存条数:", len(cache))

基因行 = defaultdict(list)
for key, val in cache.items():
    if "/link/ec/" in key and val:
        for line in val.strip().split("\n"):
            p = line.split("\t")
            if len(p) == 2 and p[0].startswith("mmu:"):
                基因行[p[0]].append(line)
print("[诊断] 有link/ec数据的基因:", len(基因行))

补了 = 0
for key in list(cache.keys()):
    if "/link/ec/" in key and cache[key] is None:
        genes = key.split("/link/ec/")[1].split("+")
        lines = []
        for g in genes:
            lines.extend(基因行.get(g, []))
        cache[key] = "\n".join(lines)
        补了 += 1

with open(缓存文件, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)
print("补齐None条目:", 补了)
print("完成。同一窗口（PYTHONHASHSEED=0）重跑ECrel主脚本。")
