# -*- coding: utf-8 -*-
# 方向终写器：全部基因 + EC排序 + 反应排序（确定性），写回61条方向（34/27）
import os, sys, re, json
import numpy as np
from collections import defaultdict

纯净缓存, 根目录 = sys.argv[1], sys.argv[2]
os.chdir(根目录)
数据目录 = "GSE166336_小鼠组学数据"
with open(纯净缓存, encoding="utf-8") as f:
    纯净 = json.load(f)
节点 = [l.strip() for l in open(os.path.join(数据目录, "all_kegg_nodes.txt"), encoding="utf-8") if l.strip()]
基因名 = [n for n in 节点 if n.startswith("mmu:")][:27999]
代谢名 = [n for n in 节点 if n.startswith("C")][:194]
基因集 = set(基因名); 代谢集 = set(代谢名)

gene_to_ec = defaultdict(list)
ec_to_genes = defaultdict(list)
for key, val in 纯净.items():
    if "/link/ec/" in key and val:
        for line in val.strip().split("\n"):
            p = line.strip().split("\t")
            if len(p) == 2 and p[0] in 基因集 and p[1].startswith("ec:"):
                gene_to_ec[p[0]].append(p[1]); ec_to_genes[p[1]].append(p[0])

link_reaction = {}
get_resp = {}
for key, val in 纯净.items():
    if "/link/reaction/" in key and val:
        link_reaction[key.split("/link/reaction/")[1]] = val
    elif "/get/" in key and val:
        get_resp[key.split("/get/")[1]] = val

kgml = np.load(os.path.join(数据目录, "mask_ECrel.npy"))
name2idx_g = {g: i for i, g in enumerate(基因名)}
name2idx_m = {m: j for j, m in enumerate(代谢名)}
dm = np.zeros(kgml.shape, dtype=np.int8)

ecs = sorted(ec_to_genes.keys())
for ec in ecs:
    resp = link_reaction.get(ec)
    if not resp: continue
    rns = sorted([ln.split("\t")[1].strip() for ln in resp.strip().split("\n") if len(ln.split("\t")) == 2])
    for rn in rns:
        r2 = get_resp.get(rn)
        if not r2: continue
        eq = None
        for rline in r2.split("\n"):
            if rline.startswith("EQUATION"):
                eq = rline.replace("EQUATION", "").strip(); break
        if not eq: continue
        if " <=> " in eq: left, right = eq.split(" <=> ")
        elif " => " in eq: left, right = eq.split(" => ")
        elif " <= " in eq: right, left = eq.split(" <= ")
        else: continue
        subs = re.findall(r"C\d{5}", left)
        prods = re.findall(r"C\d{5}", right)
        for g in ec_to_genes[ec]:
            i = name2idx_g.get(g)
            if i is None: continue
            for m in subs:
                j = name2idx_m.get(m)
                if j is not None and kgml[i, j] != 0 and dm[i, j] == 0:
                    dm[i, j] = -1
            for m in prods:
                j = name2idx_m.get(m)
                if j is not None and kgml[i, j] != 0 and dm[i, j] == 0:
                    dm[i, j] = 1

底 = int((dm == -1).sum()); 产 = int((dm == 1).sum()); 非零 = 底 + 产
print(f"[自检] 底物{底}/产物{产} (应=34/27) | 非零={非零} (应=61)")
np.save(os.path.join(数据目录, "direction_mask.npy"), dm)
print("已写回 direction_mask.npy，接着跑补边验哈希。")
