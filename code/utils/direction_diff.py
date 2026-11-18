# -*- coding: utf-8 -*-
# 方向差异对比器：找出两份direction_mask.npy取值不同的格子
import sys
import numpy as np

正身掩码, 正身节点 = sys.argv[1], sys.argv[2]
复现掩码, 复现节点 = sys.argv[3], sys.argv[4]

def load(npz_path, nodes_path):
    m = np.load(npz_path)
    genes = [l.strip() for l in open(nodes_path, encoding="utf-8") if l.strip().startswith("mmu:")][:m.shape[0]]
    mets  = [l.strip() for l in open(nodes_path, encoding="utf-8") if l.strip().startswith("C")][:m.shape[1]]
    return m, genes, mets

A, ga, ma = load(正身掩码, 正身节点)
B, gb, mb = load(复现掩码, 复现节点)
print("正身形:", A.shape, "复现形:", B.shape)
assert A.shape == B.shape, "形状不一致"

d = np.argwhere(A != B)
print("差异总数:", len(d))
for i, j in d:
    print(f"基因 {ga[i]} | 代谢物 {ma[j]} | 正身={int(A[i,j])} 复现={int(B[i,j])}")
