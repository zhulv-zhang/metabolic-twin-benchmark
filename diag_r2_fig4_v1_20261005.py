# 图4四臂·掩码R²统一重算 v1（2026-10-05 S1）
# 用法: python diag_r2_fig4_v1_20261005.py <pred_vs_obs.npz路径> [...]
# 口径: R2 = 1 - SSres/SStot, 分母逐层逐条件独立, 仅掩码位; pred取traj在tidx切片
import numpy as np, sys
def masked_r2(pred, tgt, m):
    mm = m > 0.5
    res = (((pred - tgt) ** 2) * mm).sum()
    mu = (tgt * mm).sum() / mm.sum()
    tot = (((tgt - mu) ** 2) * mm).sum()
    return 1 - res / max(tot, 1e-12)
for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    n_g = len(z['gene_ids'])
    n_m = len(z['metab_ids'])
    print(f"== {path}")
    for lay, sl in [("G",(0,n_g)),("P",(n_g,2*n_g)),("S",(2*n_g,3*n_g)),("M",(3*n_g,3*n_g+n_m))]:
        tidx = np.asarray(z[f'tidx_{lay}']).astype(int)
        obs = np.asarray(z[f'obs_{lay}']); om = np.asarray(z[f'omask_{lay}'])
        out = []
        for c in range(2):
            pred = np.asarray(z['traj_insulin' if c==0 else 'traj_pbs'])[:, sl[0]:sl[1]][tidx].T
            out.append(masked_r2(pred, obs[:,:,c], om[:,:,c]))
        print(f"  {lay}: Ins {out[0]:+.2f} / PBS {out[1]:+.2f}")
