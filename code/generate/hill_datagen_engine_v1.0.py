#!/usr/bin/env python3
# hill_datagen_engine_v1.0（Hill数据生成引擎）——砖② Hill 先行件
# 令据：Advisor 开工令 2026-09-30＋analysis_plan v1.4（ed9f5f01）⑦节
# 产出：①收敛验证页（欧拉 dt0.5 vs dt0.25，未截断，L∞<1%）②双积分对照档（欧拉/RK4 同初值同步长）
# 纪律：向量场不出引擎（本件只在服务器运行）；标定常数不写成规格件；v1.0 仅造数+验页，不训练。
# 接线假设沿用冒烟实物核验版（ASM 六条已对引擎实物验过，存储约定 (from,to)）。
import numpy as np, os, hashlib, json, sys

H_HILL, K_HILL = 3, 1.0          # Hill 族定义参数（非引擎标定）
TMAX, PULSE_T, PULSE_A = 240.0, 10.0, 0.1
OUT = 'data_sim_hill_v1'
LAYERS = 'GPSM'
ASM = {'G':[('W_GE','G',1),('W_SG','S',1),('W_FB','M',-1)],
       'S':[('W_PS','P',1)], 'P':[], 'M':[('W_MM','M',1),('W_EC','G',-1)]}

def fhill(z):
    s = z**H_HILL / (K_HILL**H_HILL + z**H_HILL)
    return 2*s - 1

def load_truth():
    gt = np.load('ground_truth.npz'); tt = np.load('train_tensors.npz')
    W = {k: gt[k] for k in ['W_GE','W_SG','W_PS','W_MM','W_FB','W_EC']}
    assert gt['decay_true'].size == 4, 'decay_true 应为分层 (4,)'
    dims = {L: tt['X'+L].shape[0] for L in LAYERS}
    dec = {L: float(gt['decay_true'][i]) for i, L in enumerate(LAYERS)}
    x0 = {L: tt['X'+L][:, 0, :].astype(np.float64) for L in LAYERS}   # (feat, cond)
    return W, dec, x0, dims, tt

def rhs(x, t, W, dec):
    """引擎 Hill 场（未截断）。x: {L:(feat,cond)}"""
    dx = {}
    for L in LAYERS:
        z = np.zeros_like(x[L])
        for wname, src, tr in ASM[L]:
            A = W[wname].T if tr < 0 else W[wname]
            z = z + A @ x[src]
        prod = fhill(z)
        u = PULSE_A if abs(t - PULSE_T) < 0.5 else 0.0
        dx[L] = prod - dec[L][:, None] * x[L] + (u if L == 'G' else 0.0)
    return dx

def integrate(W, dec, x0, dt, method):
    n = int(round(TMAX / dt)); tgrid = np.arange(n) * dt
    traj = {L: np.zeros((x0[L].shape[0], n, x0[L].shape[1])) for L in LAYERS}
    x = {L: x0[L].copy() for L in LAYERS}
    for L in LAYERS: traj[L][:, 0, :] = x[L]
    for i in range(1, n):
        t = tgrid[i]
        if method == 'euler':
            k1 = rhs(x, t, W, dec); x = {L: x[L] + dt * k1[L] for L in LAYERS}
        else:  # rk4
            k1 = rhs(x, t, W, dec)
            x2 = {L: x[L] + 0.5*dt*k1[L] for L in LAYERS}; k2 = rhs(x2, t+0.5*dt, W, dec)
            x3 = {L: x[L] + 0.5*dt*k2[L] for L in LAYERS}; k3 = rhs(x3, t+0.5*dt, W, dec)
            x4 = {L: x[L] + dt*k3[L] for L in LAYERS};      k4 = rhs(x4, t+dt, W, dec)
            x = {L: x[L] + dt*(k1[L]+2*k2[L]+2*k3[L]+k4[L])/6.0 for L in LAYERS}
        for L in LAYERS: traj[L][:, i, :] = x[L]
    return traj, tgrid

def envelope(traj):
    return {L: (float(np.abs(traj[L]).max()), bool(np.isfinite(traj[L]).all())) for L in LAYERS}

def rel_linf(a, b):
    """逐层相对 L∞：max|a-b| / max|b|（b 为细网格基准）"""
    out = {}
    for L in LAYERS:
        num = float(np.abs(a[L]-b[L]).max()); den = float(np.abs(b[L]).max())
        out[L] = num / max(den, 1e-12)
    return out

def subsample(traj_fine, dt_fine, tt):
    """按真账各层 t 网格抽样（对齐真账 5/5/8/13 异构网格）"""
    tfine = np.arange(traj_fine['G'].shape[1]) * dt_fine
    out = {}
    for L in LAYERS:
        tL = tt['t'+L]
        idx = np.clip(np.searchsorted(tfine, tL), 0, len(tfine)-1)
        out[L] = traj_fine[L][:, idx, :]
    return out

def main():
    W, dec, x0, dims, tt = load_truth()
    os.makedirs(OUT, exist_ok=True)
    # 三份轨迹：euler0.5（原生）、euler0.25（减半基准）、rk4 0.5（对照档）
    te05, tg05 = integrate(W, dec, x0, 0.5, 'euler')
    te025, tg025 = integrate(W, dec, x0, 0.25, 'euler')
    trk, _ = integrate(W, dec, x0, 0.5, 'rk4')
    # 收敛验证页（未截断轨迹；抽样到真账各层网格比对）
    e05_sub = subsample(te05, 0.5, tt)
    e025_sub = subsample(te025, 0.25, tt)
    dev = rel_linf(e05_sub, e025_sub)
    env = envelope(te05)
    passed = all(v < 0.01 for v in dev.values())
    page = {'family':'Hill','dt_coarse':0.5,'dt_fine':0.25,'threshold_Linf':0.01,
            'unclipped':True,'rel_Linf_per_layer':dev,'pass':bool(passed),
            'envelope_absmax_finite':env,
            'note':'验证在未截断轨迹上进行；包络 absmax 与 finite 见上（证实未触任何截断）'}
    with open(os.path.join(OUT,'收敛验证页_hill.json'),'w') as f:
        json.dump(page, f, ensure_ascii=False, indent=1)
    print('CONVERGENCE_PAGE', json.dumps(page, ensure_ascii=False))
    # 双积分对照档落盘（欧拉+RK4 同初值同步长，抽样到真账网格）
    rk_sub = subsample(trk, 0.5, tt)
    for tag, dat in [('euler', e05_sub), ('rk4', rk_sub)]:
        np.savez(os.path.join(OUT, f'hill_traj_{tag}_dt0.5.npz'),
                 **{f'X{L}': dat[L] for L in LAYERS},
                 **{f't{L}': tt['t'+L] for L in LAYERS},
                 conditions=tt['conditions'], gene_ids=tt['gene_ids'], metab_ids=tt['metab_ids'])
    # 双份差（数值噪声地板）
    diff = rel_linf(e05_sub, rk_sub)
    print('DUAL_DIFF_euler_vs_rk4', json.dumps(diff))
    # 指纹
    hs = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.npz') or f.endswith('.json'):
            hs[f] = hashlib.md5(open(os.path.join(OUT,f),'rb').read()).hexdigest()[:8]
    print('FINGERPRINTS', json.dumps(hs))
    print('DATAGEN_DONE pass=%s' % passed)

if __name__ == '__main__':
    main()
