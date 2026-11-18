# -*- coding: utf-8 -*-
"""
训练脚本：GSE166336 四层组学动力学数字孪生 —— 四方案统一入口
============================================================
配套数据：train_tensors.npz（由 数据准备_从mmc3重建训练张量_带注释.py 生成）

四方案（--scheme 切换，互不依赖，可四卡并行）：
  方案1  已知通路子网：本地 kegg_kgml/ 全部通路的基因/代谢物并集为子网，全层训练
  方案2  删边绕行：同方案1，但先从 GErel 掩码删掉 Srebf1→Fasn、Srebf1→Scd1，
         检验模型是否经 Mlxipl(ChREBP) 等替代路径补偿（次级因果验证）
  方案3  双组学鲁棒：子网内只用 转录+代谢 两层训练，蛋白+磷酸化作留出测试
  方案4  全网络：14076基因 + 193代谢物，全层训练

动力学模型（连续时间，手动RK4积分，无外部ODE依赖）：
  G=转录 P=蛋白 S=磷酸化 M=代谢物，均为紧凑索引（14076/193）
  dG/dt = (W_GE ⊙ M_GErel)·tanh(G)            − d_G⊙G + b_G          [基因调控]
  dP/dt = β_diag·G                              − d_P⊙P + b_P          [翻译，β≥0]
  dS/dt = (W_PS ⊙ M_PS)·tanh(P) + (W_FB ⊙ M_FB)ᵀ·tanh(M) − d_S⊙S + b_S + u(t)
                                                             [信号/磷酸化 + 胰岛素输入]
  dM/dt = (W_EC ⊙ M_ECrel)ᵀ·tanh(P) + (W_MM ⊙ M_MM)·tanh(M) − d_M⊙M + b_M
                                                             [酶促生成/消耗 + 代谢转化]
  u(t)：仅 Insulin 条件、t≤T_INPUT 时作用于信号层（受体脉冲输入），PBS 为零

七项损失：4层掩码MSE + 稀疏L1 + 方向先验(GE/EC) + 轨迹平滑
掩码语义：掩码=“允许学习”，不断言边存在；假边由稀疏损失剪掉。
方向语义：±1 相对 KEGG 方程书写方向，是软先验（L_dir 惩罚反号权重），
          生理方向可与方程方向相反（已在进度文档记录，如 PEP）。

用法示例：
  python 训练脚本_四方案_动力学数字孪生_带注释.py --scheme 1 --gpu 0
  python 训练脚本_四方案_动力学数字孪生_带注释.py --scheme 4 --gpu 0 --epochs 500
输出：GSE166336_小鼠组学数据/scheme{k}/ 下：loss曲线、学到的边表csv、
      预测vs观测npz、模型ckpt、运行日志run_log.txt
"""

import os
import re
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

# ===================== 参数 =====================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", type=int, required=True, choices=[1, 2, 3, 4],
                   help="1子网 2删边绕行 3双组学鲁棒 4全网络")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="GSE166336_小鼠组学数据",
                   help="数据目录（服务器上务必放 /dfs/data 下，别放根目录）")
    p.add_argument("--kgml_dir", type=str, default="kegg_kgml",
                   help="KGML通路文件目录")
    p.add_argument("--cpu_threads", type=int, default=8,
                   help="共享8卡机限制CPU线程，防止线程争抢拖慢邻卡")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.5,
                   help="积分步长(min)，须整除所有观测时间点；0.5可覆盖全部网格")
    p.add_argument("--tmax", type=float, default=240.0)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率；480步RK4长链反传对lr敏感，1e-2实测发散，勿调大")
    # 损失权重
    p.add_argument("--w_sparse", type=float, default=1e-4)
    p.add_argument("--w_dir", type=float, default=1e-2)
    p.add_argument("--w_smooth", type=float, default=1e-5)
    p.add_argument("--w_prox", type=float, default=0.0,
                   help="近端软阈值(每epoch执行):w←sign(w)·relu(|w|−w_prox)。Adam+L1压不出精确零(实测:w_sparse×100边数纹丝不动),稀疏化必须显式prox;0=关闭")
    p.add_argument("--oracle_init", type=str, default="",
                   help="ground_truth.npz路径:以真值初始化全部参数(地形测试——真值起手若保不住=信息/噪声问题,保得住=优化器从零出发找不到)")
    p.add_argument("--oracle_noise", type=float, default=0.0,
                   help="oracle起手加噪std(权重单位;真值~N(0,0.4),σ=0.2/0.4=0.5/1×信号std)——吸引盆半径测绘")
    p.add_argument("--oracle_blind", action="store_true",
                   help="盲随机对照:不读真值,W=候选掩码内N(0,0.4)随机起手(σ→∞极限,检验'非零随机起手是否就够')")
    p.add_argument("--oracle_zconj", action="store_true",
                   help="z空间共轭oracle(需同用--oracle_init与--zstats_file):真值是物理空间的,模型跑在z分数空间——"
                        "decay不变;β^z=β·σG/σP;tanh边一阶共轭 w^z_ij=w_ij·σ_j·sech²(μ_j)/σ_i;偏置吸收"
                        "-d_i·μ_i/σ_i+边常数项Σw·tanh(μ_j)/σ_i;u^z=u/σ_S。检验:oracle保留不全是否源于raw/z尺度错配")
    p.add_argument("--zstats_file", type=str, default="",
                   help="zstats.npz路径(zmu_G/zsd_G/.../zmu_M/zsd_M,仿真脚本2026-09-03后版本生成)")
    p.add_argument("--freeze_fbps", action="store_true",
                   help="冻结探针:每步清零W_FB/W_PS梯度(钉死在初始值),prox亦跳过——腐蚀汇机制检验:"
                        "钉死留出层真值后,观测层拟合压力若仍污染不了留出层,留出R²应恢复")
    p.add_argument("--eval_heldout", action="store_true",
                   help="留出层R²评估模式:读 OUT_DIR/best.pt 滚双条件轨迹,对P/S层逐条件算R²后退出,"
                        "不训练不改ckpt——scheme 3 留出实验的判决数字(R²<0=比平线还差)")
    p.add_argument("--freeze_x0", action="store_true", help="冻x0")
    p.add_argument("--freeze_bias", action="store_true", help="冻b四件")
    p.add_argument("--init_prior", action="store_true",
                   help="先验初始化(处方实测):候选边=方向先验定号(无先验处随机号)×init_prior_mag——模拟'文献先验能买到的起手',测它落在吸引盆内还是盆外")
    p.add_argument("--init_prior_mag", type=float, default=0.3)
    p.add_argument("--init_mag", type=float, default=0.4,
                   help="oracle_blind初始化幅值(原硬编码0.4;事故后改可调)")
    p.add_argument("--init_fanin", action="store_true",
                   help="起手按目标节点候选入度做 Xavier 式归一(1/√indegree;blind/prior 用)——稠密起手不归一,~594 路输入求和直接炸掉反传(NaN 事故根源)")
    # 方案2：要删的边（基因symbol对）
    p.add_argument("--remove_edges", type=str, default="Srebf1>Fasn,Srebf1>Scd1",
                   help="格式 TF>Target,TF>Target；仅方案2生效")
    p.add_argument("--t_input", type=float, default=10.0,
                   help="胰岛素脉冲输入持续时长(min)")
    p.add_argument("--seed", type=int, default=0)
    # --- 半合成仿真专用 ---
    p.add_argument("--tensor_file", type=str, default=None,
                   help="默认用 data_dir/train_tensors.npz；仿真时指向 data_sim/train_tensors.npz")
    p.add_argument("--tag", type=str, default="",
                   help="输出目录后缀，如 _sim → scheme1_sim，防覆盖真实数据结果")
    return p.parse_args()

ARGS = get_args()
torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.set_num_threads(ARGS.cpu_threads)     # 共享8卡机：限制CPU线程防争抢
DEV = torch.device(f"cuda:{ARGS.gpu}" if torch.cuda.is_available() else "cpu")

DATA_DIR = ARGS.data_dir
KGML_DIR = ARGS.kgml_dir
if not os.path.isdir(KGML_DIR) and os.path.isdir(os.path.join(DATA_DIR, "kegg_kgml")):
    KGML_DIR = os.path.join(DATA_DIR, "kegg_kgml")
if not os.path.isdir(DATA_DIR):
    raise SystemExit(f"找不到数据目录: {DATA_DIR}（用 --data_dir 指定，服务器上建议 /dfs/data/...）")
OUT_DIR = os.path.join(DATA_DIR, f"scheme{ARGS.scheme}{ARGS.tag}")   # tag防覆盖真实运行
os.makedirs(OUT_DIR, exist_ok=True)

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))
    with open(os.path.join(OUT_DIR, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

log("=" * 60)
log(f"方案{ARGS.scheme} | 设备 {DEV} | dt={ARGS.dt} tmax={ARGS.tmax} | {time.strftime('%F %T')}")
log("=" * 60)

def locate(fname):
    for cand in [fname, os.path.join(DATA_DIR, fname)]:
        if os.path.exists(cand):
            return cand
    return None

# ===================== 1. 读训练张量 =====================
# --tensor_file 指定时用仿真张量，否则用 data_dir 下的真实 train_tensors.npz
_tensor_path = ARGS.tensor_file if ARGS.tensor_file else locate("train_tensors.npz")
log(f"训练张量: {_tensor_path}")
npz = np.load(_tensor_path, allow_pickle=True)
gene_ids  = [str(g) for g in npz["gene_ids"]]      # 14076
metab_ids = [str(m) for m in npz["metab_ids"]]     # 193
nG_full, nM_full = len(gene_ids), len(metab_ids)
gpos = {g: i for i, g in enumerate(gene_ids)}
mpos = {m: i for i, m in enumerate(metab_ids)}
log(f"紧凑索引: 基因 {nG_full}, 代谢物 {nM_full}")

layers = {}
for key in ["G", "P", "S", "M"]:
    layers[key] = dict(X=npz[f"X{key}"], t=npz[f"t{key}"], O=npz[f"O{key}"])

# ===================== 2. 读掩码并切紧凑索引 =====================
def load_mask(name):
    p = locate(name)
    return np.load(p) if p else None

mask_GE = load_mask("mask_GErel.npy")                       # (27999,27999) int8
mask_EC = load_mask("mask_ECrel.npy")                       # (27999,194) float32
dir_EC  = load_mask("direction_mask.npy")                   # 同形 ±1/0
mask_MM = load_mask("mask_metabolite2metabolite.npy")       # (194,194) int8
mask_FB = load_mask("mask_feedback.npy")                    # (194,27999) int8
dir_GE  = load_mask("direction_GErel.npy")                  # (27999,27999) int8

# 全节点→紧凑 的映射（节点文件 dict 规则 = 唯一ID保持相对顺序，与索引map一致）
gene_nodes  = np.array(gene_ids)   # mmu: 唯一
metab_nodes = np.array(metab_ids)

def compact_slice_GE(mask):
    """27999节点序 → 14076紧凑序：直接按唯一ID首次出现行抽取。
    掩码构建与节点文件同源：用索引map做行/列选择最稳妥。"""
    with open(locate("gene_index_map.json"), encoding="utf-8") as f:
        gmap = json.load(f)
    with open(locate("metabolite_index_map.json"), encoding="utf-8") as f:
        mmap = json.load(f)
    gs = np.array([gmap[g] for g in gene_ids])   # 紧凑→全节点行号
    ms = np.array([mmap[m] for m in metab_ids])
    return gs, ms

gs_full, ms_full = compact_slice_GE(None)
M_GE = torch.tensor(mask_GE[np.ix_(gs_full, gs_full)].astype(np.float32), device=DEV)
D_GE = torch.tensor(dir_GE[np.ix_(gs_full, gs_full)].astype(np.float32), device=DEV) if dir_GE is not None else torch.zeros_like(M_GE)
M_EC = torch.tensor(mask_EC[np.ix_(gs_full, ms_full)].astype(np.float32), device=DEV)   # (G,M) 酶→代谢物
D_EC = torch.tensor(dir_EC[np.ix_(gs_full, ms_full)].astype(np.float32), device=DEV) if dir_EC is not None else torch.zeros_like(M_EC)
M_MM = torch.tensor(mask_MM[np.ix_(ms_full, ms_full)].astype(np.float32), device=DEV)
M_FB = torch.tensor(mask_FB[np.ix_(ms_full, gs_full)].astype(np.float32), device=DEV)   # (M,G) 代谢物→信号蛋白

# β 翻译对角：转录/蛋白任一层有观测的基因允许学习
obs_G = torch.tensor((np.nan_to_num(layers["G"]["O"]).sum(axis=(1, 2)) > 0), device=DEV)
obs_P = torch.tensor((np.nan_to_num(layers["P"]["O"]).sum(axis=(1, 2)) > 0), device=DEV)
beta_mask = (obs_G | obs_P).float()

# P→S 信号层掩码：feedback 白名单命中的信号基因集合内允许全连接（L1剪枝）
sig_genes = (M_FB.sum(dim=0) > 0)
M_PS = (sig_genes[:, None] & sig_genes[None, :]).float()
log(f"掩码规模: GE {int(M_GE.sum())}, EC {int(M_EC.sum())}, MM {int(M_MM.sum())}, "
    f"FB {int(M_FB.sum())}, β对角 {int(beta_mask.sum())}, 信号基因 {int(sig_genes.sum())}")

# ===================== 3. 方案配置 =====================
# --- 子网：本地 kegg_kgml\ 全部通路节点的并集（已知通路先验子网） ---
def subnet_from_kgml():
    genes, mets = set(), set()
    for fn in sorted(os.listdir(KGML_DIR)):
        if not fn.endswith((".xml", ".kgml")):
            continue
        import xml.etree.ElementTree as ET
        root = ET.parse(os.path.join(KGML_DIR, fn)).getroot()
        for ent in root.iter("entry"):
            t = ent.get("type", "")
            name = ent.get("name", "")
            if t == "gene":
                for tok in name.split():
                    if tok.startswith("mmu:") and tok in gpos:
                        genes.add(tok)
            elif t == "compound":
                for tok in name.split():
                    tok = tok.replace("cpd:", "")
                    if re.fullmatch(r"C\d{5}", tok) and tok in mpos:
                        mets.add(tok)
    return sorted(genes, key=lambda g: gpos[g]), sorted(mets, key=lambda m: mpos[m])

if ARGS.scheme in (1, 2, 3):
    sub_g, sub_m = subnet_from_kgml()
    gi = torch.tensor([gpos[g] for g in sub_g], device=DEV, dtype=torch.long)
    mi = torch.tensor([mpos[m] for m in sub_m], device=DEV, dtype=torch.long)
    log(f"子网: {len(sub_g)} 基因, {len(sub_m)} 代谢物（本地KGML通路并集）")
else:
    gi = torch.arange(nG_full, device=DEV)
    mi = torch.arange(nM_full, device=DEV)
    log(f"全网络: {nG_full} 基因, {nM_full} 代谢物")

def sub_mask(M, ri, ci):
    return M.index_select(0, ri).index_select(1, ci)

# 切片到本方案的作用域
sGE = sub_mask(M_GE, gi, gi);  sD_GE = sub_mask(D_GE, gi, gi)
sEC = sub_mask(M_EC, gi, mi);  sD_EC = sub_mask(D_EC, gi, mi)
sMM = sub_mask(M_MM, mi, mi)
sFB = sub_mask(M_FB, mi, gi)
sPS = sub_mask(M_PS, gi, gi)
sBeta = beta_mask.index_select(0, gi)

# --- 方案2：删边（symbol→mmu 用缓存的 list/mmu 翻译） ---
def symbol_to_mmu():
    cache = json.load(open(locate("kegg_api_cache.json"), encoding="utf-8"))
    for k, v in cache.items():
        if "list/mmu" in k:
            txt = v if isinstance(v, str) else v.get("text", "")
            m = {}
            for line in txt.splitlines():
                parts = line.split("\t")
                if len(parts) >= 4 and parts[1] == "CDS":
                    desc = parts[3]
                elif len(parts) >= 2:
                    desc = parts[1]
                else:
                    continue
                sym_part = desc.split(";")[0]
                for s in sym_part.split(","):
                    s = s.strip()
                    if s:
                        m.setdefault(s, parts[0].strip())
            return m
    return {}

if ARGS.scheme == 2:
    s2m = symbol_to_mmu()
    removed = []
    for pair in ARGS.remove_edges.split(","):
        tf, tg = pair.split(">")
        a, b = s2m.get(tf), s2m.get(tg)
        if a in gpos and b in gpos:
            ia = (gi == gpos[a]).nonzero().flatten()
            ib = (gi == gpos[b]).nonzero().flatten()
            if len(ia) and len(ib) and sGE[ia[0], ib[0]] > 0:
                sGE[ia[0], ib[0]] = 0
                sD_GE[ia[0], ib[0]] = 0
                removed.append(f"{a}({tf})→{b}({tg})")
    log(f"方案2删边: {removed if removed else '⚠ 未找到可删边，检查symbol映射'}")
    log("假设: 删除SREBP1路径后，模型应经 Mlxipl(ChREBP) 等替代调控补偿脂质合成")

# --- 方案3：训练层 / 留出层 ---
if ARGS.scheme == 3:
    train_layers = {"G", "M"}
    log("方案3: 仅用 转录+代谢 训练；蛋白+磷酸化 留出作独立测试")
else:
    train_layers = {"G", "P", "S", "M"}

# ===================== 4. 观测整理（按本方案作用域切片） =====================
dt, tmax = ARGS.dt, ARGS.tmax
n_step = int(round(tmax / dt))
# 检查所有观测时间点落在积分网格上
for key in layers:
    tt = layers[key]["t"]
    assert np.allclose(tt / dt, np.round(tt / dt), atol=1e-6), \
        f"层{key} 的时间点 {tt} 不能被 dt={dt} 整除，请调小dt"

obs = {}
gidx_np = gi.cpu().numpy()
midx_np = mi.cpu().numpy()
for key in layers:
    X = torch.tensor(np.nan_to_num(layers[key]["X"]), dtype=torch.float32, device=DEV)
    O = torch.tensor(layers[key]["O"].astype(np.float32), device=DEV)
    sel = torch.tensor(gidx_np if key != "M" else midx_np, device=DEV, dtype=torch.long)
    t_idx = torch.tensor(np.round(layers[key]["t"] / dt).astype(int), device=DEV)
    obs[key] = dict(X=X.index_select(0, sel), O=O.index_select(0, sel), t_idx=t_idx)

# ===================== 5. 模型 =====================
class SparseMat(nn.Module):
    """掩码候选边的边表稀疏参数化（方案④全网络专用）。
    稠密 14076²×3 个矩阵：名义参数 6 亿，且 481 步 RK4 反传要把每步的掩码乘积
    （每张 792MB）全部存下 → 约 4.5TB 显存，必爆（方案④首跑 OOM 实测）。
    候选边仅 ~37 万条，边表化后：参数 = 候选边数，前向 = gather + index_add，
    显存降三个数量级。autograd 安全性：gather 与 index_add 均有反向实现。
    """
    def __init__(self, mask, dir_mask=None):
        super().__init__()
        rc = torch.nonzero(mask > 0)                     # (nnz, 2)，掩码已在 DEV 上
        self.register_buffer("rows", rc[:, 0].contiguous())   # to 端（输出行）
        self.register_buffer("cols", rc[:, 1].contiguous())   # from 端（输入列）
        self.val = nn.Parameter(torch.zeros(rc.shape[0], device=DEV))
        self.n_out, self.n_in = mask.shape
        d = dir_mask[self.rows, self.cols].float() if dir_mask is not None \
            else torch.zeros(rc.shape[0], device=DEV)
        self.register_buffer("dirv", d)                  # 逐边方向先验（±1/0）

    def matvec(self, x):
        # y = (W ⊙ M) @ x：y[to] += w · x[from]
        contrib = self.val * x[self.cols]
        return torch.zeros(self.n_out, device=x.device).index_add_(0, self.rows, contrib)

    def mattvec(self, x):
        # y = (W ⊙ M)ᵀ @ x：y[from] += w · x[to]
        contrib = self.val * x[self.rows]
        return torch.zeros(self.n_in, device=x.device).index_add_(0, self.cols, contrib)


class Twin(nn.Module):
    """白盒动力学数字孪生：权重全部被先验掩码约束（前向时相乘）。"""
    def __init__(self, n_g, n_m):
        super().__init__()
        self.n_g, self.n_m = n_g, n_m
        # n_g>3000（方案④全网络）自动走稀疏边表；子网方案①②③保持稠密不动，
        # 保证与已完成运行的数值路径完全一致（结果可比性）
        self.sparse = n_g > 3000
        z = lambda *s: torch.zeros(*s, device=DEV)
        if self.sparse:
            self.sGE_sp = SparseMat(sGE, sD_GE)   # 基因调控（含方向先验）
            self.sSG_sp = SparseMat(sGE)          # 信号→转录，复用GErel拓扑
            self.sEC_sp = SparseMat(sEC, sD_EC)   # 酶促（含方向先验）
            self.sMM_sp = SparseMat(sMM)          # 代谢转化
            self.sFB_sp = SparseMat(sFB)          # 代谢物→信号反馈
            self.sPS_sp = SparseMat(sPS)          # 信号级联
        else:
            self.W_GE = nn.Parameter(z(n_g, n_g))
            # 信号→转录耦合：TF磷酸化活性(S层)驱动转录，复用GErel掩码拓扑
            # （v2新增：方案①首跑确诊G层无输入边学不动，生物上=胰岛素→信号→TF磷酸化→转录）
            self.W_SG = nn.Parameter(z(n_g, n_g))
            self.W_EC = nn.Parameter(z(n_g, n_m))
            self.W_MM = nn.Parameter(z(n_m, n_m))
            self.W_FB = nn.Parameter(z(n_m, n_g))
            self.W_PS = nn.Parameter(z(n_g, n_g))
        self.beta = nn.Parameter(torch.full((n_g,), 0.5, device=DEV))   # 软plus前
        self.b_G, self.b_P = nn.Parameter(z(n_g)), nn.Parameter(z(n_g))
        self.b_S, self.b_M = nn.Parameter(z(n_g)), nn.Parameter(z(n_m))
        # 衰减率用 raw 参数 + softplus 保证非负
        self.d_raw = nn.Parameter(torch.full((4,), -2.0, device=DEV))
        # 胰岛素脉冲输入（仅作用于信号基因集合）
        self.u = nn.Parameter(z(n_g))
        # 初态（2条件 × 状态维），t=0观测值会通过数据损失锚定
        self.x0 = nn.Parameter(z(2, 2 * n_g + n_m + n_g))
        self.register_buffer("zeros_dummy", torch.zeros(1, device=DEV))

    def decay(self):
        # 衰减率下限 1e-3：细胞是耗散系统，最小周转率是热力学一致性的软约束；
        # 无下限则衰减可学到 0，正反馈环路（尤其磷酸化层）会爆炸——实测教训
        return torch.nn.functional.softplus(self.d_raw) + 1e-3

    def rhs(self, x, t, cond):
        n_g, n_m = self.n_g, self.n_m
        G = x[:n_g]; P = x[n_g:2 * n_g]; S = x[2 * n_g:2 * n_g + n_g]
        M = x[2 * n_g + n_g:2 * n_g + n_g + n_m]
        G_ = torch.tanh(G); P_ = torch.tanh(P); M_ = torch.tanh(M)
        if self.sparse:
            dG = self.sGE_sp.matvec(G_) + self.sSG_sp.matvec(torch.tanh(S)) - self.decay()[0] * G + self.b_G
        else:
            dG = (self.W_GE * sGE) @ G_ + (self.W_SG * sGE) @ torch.tanh(S) - self.decay()[0] * G + self.b_G
        dP = torch.nn.functional.softplus(self.beta) * sBeta * G - self.decay()[1] * P + self.b_P
        if self.sparse:
            dS = self.sPS_sp.matvec(P_) + self.sFB_sp.mattvec(M_) - self.decay()[2] * S + self.b_S
        else:
            dS = (self.W_PS * sPS) @ P_ + (self.W_FB * sFB).t() @ M_ - self.decay()[2] * S + self.b_S
        # 胰岛素脉冲：仅 Insulin(cond=0) 且 t<=t_input
        if cond == 0 and t <= ARGS.t_input:
            dS = dS + self.u * sig_gate
        if self.sparse:
            dM = self.sEC_sp.mattvec(P_) + self.sMM_sp.matvec(M_) - self.decay()[3] * M + self.b_M
        else:
            dM = (self.W_EC * sEC).t() @ P_ + (self.W_MM * sMM) @ M_ - self.decay()[3] * M + self.b_M
        return torch.cat([dG, dP, dS, dM])

def rk4(f, x, t, h, cond):
    k1 = f(x, t, cond)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h, cond)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h, cond)
    k4 = f(x + h * k3, t + h, cond)
    return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# ===================== 6. 训练 =====================
model = Twin(len(gi), len(mi)).to(DEV)
if ARGS.freeze_x0:
    model.x0.requires_grad_(False); log("x0冻结探针: 初值钉死")
if ARGS.freeze_bias:
    for _p in [model.b_G, model.b_P, model.b_S, model.b_M]:
        _p.requires_grad_(False)
    log("偏置冻结探针: b_G/b_P/b_S/b_M 钉死")

# 信号层胰岛素输入门：只允许 feedback 命中的信号基因接收输入
sig_gate = (sPS.sum(dim=0) > 0).float().to(DEV)

opt = torch.optim.Adam(model.parameters(), lr=ARGS.lr)

# ---- Oracle 初始化:以真值起手(可辨识性地形测试;仅稠密/子网路径) ----
if ARGS.oracle_init:
    _gt = np.load(ARGS.oracle_init)
    assert not model.sparse, "oracle 初始化仅实现稠密(子网)路径"
    for _k in ["W_GE", "W_SG", "W_EC", "W_MM", "W_FB", "W_PS",
               "beta_true", "decay_true", "b_true", "u_true"]:
        assert _k in _gt, f"ground_truth 缺键 {_k}——需用 2026-09-02 后版本的仿真脚本重新生成"
    with torch.no_grad():
        if ARGS.oracle_zconj:
            # ---- z空间一阶共轭:真值(物理空间)→模型实际运行的z分数空间 ----
            # x = σ·x_z + μ ⇒ dx_z/dt = (1/σ)·dx/dt
            #   线性项(decay/β生成): 系数按σ比值缩放,常数项进偏置;decay本身不变
            #   tanh边: tanh(σ_j·x_zj+μ_j) ≈ tanh(μ_j)+σ_j·sech²(μ_j)·x_zj  (x_z=0处一阶)
            assert ARGS.zstats_file, "--oracle_zconj 需 --zstats_file 指向 zstats.npz"
            _zs = np.load(ARGS.zstats_file)
            mu, sd = {}, {}
            _n_nan = 0
            for k in "GPSM":
                _m = _zs[f"zmu_{k}"].astype(np.float64)
                _s = _zs[f"zsd_{k}"].astype(np.float64)
                # 全缺失特征(整个节点无观测)的 z 统计量为 NaN——X 里它们被 nan_to_num 洗成 0,
                # 此处按同一中性约定净化: mu=0, sd=1(该特征恒为 z=0,tanh(0)=0,对动力学无贡献)
                _bad = ~np.isfinite(_m) | ~np.isfinite(_s) | (_s < 1e-8)
                _n_nan += int(_bad.sum())
                mu[k] = np.where(_bad, 0.0, _m)
                sd[k] = np.where(_bad, 1.0, _s)
            if _n_nan:
                log(f"zstats净化: {_n_nan} 个全缺失特征按 mu=0/sd=1 中性约定处理")
            th = {k: np.tanh(mu[k]) for k in "GPSM"}
            sh2 = {k: 1.0 - th[k] ** 2 for k in "GPSM"}          # sech²(μ)=1−tanh²(μ)
            _W0 = {nm: _gt[f"W_{nm}"].astype(np.float64) for nm in ["GE", "SG", "EC", "MM", "FB", "PS"]}
            _bt = _gt["b_true"].astype(np.float64)
            _sp = np.log1p(np.exp(_gt["beta_true"].astype(np.float64)))   # softplus(β_true)
            _dc = _gt["decay_true"].astype(np.float64)                    # (4,)
            _bG, _bP = _bt[:model.n_g], _bt[model.n_g:2 * model.n_g]
            _bS, _bM = _bt[2 * model.n_g:3 * model.n_g], _bt[3 * model.n_g:]
            # 边权共轭: w^z_ij = w_ij·σ_j·sech²(μ_j)/σ_i (EC/FB 按 [源,目标] 存储,行=源)
            _Wz = {
                "GE": _W0["GE"] * (sd["G"] * sh2["G"])[None, :] / sd["G"][:, None],
                "SG": _W0["SG"] * (sd["S"] * sh2["S"])[None, :] / sd["G"][:, None],
                "PS": _W0["PS"] * (sd["P"] * sh2["P"])[None, :] / sd["S"][:, None],
                "MM": _W0["MM"] * (sd["M"] * sh2["M"])[None, :] / sd["M"][:, None],
                "FB": _W0["FB"] * (sd["M"] * sh2["M"])[:, None] / sd["S"][None, :],
                "EC": _W0["EC"] * (sd["P"] * sh2["P"])[:, None] / sd["M"][None, :],
            }
            for _nm in ["GE", "SG", "EC", "MM", "FB", "PS"]:
                getattr(model, f"W_{_nm}").data.copy_(torch.tensor(_Wz[_nm], device=DEV))
            # β^z: softplus(β^z)=softplus(β)·σG/σP(逐基因);decay 不变;偏置吸收全部常数项
            model.beta.data.copy_(torch.log(torch.expm1(torch.tensor(
                _sp * sd["G"] / sd["P"], device=DEV))))
            model.d_raw.data.copy_(torch.log(torch.expm1(torch.tensor(_dc, device=DEV) - 1e-3)))
            model.b_G.data.copy_(torch.tensor(
                (_bG + _W0["GE"] @ th["G"] + _W0["SG"] @ th["S"] - _dc[0] * mu["G"]) / sd["G"], device=DEV))
            model.b_P.data.copy_(torch.tensor(
                (_bP + _sp * mu["G"] - _dc[1] * mu["P"]) / sd["P"], device=DEV))
            model.b_S.data.copy_(torch.tensor(
                (_bS + _W0["PS"] @ th["P"] + _W0["FB"].T @ th["M"] - _dc[2] * mu["S"]) / sd["S"], device=DEV))
            model.b_M.data.copy_(torch.tensor(
                (_bM + _W0["EC"].T @ th["P"] + _W0["MM"] @ th["M"] - _dc[3] * mu["M"]) / sd["M"], device=DEV))
            model.u.data.copy_(torch.tensor(_gt["u_true"].astype(np.float64) / sd["S"], device=DEV))
            log(f"z共轭oracle: W/β/b/u 已变换到z分数空间(decay不变);sBeta掩码与raw臂同口径未并入")
        else:
            for _nm in ["GE", "SG", "EC", "MM", "FB", "PS"]:
                getattr(model, f"W_{_nm}").data.copy_(torch.tensor(_gt[f"W_{_nm}"], device=DEV))
            # softplus 逆变换:softplus(β)=β_true → β=ln(exp(β_true)−1);decay 同理(含 1e-3 下界)
            model.beta.data.copy_(torch.log(torch.expm1(torch.tensor(_gt["beta_true"], device=DEV))))
            model.d_raw.data.copy_(torch.log(torch.expm1(torch.tensor(_gt["decay_true"], device=DEV) - 1e-3)))
            _b = torch.tensor(_gt["b_true"], device=DEV)
            model.b_G.data.copy_(_b[:model.n_g]); model.b_P.data.copy_(_b[model.n_g:2*model.n_g])
            model.b_S.data.copy_(_b[2*model.n_g:3*model.n_g]); model.b_M.data.copy_(_b[3*model.n_g:])
            model.u.data.copy_(torch.tensor(_gt["u_true"], device=DEV))
        # 吸引盆半径测绘:在真值上叠加掩码内高斯噪声(σ 单位=权重;真值 std=0.4)
        if ARGS.oracle_noise > 0:
            for _nm, _M in {"GE": sGE, "SG": sGE, "EC": sEC, "MM": sMM, "FB": sFB, "PS": sPS}.items():
                _W = getattr(model, f"W_{_nm}")
                _W.data += torch.randn_like(_W) * ARGS.oracle_noise * _M
            log(f"Oracle 加噪: σ={ARGS.oracle_noise}(掩码内,信号std=0.4)")
    log(f"Oracle初始化: {ARGS.oracle_init}(W/β/decay/b/u=真值,x0 仍可学习)")

# ---- 盲随机对照:掩码内 N(0,0.4) 起手,不读真值(σ→∞ 极限点) ----
# 掩码朝向约定: GE/SG/MM/PS 行=目标(入度按行求和); EC/FB 转置使用,目标在列
def _fanin_scale(M, axis):
    """Xavier 式入度归一:每个目标节点除以√候选入度,防稠密起手正反向爆炸"""
    return 1.0 / M.sum(dim=axis, keepdim=True).clamp(min=1.0).sqrt()

_INIT_AX = {"GE": (sGE, 1), "SG": (sGE, 1), "EC": (sEC, 0),
            "MM": (sMM, 1), "FB": (sFB, 0), "PS": (sPS, 1)}

if ARGS.oracle_blind:
    assert not model.sparse, "oracle_blind 仅实现稠密(子网)路径"
    with torch.no_grad():
        for _nm, (_M, _ax) in _INIT_AX.items():
            _W = getattr(model, f"W_{_nm}")
            _sc = _fanin_scale(_M, _ax) if ARGS.init_fanin else 1.0
            _W.data = torch.randn_like(_W) * ARGS.init_mag * _M * _sc
    log(f"Oracle-blind 初始化: W=掩码内 N(0,{ARGS.init_mag}) 随机(不读真值){', 入度归一' if ARGS.init_fanin else ''}")
    with torch.no_grad():
        for _nm2, (_M2, _ax2) in _INIT_AX.items():
            _Wm = getattr(model, f"W_{_nm2}") * _M2
            log(f"  生效检验 {_nm2}: 掩码内|W| mean={_Wm.abs().sum().item()/_M2.sum().item():.4f} max={_Wm.abs().max().item():.4f} 边数={int(_M2.sum().item())}")

# ---- 先验初始化(处方实测):从业者手里只有方向先验,没有真值——这个起手落盆吗? ----
if ARGS.init_prior:
    assert not model.sparse, "init_prior 仅实现稠密(子网)路径"
    with torch.no_grad():
        for _nm, (_M, _ax) in _INIT_AX.items():
            _D = {"GE": sD_GE, "EC": sD_EC}.get(_nm)
            _W = getattr(model, f"W_{_nm}")
            _rs = torch.where(torch.rand_like(_W) < 0.5, -1.0, 1.0)          # 无先验处随机定号
            _sign = _rs if _D is None else torch.where(_D != 0, torch.sign(_D), _rs)
            _sc = _fanin_scale(_M, _ax) if ARGS.init_fanin else 1.0
            _W.data = ARGS.init_prior_mag * _sign * _M * _sc
    log(f"先验初始化: 幅度 {ARGS.init_prior_mag}(GE/EC 方向先验定号,其余随机号){', 入度归一' if ARGS.init_fanin else ''}")
    with torch.no_grad():
        for _nm2, (_M2, _ax2) in _INIT_AX.items():
            _Wm = getattr(model, f"W_{_nm2}") * _M2
            log(f"  生效检验 {_nm2}: 掩码内|W| mean={_Wm.abs().sum().item()/_M2.sum().item():.4f} max={_Wm.abs().max().item():.4f} 边数={int(_M2.sum().item())}")

# ---- 冻结探针:W_FB/W_PS 钉死在初始值(腐蚀汇机制检验臂) ----
# 假设:留出层边被"观测层拟合压力"主动污染(oracle-s3: FB/PS 1.0→0.62/0.59,留出R² −65/−370)。
# 钉死留出层真值后若留出 R² 恢复=污染确经梯度通道;若仍崩=另有通路(如 x0/偏置代偿)。
if ARGS.freeze_fbps:
    assert not model.sparse, "freeze_fbps 仅实现稠密(子网)路径"
    log("冻结探针: W_FB/W_PS 全程钉死在初始值(梯度每步清零,prox 跳过)")
n_g, n_m = len(gi), len(mi)

def rollout(cond):
    """积分一条轨迹，返回 (n_step+1, state_dim)"""
    x = model.x0[cond]
    traj = [x]
    t = 0.0
    for _ in range(n_step):
        x = rk4(model.rhs, x, t, dt, cond)
        traj.append(x)
        t += dt
    return torch.stack(traj)          # (n_step+1, dim)

def data_loss(trajs):
    """逐层逐条件掩码MSE；trajs=[traj_cond0, traj_cond1]"""
    total, detail = 0.0, {}
    for key in ["G", "P", "S", "M"]:
        if key not in train_layers:
            continue
        X, O, t_idx = obs[key]["X"], obs[key]["O"], obs[key]["t_idx"]
        lk = 0.0
        for cond in range(2):
            sl = dict(G=trajs[cond][:, :n_g], P=trajs[cond][:, n_g:2 * n_g],
                      S=trajs[cond][:, 2 * n_g:2 * n_g + n_g],
                      M=trajs[cond][:, 2 * n_g + n_g:])[key]
            pred = sl[t_idx]                              # (T, n)
            diff2 = (pred.t() - X[:, :, cond]) ** 2       # (n, T)
            m = O[:, :, cond]
            lk = lk + (diff2 * m).sum() / m.sum().clamp(min=1.0)
        detail[key] = lk.item()
        total = total + lk
    return total, detail

def reg_loss():
    """稀疏 + 方向 + 平滑"""
    if model.sparse:
        # 边表化后 val 只存在于候选边上，|val| 求和 ≡ 稠密版掩码内 L1
        l_sp = (model.sGE_sp.val.abs().sum() + model.sSG_sp.val.abs().sum()
                + model.sEC_sp.val.abs().sum() + model.sMM_sp.val.abs().sum()
                + model.sFB_sp.val.abs().sum() + model.sPS_sp.val.abs().sum())
        l_dir = (torch.relu(-model.sGE_sp.dirv * model.sGE_sp.val).sum()
                 + torch.relu(-model.sEC_sp.dirv * model.sEC_sp.val).sum())
    else:
        l_sp = (model.W_GE.abs().sum() + model.W_EC.abs().sum() + model.W_MM.abs().sum()
                + model.W_FB.abs().sum() + model.W_PS.abs().sum() + model.W_SG.abs().sum())
        l_dir = (torch.relu(-sD_GE * model.W_GE).sum() + torch.relu(-sD_EC * model.W_EC).sum())
    return ARGS.w_sparse * l_sp, ARGS.w_dir * l_dir

# ---- 留出层 R² 评估模式:读 best.pt 滚轨迹,对留出层逐条件算 R²,不训练不碰 ckpt ----
# R² = 1 − SS_res/SS_tot,SS_tot 以该层观测点均值为基准(即"平坦预测");R²<0 = 比平线还差=臆造动态
if ARGS.eval_heldout:
    _ck = os.path.join(OUT_DIR, "best.pt")
    assert os.path.isfile(_ck), f"找不到 {_ck}——先训练再评估"
    model.load_state_dict(torch.load(_ck, map_location=DEV))
    model.eval()
    with torch.no_grad():
        _trajs = [rollout(0), rollout(1)]
    print(f"\n留出层 R² 评估: {_ck}")
    print("层    条件0(Insulin)    条件1(对照)")
    for _k in ["P", "S"]:
        if _k not in obs:
            continue
        _X, _O, _ti = obs[_k]["X"], obs[_k]["O"], obs[_k]["t_idx"]
        _row = []
        for _c in range(2):
            _sl = dict(G=_trajs[_c][:, :n_g], P=_trajs[_c][:, n_g:2 * n_g],
                       S=_trajs[_c][:, 2 * n_g:2 * n_g + n_g], M=_trajs[_c][:, 2 * n_g + n_g:])[_k]
            _pred = _sl[_ti].t()                              # (n, T)，与 data_loss 同切片
            _m = _O[:, :, _c] > 0.5
            _yc = _X[:, :, _c]
            _res = (((_pred - _yc) ** 2) * _m).sum()
            _mu = (_yc * _m).sum() / _m.sum().clamp(min=1.0)
            _tot = (((_yc - _mu) ** 2) * _m).sum()
            _row.append(1.0 - (_res / _tot.clamp(min=1e-12)).item())
        print(f"{_k}      {_row[0]:+.2f}           {_row[1]:+.2f}")
    raise SystemExit(0)

log(f"可学习参数: {sum(p.numel() for p in model.parameters())}")
t0 = time.time()
best = float("inf")
for ep in range(1, ARGS.epochs + 1):
    opt.zero_grad()
    trajs = [rollout(0), rollout(1)]
    l_data, det = data_loss(trajs)
    l_sp, l_dir = reg_loss()
    # 轨迹平滑（二阶差分，仅子网默认开；全网络权重已很小）
    l_sm = 0.0
    for tr in trajs:
        d2 = tr[2:] - 2 * tr[1:-1] + tr[:-2]
        l_sm = l_sm + (d2 ** 2).mean()
    loss = l_data + l_sp + l_dir + ARGS.w_smooth * l_sm
    loss.backward()
    # 梯度非有限→跳过本步(blind/prior 事故:稠密大初始化使 481 步 RK4 反传爆炸,
    # NaN 梯度经裁剪失败→Adam 把全部参数污染成 NaN;发现靠"两臂 best.pt 字节全同")
    _g_ok = all(bool(torch.isfinite(pp.grad).all().item()) for pp in model.parameters() if pp.grad is not None)
    if not _g_ok:
        opt.zero_grad()
        log(f"ep{ep:4d} 梯度非有限,跳过本步(模型保持上一有限状态)")
        continue
    # 冻结探针:裁剪前清零(冻结矩阵不计入裁剪范数;Adam 见 None 梯度整体跳过该参数)
    if ARGS.freeze_fbps:
        model.W_FB.grad = None
        model.W_PS.grad = None
    # 梯度裁剪：480步RK4长链反传梯度易爆（实测 lr=1e-2 无裁剪 ep20 发散），必开
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    # 近端软阈值:Adam+L1 压不出精确零(实测 w_sparse×100 边数纹丝不动——逐参数归一化使更新
    # 幅值≈lr、权重绕0恒幅振荡,永远到不了导出阈值),稀疏化只能靠显式 prox 步
    if ARGS.w_prox > 0:
        with torch.no_grad():
            _mats = ([model.sGE_sp.val, model.sSG_sp.val, model.sEC_sp.val,
                      model.sMM_sp.val, model.sFB_sp.val, model.sPS_sp.val] if model.sparse
                     else [model.W_GE, model.W_SG, model.W_EC, model.W_MM]
                          + ([] if ARGS.freeze_fbps else [model.W_FB, model.W_PS]))
            for _W in _mats:
                _W.data = torch.sign(_W.data) * torch.relu(_W.data.abs() - ARGS.w_prox)
    # 保存前双有限检查:loss 有限 + 参数有限。防"参数已在 step 中被 NaN 污染、却以 step 前
    # 算出的有限 loss 通过检查"——blind/prior 事故的保存侧漏洞;且全 NaN 张量字节序相同,
    # 会产出与 init 无关的字节全同 best.pt(评估表全同即由此而来)
    _ok = bool(torch.isfinite(loss).item()) and all(bool(torch.isfinite(pp).all().item()) for pp in model.parameters())
    if _ok and loss.item() < best:
        best = loss.item()
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
    if ep % 20 == 0 or ep == 1:
        log(f"ep{ep:4d} loss {loss.item():.4f} | data {l_data.item():.4f} "
            f"{det} | sp {l_sp.item():.4f} dir {l_dir.item():.4f} "
            f"| {(time.time()-t0)/ep:.2f}s/ep")

# ===================== 7. 评估与导出 =====================
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt")))
with torch.no_grad():
    trajs = [rollout(0), rollout(1)]

    # 全层（含留出层）掩码MSE报告
    log("\n--- 各层拟合/预测（掩码MSE，越小越好；留出层=独立测试） ---")
    for key in ["G", "P", "S", "M"]:
        X, O, t_idx = obs[key]["X"], obs[key]["O"], obs[key]["t_idx"]
        per = []
        for cond in range(2):
            sl = dict(G=trajs[cond][:, :n_g], P=trajs[cond][:, n_g:2 * n_g],
                      S=trajs[cond][:, 2 * n_g:2 * n_g + n_g],
                      M=trajs[cond][:, 2 * n_g + n_g:])[key]
            pred = sl[t_idx]
            diff2 = (pred.t() - X[:, :, cond]) ** 2
            m = O[:, :, cond]
            per.append(((diff2 * m).sum() / m.sum().clamp(min=1.0)).item())
        tag = "训练" if key in train_layers else "【留出测试】"
        log(f"  层{key} ({tag}): Insulin {per[0]:.4f} / PBS {per[1]:.4f}")

    # 导出学到的边（非零权重）—— 每类一张csv
    import csv
    def export_edges(W, mask, rows, cols, fname, layer_name):
        Wd = (W * mask).detach().cpu().numpy()
        rr, cc = np.nonzero(np.abs(Wd) > 1e-6)
        with open(os.path.join(OUT_DIR, fname), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["from", "to", "weight", "layer"])
            for i, j in zip(rr, cc):
                w.writerow([rows[i], cols[j], f"{Wd[i, j]:.5f}", layer_name])
        log(f"  边表 {fname}: {len(rr)} 条非零边")

    sub_g_ids = [gene_ids[i] for i in gidx_np]
    sub_m_ids = [metab_ids[i] for i in midx_np]
    if model.sparse:
        # 稀疏边表直接遍历候选边导出（无需稠密化，14076² 稠密化本身就要792MB）
        def export_sp(sp, rows, cols, fname, layer_name):
            v = sp.val.detach().cpu().numpy()
            rr = sp.rows.cpu().numpy(); cc = sp.cols.cpu().numpy()
            n = 0
            with open(os.path.join(OUT_DIR, fname), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["from", "to", "weight", "layer"])
                for i, j, wv in zip(rr, cc, v):
                    if abs(wv) > 1e-6:
                        w.writerow([rows[i], cols[j], f"{wv:.5f}", layer_name])
                        n += 1
            log(f"  边表 {fname}: {n} 条非零边")
        export_sp(model.sGE_sp, sub_g_ids, sub_g_ids, "edges_GE_基因调控.csv", "GErel")
        export_sp(model.sSG_sp, sub_g_ids, sub_g_ids, "edges_SG_信号转录耦合.csv", "SG")
        export_sp(model.sEC_sp, sub_g_ids, sub_m_ids, "edges_EC_酶促.csv", "ECrel")
        export_sp(model.sMM_sp, sub_m_ids, sub_m_ids, "edges_MM_代谢转化.csv", "MM")
        export_sp(model.sFB_sp, sub_m_ids, sub_g_ids, "edges_FB_代谢反馈.csv", "feedback")
        export_sp(model.sPS_sp, sub_g_ids, sub_g_ids, "edges_PS_信号级联.csv", "PS")
    else:
        export_edges(model.W_GE, sGE, sub_g_ids, sub_g_ids, "edges_GE_基因调控.csv", "GErel")
        export_edges(model.W_SG, sGE, sub_g_ids, sub_g_ids, "edges_SG_信号转录耦合.csv", "SG")
        export_edges(model.W_EC, sEC, sub_g_ids, sub_m_ids, "edges_EC_酶促.csv", "ECrel")
        export_edges(model.W_MM, sMM, sub_m_ids, sub_m_ids, "edges_MM_代谢转化.csv", "MM")
        export_edges(model.W_FB, sFB, sub_m_ids, sub_g_ids, "edges_FB_代谢反馈.csv", "feedback")
        export_edges(model.W_PS, sPS, sub_g_ids, sub_g_ids, "edges_PS_信号级联.csv", "PS")

    # 方案2专项：补偿路径权重报告
    if ARGS.scheme == 2 and not model.sparse:
        s2m = symbol_to_mmu()
        log("\n--- 删边补偿检验（对照方案1同位置权重） ---")
        for tf, tg in [("Mlxipl", "Fasn"), ("Mlxipl", "Scd1")]:
            a, b = s2m.get(tf), s2m.get(tg)
            if a in gpos and b in gpos:
                ia = (gi == gpos[a]).nonzero().flatten()
                ib = (gi == gpos[b]).nonzero().flatten()
                if len(ia) and len(ib):
                    w = (model.W_GE * sGE)[ia[0], ib[0]].item()
                    log(f"  {tf}→{tg}: 学习权重 = {w:.5f}"
                        f"（非零即提示替代路径被启用，与方案1对比可定量）")

    # 保存预测轨迹与观测，供画图/复核
    np.savez_compressed(
        os.path.join(OUT_DIR, "pred_vs_obs.npz"),
        traj_insulin=trajs[0].cpu().numpy(), traj_pbs=trajs[1].cpu().numpy(),
        dt=dt, tmax=tmax,
        gene_ids=np.array(sub_g_ids), metab_ids=np.array(sub_m_ids),
        **{f"obs_{k}": obs[k]["X"].cpu().numpy() for k in obs},
        **{f"omask_{k}": obs[k]["O"].cpu().numpy() for k in obs},
        **{f"tidx_{k}": obs[k]["t_idx"].cpu().numpy() for k in obs},
    )

log(f"\n完成，耗时 {(time.time()-t0)/60:.1f} min，输出目录: {OUT_DIR}")