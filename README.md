# 虚拟代谢论文：评估与审计代码

本包只公开论文结果所用的评估、审计代码，以及特征级 z 统计文件 `data_sim2/zstats.npz`。

不包含模型训练、掩码生成、真值生成、训练权重或绘图代码。它不是完整训练项目，也不能单独一键复现论文全部结果。

## 1. 环境

建议 Python 3.10 或更新版本。安装依赖：

```bash
python -m pip install -r requirements.txt
```

本次复核使用 Python 3.12.12、NumPy 2.2.5、pandas 2.3.2、SciPy 1.16.2、PyTorch 2.8.0。

## 2. 已公开的数据

`data_sim2/zstats.npz` 包含 8 个数组：

- `zmu_G`、`zsd_G`：691 个基因特征；
- `zmu_P`、`zsd_P`：691 个蛋白特征；
- `zmu_S`、`zsd_S`：691 个信号特征；
- `zmu_M`、`zsd_M`：33 个代谢物特征。

这些数组是特征级均值和标准差，不含训练权重、候选掩码或真值网络。

## 3. 可以直接运行的审计

在本目录运行：

```bash
python 跑饱和审计_v3_1024.py
python 审计_饱和死列_v2.py
```

两份脚本只读取已公开的 `data_sim2/zstats.npz`。第二份脚本在未提供 `train_tensors.npz` 时，会明确跳过轨迹级统计，不会假装已经计算。

本次实际运行结果：

- P 层共 691 个特征，其中有观测 162 个、无观测 529 个；
- 有观测特征中，`|zmu_P| > 3` 为 127 个，`|zmu_P| >= 19` 为 35 个；
- 严判计数 P35／G7／S7／M2，与已核验计数一致。

## 4. 需要另外提供输入的脚本

下列输入不随包公开：`ground_truth.npz`、`best.pt`、`train_tensors.npz`、`pred_vs_obs.npz`、候选掩码、KO 轨迹、内部结果表和各臂训练日志。

下列命令展示参数格式。路径需换成本地输入路径；没有这些输入时，相应脚本不能独立运行。

### 4.1 轨迹拟合 R²

```bash
python diag_r2_fig4_v1_20261005.py /path/to/pred_vs_obs.npz
```

可依次传入多个 `pred_vs_obs.npz`。按层、按条件计算观测掩码内的 R²。

### 4.2 六类边的恢复评估

```bash
python 仿真评估_边恢复率_tie_v2.py --data_dir /path/to/data --sim_dir /path/to/data_sim2 --tag _sim2or01
```

读取训练检查点和真值输入，使用并列平均秩计算 AUROC。检查点和真值输入不公开。

### 4.3 GE 362 候选域评分

```bash
python 边评估四件套_1005.py --gt /path/to/ground_truth.npz /path/to/edges.csv
```

边评分 CSV 必须含 `from`、`to`、`weight` 三列。

### 4.4 dyn45 的 15 次结果评估

```bash
python dyn45对死评估_v1_20261006.py --gt /path/to/ground_truth.npz --mask /path/to/mask_GErel_edges.csv --input-csv /path/to/input.csv --vim-dir /path/to/vim_results
```

`--vim-dir` 中还需有 `dyn45_manifest_1006.json` 和清单所列的 VIM 文件。只读取受信任的 pickle 文件。

### 4.5 静态 LASSO 基线

```bash
python 基线对比_静态推断LASSO_服务器版.py --sim_dir /path/to/data_sim2
```

该目录需含 `train_tensors.npz` 和 `ground_truth.npz`。脚本只实现基线推断与评分，不包含主模型训练代码。

### 4.6 KO 直连主导检验

```bash
python 仿真KO_直连主导检验v1.py --gt /path/to/ground_truth.npz --traj /path/to/仿真KO_轨迹.npz --pool /path/to/仿真KO_v6_多KO池化.csv
```

### 4.7 开发批与独立批稳健性

```bash
python 独立验证稳健性_1024.py --dev /path/to/detail_1024.csv --indep /path/to/detail_indep_1024.csv --out result.txt
```

### 4.8 动态范围归一指标

```bash
python 补算动态范围归一_1024.py --input /path/to/detail_1024.csv --out result.txt
```

### 4.9 方向准确率与二项检验

```bash
python 补算方向统计_1024.py --input /path/to/detail_1024.csv --out result.txt
```

### 4.10 Q／D／N／X 幅度校准

```bash
python 跑Q组幅度校准_1024.py --pool /path/to/仿真KO_v6_多KO池化.csv --detail /path/to/仿真KO_定量验证_逐物种.csv --out result.txt
```

### 4.11 轨迹级饱和审计

```bash
python 审计_饱和死列_v2.py --zstats data_sim2/zstats.npz --train-tensors /path/to/train_tensors.npz
```

## 5. 复核范围

- 12 份 Python 文件均通过语法解析和运行测试；
- 两份只依赖公开 `zstats.npz` 的审计脚本已用公开数据实际运行，结果见第 3 节；
- 其余 10 份脚本已用临时合成输入做运行测试，只检查输入、计算和输出能否走通；合成输入及其输出不属于论文结果，也未放入本包；
- 文件大小和 MD5 见 `文件清单.csv`。清单列出除清单自身外的全部文件。

读取 NPZ、检查点或 pickle 输入时，只使用来源可信的文件。
