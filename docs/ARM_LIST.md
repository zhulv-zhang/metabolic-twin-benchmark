# ARM_LIST — paper arms, commands and seeds

Environments (see docs/pip_freeze_venv.txt and docs/pip_freeze_dyn.txt):
- `venv`: every step except real-data tensor preparation.
- `dyn`: real-data tensor preparation only (README step 2; adds openpyxl).

Hash seeds: main chain PYTHONHASHSEED=0; the +590-edge supplement step (build_masks
step 1) uses 621; the direction replayer (code/utils/direction_finalizer.py) is seed-free.
Training outputs land in data/scheme<scheme><tag>/ — one directory per arm, no overwrites.

Data. Build masks (README step 1), then generate the universe:
  python code/generate/semisynth_simulation.py              # seed 42, p 0.3/0.1/0.01, noise 0.1, reps 3 -> data_sim2/
  python code/generate/semisynth_simulation.py --ideal      # ideal-data mode -> data_sim2i/

Cold starts (paper 2.4; 300 epochs, lr 1e-3):
  # zero init (paper arm; ideal-data universe):
  PYTHONHASHSEED=0  python code/train/train_twin.py --scheme 1 --tensor_file data_sim2i/train_tensors.npz --tag _sim2izero
  # determinism re-runs (bit-identical, counted once; paper 2.4):
  PYTHONHASHSEED=43 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2i/train_tensors.npz --tag _sim2izero_s43
  PYTHONHASHSEED=44 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2i/train_tensors.npz --tag _sim2izero_s44
  # blind arms: two independent runs, oracle-blind W~N(0,0.05) in-mask, fan-in norm (mag 0.05: >=0.3
  # fails to train, paper 4.5):
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --oracle_blind --blind_init_mag 0.05 --init_fanin --tag _sim2blind3
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --oracle_blind --blind_init_mag 0.05 --init_fanin --tag _sim2blind4
  # prior-signed arms: two independent runs, magnitude 0.05, GE/EC sign prior (run logs:
  # "先验初始化区: 幅度 0.05"):
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --init_prior --init_prior_mag 0.05 --init_fanin --tag _sim2prior3
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --init_prior --init_prior_mag 0.05 --init_fanin --tag _sim2prior4

Oracle basin (paper 2.4; ideal-data universe; Fig. 3 plots sigma 0, 0.1, 0.4):
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --oracle_init data_sim2/ground_truth.npz --oracle_noise 0   --tag _sim2oracle
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --oracle_init data_sim2/ground_truth.npz --oracle_noise 0.1 --tag _sim2or01
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --oracle_init data_sim2/ground_truth.npz --oracle_noise 0.2 --tag _sim2or02
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --oracle_init data_sim2/ground_truth.npz --oracle_noise 0.3 --tag _sim2or03
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --oracle_init data_sim2/ground_truth.npz --oracle_noise 0.4 --tag _sim2or04
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2i/train_tensors.npz --oracle_init data_sim2i/ground_truth.npz --oracle_noise 0 --tag _sim2ior00   # ideal-data upper-bound arm

Held-out-layer probes (paper 2.5; scheme 3 trains G+M only, P/S held out; freeze/pin
blocks are set in each script's header; per-arm output dirs data/scheme3_sim2s3*/):
  PYTHONHASHSEED=0 python code/train/train_twin_zf_1005.py                    --tensor_file data_sim2/train_tensors.npz --tag _sim2s3zf
  PYTHONHASHSEED=0 python code/train/train_v5_z_corrected_freeze_probe.py     --tensor_file data_sim2/train_tensors.npz --tag _sim2s3zc
  PYTHONHASHSEED=0 python code/train/train_v5_x0_freeze_probe.py              --tensor_file data_sim2/train_tensors.npz --tag _sim2s3fz
  PYTHONHASHSEED=0 python code/train/train_v5_P_pathway_freeze_probe.py       --tensor_file data_sim2/train_tensors.npz --tag _sim2s3
  # zf: freeze feedback+phospho-signalling blocks; zc: z-conjugate initialization;
  # fz: x0 probe; s3: P-pathway pinning; x0b dual variant in train_v5_x0b_dual_freeze_probe.py

Kinetic families (paper 2.8-2.9):
  # Hill (seeds 42/43/44; universes data_sim_hill, data_sim_hill_s43, data_sim_hill_s44;
  # hill_wrapper_v1.0.py orchestrates; training tags _hill_tt_basin / _hill_tt_blind / _hill_tt_prior):
  python code/generate/semisynth_hill_generate.py --seed 42
  python code/generate/semisynth_hill_generate.py --seed 43
  python code/generate/semisynth_hill_generate.py --seed 44
  # Delayed (run accounts, accounts.txt; noise tiers 0.1/0.2/0.4/0.8 — a different grid from
  # the main-chain basin {0,0.1,0.2,0.3,0.4}, paper 2.9):
  python code/generate/delay_wrapper_v1.2_1005.py --dt 0.5    # base; arms _blind05 (--oracle_blind --init_fanin, mag 0.05),
                                                              # _prior05 (--init_prior --init_prior_mag 0.05 --init_fanin),
                                                              # _noise01/02/04/08 (--oracle_noise 0.1/0.2/0.4/0.8)
  python code/generate/delay_wrapper_v1.2_1005.py --dt 0.25   # _d025
  python code/generate/delay_wrapper_v1.2_1005.py --dt 0.125  # _d0125

Regularization-weight scan (paper 2.3; from-scratch, realistic universe; run logs in
data/scheme1_sim2sp2/ and sp3/):
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --w_sparse 1e-3 --tag _sim2sp2
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file data_sim2/train_tensors.npz --w_sparse 1e-2 --tag _sim2sp3

Real-data fit (paper 2.3; 600 epochs):
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file GSE166336_小鼠组学数据/train_tensors.npz --w_prox 3e-4 --w_sparse 1e-3 --tag _prox3e4
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file GSE166336_小鼠组学数据/train_tensors.npz --w_prox 1e-3 --tag _prox1e3
  PYTHONHASHSEED=0 python code/train/train_twin.py --scheme 1 --tensor_file GSE166336_小鼠组学数据/train_tensors.npz --w_prox 0 --tag _prox0

Evaluation and audit (paper 2.2, 2.6-2.7; outputs scored against ground_truth.npz):
  python code/evaluate/edge_recovery_eval.py            # edge AUROC/AUPRC/Top-k
  python code/evaluate/static_LASSO_baseline.py         # FD+LASSO static baseline
  python code/evaluate/ko_validation_v9_checkup.py      # 12-source knockout battery (Q/D/N/X triage)
  python code/evaluate/independent_validation.py        # independent-source batch
  python code/evaluate/saturation_audit.py              # tanh-saturation audit (Table 1)
  python code/evaluate/triage_calibration_v1.py         # per-prediction grade calibration

dynGENIE3 comparison (paper 4.7): run dynGENIE3 externally (2 observation modes x 3 tree
settings x 5 seeds); aggregate run-level scores with the scripts listed in MANIFEST (VIM
files are method outputs, not shipped; per-run seeds per run records).