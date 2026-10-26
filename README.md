# MonoBracket

MonoBracket: Semi-supervised tool wear monitoring using sparse cutting pass inspections as physical monotone brackets.

## Python Codes

| File | Description |
|---|---|
| `monopath.py` | Physical monotone bracketing bounds ($\ell_i, h_i, \underline{y}_i, \overline{y}_i$), certification of unlabelled windows, Viterbi MAP path decoding, and forward-backward marginals. |
| `methods.py` | Training routines: proposed MonoBracket variants (`mbpl_nopath`, `mbpl`, `mbpl_cert`, `mbpl_noanch`) and baseline methods (supervised, pseudo-labeling, FixMatch, Mean Teacher, SimCLR, TCL). |
| `model.py` | Dilated Temporal Convolutional Network (TCN) backbone, temporal projection head, classifier head, and heteroscedastic Gaussian regression head with Monte Carlo dropout. |
| `losses.py` | Loss functions: temporal contrastive loss (NT-Xent with Gaussian temporal kernel), class-balanced focal loss, and heteroscedastic Gaussian negative log-likelihood. |
| `uncertainty.py` | Uncertainty estimation: BALD predictive mutual information from MC-dropout, aleatoric/epistemic variance decomposition, weighted PAVA isotonic regression, and adaptive per-class quantiles. |
| `tta.py` | Online test-time adaptation: initial wear offset shrinkage, monotone isotonic wear smoothing, and fused stage readout without test labels. |
| `splits.py` | Leave-One-Tool-Out (LOTO) cross-validation splitting and pass-level inspection sampling under a budget cap. |
| `augment.py` | Weak and strong augmentations for time-series sensor windows. |
| `data_nasa.py` | NASA Ames Milling dataset loading, segmentation into windows, and preprocessing. |
| `data_phm.py` | PHM 2010 Milling Challenge dataset loading, segmentation, and preprocessing. |
| `data_nuaa.py` | NUAA Ideahouse Milling dataset loading, segmentation, and preprocessing. |
| `pretrain.py` | Contrastive pretraining and cached encoder checkpoint loading. |
| `run_experiments.py` | Experiment execution driver across benchmarks, splits, and methods (resumable per cell). |
| `run_cell.py` | Utility to execute a single benchmark/method/seed cell. |
| `readout_v5.py` | Evaluation of online tool-life readout on held-out tools. |
| `analyses_v5.py` | Statistical tests (Wilcoxon signed-rank, Friedman), metric aggregations, and facts extraction. |
| `make_tables_v5.py` | Automated LaTeX table generation from experiment facts. |
| `export_panels.py` | Exporting trajectory and curve data for figure visualization. |
| `test_losses.py` | Unit test suite for contrastive, focal, and regression loss functions. |
| `audit_code.py` | Code audit suite verifying data leakage prevention, budget caps, and label-free online inference. |
| `sensitivity.py` | Hyperparameter sensitivity sweeps across quantiles, thresholds, and MC passes. |
| `aggregate.py` | Utilities for aggregating metric rows across experimental runs. |
| `analyses.py` | Statistical analyses and ranking across methods. |
| `figures.py` | Figure plotting routines. |
| `figures3.py` | Multi-benchmark comparative figure generation. |
| `make_tables.py` | Benchmark table generation utility. |
| `make_tables2.py` | Additional table generation utility. |
| `make_tables3.py` | Extended multi-benchmark table generation utility. |
| `verify_paper.py` | Automated audit verifying quantitative claims against raw experiment results. |
| `verify_paper3.py` | Extended multi-benchmark manuscript verification audit. |

## How to Simulate

### 1. Requirements

Install the dependencies:
```bash
pip install torch numpy scipy scikit-learn
```

### 2. Verify Correctness

Run the unit check suites to verify the mathematics, monotone bracketing, and loss implementations:
```bash
python monopath.py
python test_losses.py
python -c "import uncertainty, tta; uncertainty.verify_math(); tta.verify()"
```

### 3. Run Experiments (Simulation)

Run the benchmark simulation suites:

- **Main proposed method and ablation variants** (120 runs at 8% inspection budget):
  ```bash
  python run_experiments.py --suite v5
  ```

- **Low-budget label sweep** (30 runs at 2% inspection budget):
  ```bash
  python run_experiments.py --suite v5sweep
  ```

- **Baselines comparison suite** (Random Forest, SVM, Supervised, Pseudo-label, FixMatch, Mean Teacher, SimCLR, TCL):
  ```bash
  python run_experiments.py --suite main
  ```

*Note*: If raw dataset files (`mill.mat`, `phm2010_bundle.csv`, `nuaa_bundle.csv`) are located in a custom directory, set the `UCTCL_ASSETS` environment variable:
```bash
export UCTCL_ASSETS=/path/to/dataset_folder
```

### 4. Evaluate and Extract Results

After simulation runs complete:

1. **Evaluate online readouts on held-out tools**:
   ```bash
   python readout_v5.py
   ```

2. **Compute statistical metrics and facts**:
   ```bash
   python analyses_v5.py
   ```

3. **Generate result tables**:
   ```bash
   python make_tables_v5.py
   ```

4. **Export data panels for trajectory plotting**:
   ```bash
   python export_panels.py
   ```
