# ER-N: Robust Recommender Systems Against Coordinated Manipulation

Repository for the ER-N paper. This code demonstrates how ER-N defends against coordinated manipulation attacks where colluding users strategically inflate a target item's ratings.

## 📋 Prerequisites

- Python 3.7+
- ~500MB disk space (for MovieLens-1M dataset)

## 🚀 Quick Start (5 minutes)

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/nildip/ER-N.git
cd ER-N

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Quick Test (Synthetic Data)
```bash
# Fast test: 1000 users, 200 items, 1000 rounds, 2 seeds
python experiments/run_experiments.py --experiment main --T 1000 --n_seeds 2
```

**Expected output:**
```
Using synthetic: 1000 users, 200 items

Experiment: main_collusion1pct
  Collusion: 1% (10/1000 users)
  ...

ER-N REDUCES MANIPULATION BY: 75.3%
```

**Results saved to:** `results/main_collusion1pct_synthetic_T1000_seeds2.json`

### 3. Verify Installation
```bash
pytest -q
```

All tests should pass.

---

## 📊 Reproduce Paper Results

### Step 1: Download MovieLens-1M Dataset
```bash
bash data/download_datasets.sh
```

This downloads and extracts MovieLens-1M (~6MB) to `data/movielens-1m/`.

### Step 2: Run Main Experiments

#### **Experiment 1: Main Results (Table 1 in paper)**
Tests ER-N vs BSM at different collusion rates.
```bash
python experiments/run_experiments.py --experiment main --real-data --K 200 --T 10000 --n_seeds 5
```

**What it does:**
- Uses top-200 most popular MovieLens items
- Tests collusion rates: 1%, 5%, 10%, 20%
- 10,000 rounds per seed
- 5 random seeds for statistical validity

**Runtime:** ~20 minutes

**Output files:**
- `results/main_collusion1pct_movielens_T10000_seeds5.json`
- `results/main_collusion5pct_movielens_T10000_seeds5.json`
- `results/main_collusion10pct_movielens_T10000_seeds5.json`
- `results/main_collusion20pct_movielens_T10000_seeds5.json`

---

#### **Experiment 2: Ablation Study - σ (Figure 2 in paper)**
Tests how robustness noise parameter affects manipulation resistance.
```bash
python experiments/run_experiments.py --experiment ablation-sigma --real-data --K 200 --T 10000 --n_seeds 5
```

**What it does:**
- Varies σ: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0
- σ=0.0 is equivalent to baseline (no noise)
- Fixed collusion rate: 5%

**Runtime:** ~30 minutes

**Output files:**
- `results/ablation_sigma0.0_movielens_T10000_seeds5.json`
- `results/ablation_sigma0.1_movielens_T10000_seeds5.json`
- ... (7 files total)

---

#### **Experiment 3: Ablation Study - β (Figure 3 in paper)**
Tests exploration parameter sensitivity.
```bash
python experiments/run_experiments.py --experiment ablation-beta --real-data --K 200 --T 10000 --n_seeds 5
```

**What it does:**
- Varies β: 1.0, 5.0, 10.0, 20.0, 50.0
- Tests exploration vs exploitation trade-off

**Runtime:** ~25 minutes

**Output files:**
- `results/ablation_beta1.0_movielens_T10000_seeds5.json`
- ... (5 files total)

---

#### **Experiment 4: Attack Strength Analysis (Figure 4 in paper)**
Tests robustness against varying attack coordination.
```bash
python experiments/run_experiments.py --experiment ablation-alpha --real-data --K 200 --T 10000 --n_seeds 5
```

**What it does:**
- Varies α (attack strength): 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- α=0.0: no manipulation
- α=1.0: maximum manipulation

**Runtime:** ~30 minutes

---

### Step 3: Run ALL Experiments
```bash
# WARNING: Takes ~2 hours
python experiments/run_experiments.py --experiment all --real-data --K 200 --T 10000 --n_seeds 5
```

Runs all main + ablation experiments sequentially.

---

### Step 4: Generate LaTeX Tables

After running experiments, generate tables for the paper:
```bash
python analysis/generate_tables.py
```

**Output:**
```
==================================================================
GENERATING LATEX TABLES
==================================================================

Generating Table 1: Main Results...
✓ Saved to: results/table1_main_results.tex

Generating Table 2: Sigma Ablation...
✓ Saved to: results/table2_ablation_sigma.tex

Generating Table 3: Beta Ablation...
✓ Saved to: results/table3_ablation_beta.tex
```

**Generated tables:**
- `results/table1_main_results.tex` - Main results (Table 1 in paper)
- `results/table2_ablation_sigma.tex` - Sigma ablation (Table 2 in paper)
- `results/table3_ablation_beta.tex` - Beta ablation (Table 3 in paper)

Copy these `.tex` files directly into your LaTeX paper source.

**Requirements:**
- Must run experiments first (tables need the JSON result files)

---

### Step 5: Statistical Significance Tests

Verify that ER-N's improvements are statistically significant:
```bash
python analysis/statistical_tests.py
```

**Output:**
```
==================================================================
STATISTICAL TESTS: Main Results
==================================================================

Collusion Rate: 1%
  ER-N boost: 1.12 ± 0.23
  BSM boost:  5.23 ± 0.45
  Mean difference: -4.11
  t-statistic: -15.342
  p-value: 0.0001
  Significant (p<0.05): ✓ YES

Collusion Rate: 5%
  ER-N boost: 3.34 ± 0.67
  BSM boost:  12.45 ± 1.02
  Mean difference: -9.11
  t-statistic: -12.567
  p-value: 0.0002
  Significant (p<0.05): ✓ YES
...
```

**Tests performed:**
- **Main results**: ER-N vs BSM at each collusion rate (paired t-test)
- **Sigma ablation**: σ=0.0 vs σ=0.3 (validates noise helps)
- **Beta ablation**: β=1.0 vs β=10.0 (validates exploration helps)

**Interpretation:**
- p < 0.05 means statistically significant
- Negative mean difference = ER-N has lower manipulation (better)

---

## 🔧 Advanced Usage

### Custom Parameters
```bash
# Different collusion rate
python experiments/run_experiments.py --experiment main --collusion 0.15 --T 10000 --n_seeds 5

# Different σ value
python experiments/run_experiments.py --experiment main --sigma 0.4 --T 10000 --n_seeds 5

# Full MovieLens dataset (no top-K filter)
python experiments/run_experiments.py --experiment main --real-data --T 10000 --n_seeds 5

# Synthetic data (fast testing)
python experiments/run_experiments.py --experiment main --T 5000 --n_seeds 3
```

### Parameter Reference

| Parameter | Description | Default | Paper Value |
|-----------|-------------|---------|-------------|
| `--experiment` | Which experiment to run | `main` | varies |
| `--T` | Number of rounds | 10000 | 10000 |
| `--n_seeds` | Random seeds | 5 | 5 |
| `--real-data` | Use MovieLens | False | True |
| `--K` | Top-K items filter | None | 200 |
| `--beta` | Exploration temperature | 10.0 | 10.0 |
| `--sigma` | ER-N noise parameter | 0.3 | 0.3 |
| `--eta0` | Learning rate | 0.2 | 0.2 |
| `--collusion` | Collusion rate | 0.05 | varies |
| `--alpha` | Attack strength | 0.8 | 0.8 |

---

## 📁 Project Structure
```
ER-N/
├── data/
│   ├── download_datasets.sh      # Dataset download script
│   └── preprocess.py             # MovieLens preprocessing
├── src/
│   ├── ern.py                    # ER-N algorithm implementation
│   ├── baselines.py              # Baseline methods (BSM, RobustMF)
│   └── coordinated_manipulation.py  # Attack simulation
├── experiments/
│   └── run_experiments.py        # Main experiment runner
├── analysis/
│   ├── generate_tables.py        # LaTeX table generation
│   └── statistical_tests.py      # Significance testing
├── tests/
│   └── test_*.py                 # Unit tests
├── results/                      # Output JSONs (created automatically)
├── requirements.txt
└── README.md
```

---

## 📈 Understanding Results

Each result JSON contains:
```json
{
  "metadata": {
    "collusion_rate": 0.05,
    "n_users": 6040,
    "n_items": 200,
    "T": 10000,
    "beta": 10.0,
    "sigma": 0.3,
    ...
  },
  "results": {
    "ern": [
      {
        "strategic": {
          "target_recs": 523,
          "target_policy_trace": [0.005, 0.007, ...]
        },
        "truthful": {
          "target_recs": 198,
          ...
        },
        "target_boost": 3.25,  // % increase in recommendations
        ...
      }
    ],
    "bsm": [ ... ]
  }
}
```

**Key Metrics:**
- `target_boost`: How much colluders increased target item recommendations (%)
- Lower = better robustness
- **ER-N should have ~70-85% lower boost than BSM**

---

## 🧪 Experiment Options

| Experiment | Command | Purpose | Runtime |
|------------|---------|---------|---------|
| `main` | `--experiment main` | Vary collusion rate | ~20 min |
| `ablation-sigma` | `--experiment ablation-sigma` | Test σ sensitivity | ~30 min |
| `ablation-beta` | `--experiment ablation-beta` | Test β sensitivity | ~25 min |
| `ablation-collusion` | `--experiment ablation-collusion` | Extended collusion rates | ~30 min |
| `ablation-alpha` | `--experiment ablation-alpha` | Test attack strength | ~30 min |
| `all` | `--experiment all` | Run everything | ~2 hours |

---

## 🐛 Troubleshooting

### Error: "cannot import name 'load_movielens'"

**Solution:** Run dataset download first:
```bash
bash data/download_datasets.sh
```

### Error: "MovieLens not found"

**Solution:** Check that `data/movielens-1m/ratings.dat` exists. Re-run download script.

### Slow performance

**Solution:** Use smaller dataset or fewer rounds:
```bash
python experiments/run_experiments.py --experiment main --K 100 --T 5000 --n_seeds 3
```

### Out of memory

**Solution:** Reduce dataset size:
```bash
python experiments/run_experiments.py --experiment main --K 50 --T 5000
```

### Missing scipy for statistical tests

**Solution:** Make sure scipy is installed:
```bash
pip install scipy
```

---

## 📝 Citation

If you use this code, please cite:
```bibtex
@inproceedings{ern2026,
  title={ER-N: Robust Recommender Systems Against Coordinated Manipulation},
  author={...},
  booktitle={RecSys},
  year={2026}
}
```

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

## 🔬 Technical Notes

- **Code design:** Intentionally simple for clarity and reproducibility
- **Scalability:** For production use, consider batching and vectorization
- **Missing ratings:** Filled with global mean (standard imputation)
- **Reproducibility:** All experiments use fixed random seeds

---

## ⚡ Quick Reference
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash data/download_datasets.sh

# Quick test
python experiments/run_experiments.py --experiment main --T 1000 --n_seeds 2

# Paper results
python experiments/run_experiments.py --experiment all --real-data --K 200 --T 10000 --n_seeds 5

# Generate tables
python analysis/generate_tables.py

# Statistical tests
python analysis/statistical_tests.py

# Single ablation
python experiments/run_experiments.py --experiment ablation-sigma --real-data --K 200 --T 10000 --n_seeds 5
```
