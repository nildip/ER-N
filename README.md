# ER-N-RecSys2026

Repository scaffold and implementations for the ER-N robust recommender project.

## Quick start (quick-test)
1. create python venv and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run quick experiment (synthetic small dataset):
```bash
python experiments/run_movielens.py --quick-test --T 500 --n_seeds 2
```

3. Run unit tests:
```bash
pytest -q
```

## To run full MovieLens-1M experiment
1. Download dataset:
```bash
bash data/download_datasets.sh
```

2. Preprocess and run:
```bash
python experiments/run_movielens.py --T 10000 --n_seeds 5
```

## Project structure
- `data/` - dataset download & preprocessing
- `src/` - core algorithm and baselines
- `experiments/` - experiment runners
- `analysis/` - figure/table generation
- `tests/` - unit tests
- `results/` - output JSONs

## Notes
- Code is intentionally simple for clarity and testability.
- For large-scale runs, adapt memory usage and consider batching and vectorized updates.
