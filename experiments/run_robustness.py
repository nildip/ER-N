"""
Robustness checks across alpha distributions (corrected)
"""

import json
import os
from experiments.run_movielens import run_full_experiment

def run_robustness(variant='uniform', seed=0):
    print(f"Running robustness for alpha distribution: {variant}")
    res = run_full_experiment(dataset='movielens', n_seeds=3, T=1000, quick_test=True, seed=seed, alpha_distribution=variant)
    os.makedirs('results', exist_ok=True)
    out = f'results/robustness_{variant}.json'
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print('Saved robustness results to', out)
    return res

if __name__ == '__main__':
    for v in ['uniform', 'beta', 'bimodal', 'adversarial']:
        run_robustness(variant=v)
