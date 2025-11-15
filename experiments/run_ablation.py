"""
Ablation study runner (corrected)

Uses the ER-N learner with the 4 configurations:
1. BSM: beta=0, sigma=0
2. Noise-Only: beta=0, sigma=0.05
3. Entropy-Only: beta=0.1, sigma=0
4. ER-N-Full: beta=0.1, sigma=0.01
"""

import json
import os
from src.ern import ERNLearner
from src.strategic_users import generate_user_utilities, generate_alpha_distribution, StrategicUser

def run_ablation_quick(seed=0, T=2000):
    configs = {
        'BSM': {'beta': 0.0, 'sigma': 0.0},
        'NoiseOnly': {'beta': 0.0, 'sigma': 0.05},
        'EntropyOnly': {'beta': 0.1, 'sigma': 0.0},
        'ERNFull': {'beta': 0.1, 'sigma': 0.01}
    }

    n_users = 50
    n_items = 20
    utilities = generate_user_utilities(n_users, n_items, seed=seed)
    alphas = generate_alpha_distribution(n_users, distribution='uniform', seed=seed)
    users = [StrategicUser(true_utilities=utilities[u], alpha=float(alphas[u])) for u in range(n_users)]

    results = {}

    from experiments.run_movielens import run_single_experiment

    for name, cfg in configs.items():
        print(f"Running ablation config: {name} (beta={cfg['beta']}, sigma={cfg['sigma']})")
        constructor = lambda: ERNLearner(n_items=n_items, beta=cfg['beta'], sigma=cfg['sigma'], eta0=0.001, seed=seed)
        runs = []
        for s in range(2):
            run_res = run_single_experiment(constructor, users, T=T, seed=seed + s)
            runs.append(run_res)
        results[name] = runs

    os.makedirs('results', exist_ok=True)
    out = 'results/ablation_results_quick.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved ablation results to', out)
    return results

if __name__ == '__main__':
    run_ablation_quick()
