"""Ablation study runner"""
import json
import os
from src.ern import ERNLearner
from src.baselines import BaselineSoftmaxModel
from src.strategic_users import generate_user_utilities, generate_alpha_distribution, StrategicUser
from experiments.run_movielens import run_single_experiment

def run_ablation_quick(seed=0, T=2000):
    n_users = 50
    n_items = 20
    utilities = generate_user_utilities(n_users, n_items, seed=seed)
    alphas = generate_alpha_distribution(n_users, distribution='uniform', seed=seed)
    users = [StrategicUser(true_utilities=utilities[u], alpha=float(alphas[u])) for u in range(n_users)]
    
    configs = {
        'BSM': lambda: BaselineSoftmaxModel(n_items=n_items, beta=0.0, eta0=0.001, seed=seed),
        'NoiseOnly': lambda: ERNLearner(n_items=n_items, beta=0.0, sigma=0.05, eta0=0.001, seed=seed),
        'EntropyOnly': lambda: ERNLearner(n_items=n_items, beta=0.1, sigma=0.0, eta0=0.001, seed=seed),
        'ERNFull': lambda: ERNLearner(n_items=n_items, beta=0.1, sigma=0.01, eta0=0.001, seed=seed)
    }
    
    results = {}
    for name, constructor in configs.items():
        print(f"Running {name}...")
        runs = []
        for s in range(2):
            run_res = run_single_experiment(constructor, users, T=T, seed=seed + s)
            runs.append(run_res)
        results[name] = runs
    
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_results_quick.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved to results/ablation_results_quick.json')
    return results

if __name__ == '__main__':
    run_ablation_quick()
