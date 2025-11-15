"""
Generate figures (matplotlib) from results JSON files.
Produces PDF files at 300 DPI.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_learning_curves(results_dict, outpath='figures/learning_curves.pdf'):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(8, 6))
    for method, seeds in results_dict.items():
        cum_regs = []
        for seed_res in seeds:
            if 'strategic' in seed_res:
                true_rewards = seed_res['strategic']['true_rewards']
            else:
                true_rewards = seed_res['true_rewards']
            cum = np.cumsum([1.0 - r for r in true_rewards])
            cum_regs.append(cum)
        min_len = min(map(len, cum_regs))
        aligned = np.vstack([c[:min_len] for c in cum_regs])
        mean = np.mean(aligned, axis=0)
        std = np.std(aligned, axis=0)
        x = np.arange(1, min_len + 1)
        plt.plot(x, mean, label=method)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel('Round')
    plt.ylabel('Cumulative regret (proxy)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    path = 'results/movielens_experiment_quick_T1000_seeds3.json'
    if not os.path.exists(path):
        files = [f for f in os.listdir('results') if f.endswith('.json')]
        if not files:
            print('No results found in results/. Run experiments first.')
            return
        path = os.path.join('results', files[0])
    results = load_results(path)
    plot_learning_curves(results, outpath='figures/learning_curves.pdf')
    print('Saved figures to figures/')

if __name__ == '__main__':
    main()
