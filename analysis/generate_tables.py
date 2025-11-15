"""
Generate LaTeX tables summarizing results.
"""
import json
import os
import numpy as np


def summarize_main_results(results):
    rows = []
    for method, seeds in results.items():
        gains = []
        regrets = []
        ndcgs = []
        for seed in seeds:
            reported = seed['reported_rewards']
            true = seed['true_rewards']
            gains.append(sum(reported) - sum(true))
            regrets.append(sum(1.0 - np.array(true)))
            ndcgs.append(0.0)
        rows.append((method, float(np.mean(gains)), float(np.mean(regrets)), float(np.mean(ndcgs))))
    return rows


def format_latex_table(rows, outpath='results/main_results_table.tex'):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    header = r"""\\begin{tabular}{lrrr}
\\hline
Method & ManipGain & Regret & NDCG@10 \\\
\\hline
"""
    footer = "\\\\hline\\n\\\\end{tabular}\\n"
    with open(outpath, 'w') as f:
        f.write(header)
        for r in rows:
            f.write(f"{r[0]} & {r[1]:.4f} & {r[2]:.4f} & {r[3]:.4f} \\\
")
        f.write(footer)
    print('Wrote LaTeX table to', outpath)

if __name__ == '__main__':
    files = [f for f in os.listdir('results') if f.endswith('.json')]
    if not files:
        print('No results found in results/. Run experiments first.')
    else:
        path = os.path.join('results', files[0])
        with open(path) as f:
            results = json.load(f)
        rows = summarize_main_results(results)
        format_latex_table(rows)
