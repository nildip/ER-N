"""
Statistical tests between ER-N and baselines
"""
import json
import os
import numpy as np
from src.metrics import paired_ttest


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_metric(results, method, metric='true_rewards'):
    seeds = results.get(method, [])
    vals = [np.sum(seed[metric]) for seed in seeds]
    return np.array(vals)


def run_tests(path):
    results = load_results(path)
    ern_vals = extract_metric(results, 'ern')
    report = {}
    for method in results.keys():
        if method == 'ern':
            continue
        other = extract_metric(results, method)
        if len(ern_vals) != len(other):
            minlen = min(len(ern_vals), len(other))
            a = ern_vals[:minlen]
            b = other[:minlen]
        else:
            a = ern_vals
            b = other
        res = paired_ttest(a, b)
        report[method] = res
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    files = [f for f in os.listdir('results') if f.endswith('.json')]
    if not files:
        print('No results in results/. Run experiments first')
    else:
        run_tests(os.path.join('results', files[0]))
