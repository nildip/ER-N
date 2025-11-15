"""Evaluation metrics"""
import numpy as np
from scipy import stats

def compute_manipulation_gain(strategic_rewards, truthful_rewards):
    strategic_rewards = np.asarray(strategic_rewards, dtype=np.float64)
    truthful_rewards = np.asarray(truthful_rewards, dtype=np.float64)
    return float(np.mean(strategic_rewards - truthful_rewards))

def compute_cumulative_regret(true_utilities, actions, optimal_action):
    true_utilities = np.asarray(true_utilities, dtype=np.float64)
    regrets = [true_utilities[optimal_action] - true_utilities[a] for a in actions]
    cum = np.cumsum(regrets)
    return cum

def compute_ndcg_at_k(recommended_items, relevant_items, k=10):
    recommended = list(recommended_items)[:k]
    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2.0)
    n_rel = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(n_rel))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)

def paired_ttest(method_a, method_b, alpha=0.05):
    res = stats.ttest_rel(method_a, method_b)
    t_stat, p_val = float(res.statistic), float(res.pvalue)
    return {'t_statistic': t_stat, 'p_value': p_val, 'significant': p_val < alpha}
