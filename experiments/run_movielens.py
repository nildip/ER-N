"""
Experiment runner for MovieLens / quick-tests (corrected)

Fixes applied:
- Single-signature check for get_policy (inspect.signature)
- Runs each learner twice (strategic vs truthful) and computes manipulation_gain
- Uses isinstance(RobustMF) for update routing
- Pre-samples user sequence for reproducibility and speed
- Tracks per-round cumulative regret (for learning curves)
- Accepts learner constructors so we can create fresh learners for each run
"""

import argparse
import json
import os
import random
import inspect
import numpy as np

from src.ern import ERNLearner
from src.baselines import BaselineSoftmaxModel, RobustMF, SASRec, LightGCN
from src.strategic_users import generate_user_utilities, generate_alpha_distribution, StrategicUser
from src.metrics import compute_cumulative_regret

# Helper: set random seeds for reproducibility
def _set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def _run_once(learner, users, T, seed=0, strategy='greedy'):
    """
    Run a single pass (either strategic or truthful) with the provided learner instance.
    Returns a dict with per-round lists: actions, true_rewards, reported_rewards, cumulative_regret.
    """

    _set_seed(seed)
    n_users = len(users)

    # Pre-sample users for speed & reproducibility
    user_sequence = np.random.randint(0, n_users, size=T)

    # Detect whether get_policy expects user index (inspect once)
    needs_user = False
    if hasattr(learner, 'get_policy'):
        try:
            sig = inspect.signature(learner.get_policy)
            params = sig.parameters
            # check for common names 'user' or 'u_idx' or 'user_id'
            needs_user = any(name in params for name in ('user', 'u_idx', 'user_id'))
        except (ValueError, TypeError):
            needs_user = False

    results = {
        'actions': [],
        'true_rewards': [],
        'reported_rewards': [],
        'cumulative_regret': []
    }

    cumulative_regret = 0.0

    for t in range(T):
        u_idx = int(user_sequence[t])
        user = users[u_idx]

        # Action selection
        if needs_user:
            p = learner.get_policy(u_idx)
        else:
            p = learner.get_policy()
        # Safety: ensure p is a probability vector that sums > 0
        p = np.asarray(p, dtype=float)
        if p.sum() <= 0 or np.any(p < 0):
            # fallback uniform
            p = np.ones_like(p) / len(p)
        else:
            p = p / p.sum()

        action = int(np.random.choice(len(p), p=p))

        # Rewards
        true_reward = float(user.get_true_reward(action))
        reported_reward = float(user.report_feedback(action, strategy=strategy))

        # Update learner
        if isinstance(learner, RobustMF):
            learner.update(u_idx, action, reported_reward)
        else:
            learner.update(action, reported_reward)

        # Regret: use user's own optimal action
        optimal_action = int(np.argmax(user.true_utilities))
        regret_t = float(user.true_utilities[optimal_action] - user.true_utilities[action])
        cumulative_regret += regret_t

        # Track
        results['actions'].append(action)
        results['true_rewards'].append(true_reward)
        results['reported_rewards'].append(reported_reward)
        results['cumulative_regret'].append(cumulative_regret)

    return results


def run_single_experiment(learner_constructor, users, T, seed=0):
    """
    Runs one learner with both strategic and truthful feedback.
    learner_constructor: a zero-arg callable that returns a fresh learner instance.
    Returns a dict with 'strategic', 'truthful', and 'manipulation_gain'.
    """

    # Strategic run (users report with strategy='greedy')
    learner = learner_constructor()
    strategic_results = _run_once(learner, users, T, seed=seed, strategy='greedy')

    # Truthful run: create a fresh learner using the constructor again
    learner_truth = learner_constructor()
    truthful_results = _run_once(learner_truth, users, T, seed=seed, strategy='truthful')

    # Manipulation gain: mean(true_rewards under strategic) - mean(true_rewards under truthful)
    manip_gain = float(np.mean(strategic_results['true_rewards']) - np.mean(truthful_results['true_rewards']))

    return {
        'strategic': strategic_results,
        'truthful': truthful_results,
        'manipulation_gain': manip_gain
    }


def run_full_experiment(dataset='movielens', n_seeds=5, T=2000, quick_test=False, seed=0, alpha_distribution='uniform'):
    """
    Load or synthesize users, construct learners, run experiments across seeds and methods.
    Accepts alpha_distribution to control strategic user manip intensity sampling.
    """

    rng = np.random.RandomState(seed)
    if quick_test:
        n_users = 10
        n_items = 5
    else:
        try:
            from data.preprocess import load_movielens
            data = load_movielens()
            n_users = data['n_users']
            n_items = data['n_items']
        except Exception:
            print("Warning: MovieLens not found; falling back to synthetic dataset.")
            n_users = 100
            n_items = 50

    utilities = generate_user_utilities(n_users, n_items, seed=seed)
    alphas = generate_alpha_distribution(n_users, distribution=alpha_distribution, seed=seed)

    users = [StrategicUser(true_utilities=utilities[u], alpha=float(alphas[u])) for u in range(n_users)]

    methods = {
        'ern': lambda: ERNLearner(n_items=n_items, beta=1.0, sigma=0.05, eta0=0.01, seed=seed),
        'bsm': lambda: BaselineSoftmaxModel(n_items=n_items, beta=1.0, eta0=0.01, seed=seed),
        'robustmf': lambda: RobustMF(n_users=n_users, n_items=n_items, n_factors=16, lr=0.01, delta=0.5, seed=seed),
        'sasrec': lambda: SASRec(n_items=n_items, n_factors=16, max_history=50, seed=seed),
        'lightgcn': lambda: LightGCN(n_users=n_users, n_items=n_items, n_factors=16, seed=seed)
    }

    all_results = {}
    for method_name, constructor in methods.items():
        method_results = []
        for s in range(n_seeds):
            print(f"Running method {method_name}, seed {s+1}/{n_seeds}...")
            res = run_single_experiment(constructor, users, T=T, seed=seed + s)
            method_results.append(res)
        all_results[method_name] = method_results

    os.makedirs("results", exist_ok=True)
    out_path = f"results/movielens_experiment_{'quick' if quick_test else 'full'}_T{T}_seeds{n_seeds}_{alpha_distribution}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f)
    print(f"Saved results to {out_path}")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true", help="Run quick test (small synthetic)")
    parser.add_argument("--T", type=int, default=1000, help="Rounds per run")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--alpha_dist", type=str, default="uniform", help="alpha distribution: uniform|beta|bimodal|adversarial")
    args = parser.parse_args()
    run_full_experiment(quick_test=args.quick_test, T=args.T, n_seeds=args.n_seeds, alpha_distribution=args.alpha_dist)
