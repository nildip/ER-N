"""
Unified experiment runner for ER-N paper.
Handles: main experiments, ablations, parameter sweeps.
"""
import argparse
import json
import os
import numpy as np
import random

from src.ern import ERNLearner
from src.baselines import BaselineSoftmaxModel
from src.coordinated_manipulation import CoordinatedAttack


def generate_user_utilities(n_users, n_items, seed=42):
    """
    Generate synthetic user utilities using Beta distribution.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        seed: Random seed for reproducibility
    
    Returns:
        utilities: (n_users, n_items) array with values in [0, 1]
    """
    rng = np.random.RandomState(seed)
    return rng.beta(2.0, 2.0, size=(n_users, n_items)).astype(np.float64)


def ratings_to_utilities(data):
    """Convert MovieLens ratings to utility matrix."""
    n_users = data['n_users']
    n_items = data['n_items']
    
    users = data['users']
    items = data['items']
    ratings = data['ratings']
    
    # Initialize with zeros
    utilities = np.zeros((n_users, n_items))
    
    # Fill in actual ratings
    print(f"  Converting {len(ratings)} ratings to utility matrix...")
    for u, i, r in zip(users, items, ratings):
        utilities[u, i] = r
    
    # Fill missing with global mean (standard imputation)
    global_mean = ratings.mean()
    utilities[utilities == 0] = global_mean
    
    sparsity = (len(ratings) / (n_users * n_items)) * 100
    print(f"  Utility matrix: {n_users}×{n_items}, {sparsity:.2f}% real ratings")
    print(f"  Missing entries filled with global mean: {global_mean:.3f}\n")
    
    return utilities


def run_single(learner, attack, utilities, T, seed, strategy='coordinated'):
    """Run single experiment."""
    np.random.seed(seed)
    random.seed(seed)
    
    n_users = utilities.shape[0]
    n_items = utilities.shape[1]
    
    results = {
        'true_rewards': [],
        'target_recs': 0,
        'target_policy_trace': []
    }
    
    for t in range(T):
        u_idx = np.random.randint(n_users)
        
        policy = learner.get_policy()
        policy = np.asarray(policy, dtype=float)
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = np.ones(n_items) / n_items
        
        action = int(np.random.choice(n_items, p=policy))
        
        if t % 100 == 0:
            results['target_policy_trace'].append(float(policy[attack.target_item]))
        
        if action == attack.target_item:
            results['target_recs'] += 1
        
        true_reward = float(utilities[u_idx, action])
        results['true_rewards'].append(true_reward)
        
        feedback = attack.get_feedback(u_idx, action, true_reward, strategy=strategy)
        learner.update(action, feedback)
    
    return results


def run_experiment(collusion_rate=0.05, n_seeds=5, T=10000, K=None, 
                   use_real_data=False, beta=10.0, sigma=0.3, eta0=0.2, alpha=0.8,
                   exp_name='main'):
    """Run coordinated manipulation experiment with configurable parameters."""
    
    # Load data
    if use_real_data:
        try:
            if K is None:
                from data.preprocess import load_movielens
                data = load_movielens()
            else:
                from data.preprocess import load_movielens_topK
                data = load_movielens_topK(K=K)
            
            n_users = data['n_users']
            n_items = data['n_items']
            utilities = ratings_to_utilities(data)
            print(f"✓ Loaded MovieLens: {n_users} users, {n_items} items\n")
        except Exception as e:
            print(f"⚠️  Failed to load MovieLens: {e}")
            print(f"   Falling back to synthetic\n")
            n_users = 1000
            n_items = K if K is not None else 200
            utilities = generate_user_utilities(n_users, n_items, seed=42)
    else:
        n_users = 1000
        n_items = K if K is not None else 200
        utilities = generate_user_utilities(n_users, n_items, seed=42)
        print(f"Using synthetic: {n_users} users, {n_items} items\n")
    
    target_item = n_items // 4
    
    print(f"Experiment: {exp_name}")
    print(f"  Collusion: {collusion_rate*100:.0f}% ({int(n_users * collusion_rate)}/{n_users} users)")
    print(f"  Target item: {target_item}")
    print(f"  Rounds: {T:,}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Params: β={beta}, σ={sigma}, η₀={eta0}, α={alpha}\n")
    
    attack = CoordinatedAttack(
        n_users=n_users,
        collusion_rate=collusion_rate,
        target_item=target_item,
        alpha=alpha,
        seed=42
    )
    
    methods = {
        'ern': lambda: ERNLearner(n_items=n_items, beta=beta, sigma=sigma, eta0=eta0, seed=42),
        'bsm': lambda: BaselineSoftmaxModel(n_items=n_items, beta=beta, eta0=eta0, seed=42),
    }
    
    all_results = {}
    
    for method_name, constructor in methods.items():
        print(f"Running {method_name.upper()}...")
        method_results = []
        
        for s in range(n_seeds):
            print(f"  Seed {s+1}/{n_seeds}...", end=' ')
            
            learner_strat = constructor()
            strategic = run_single(learner_strat, attack, utilities, T, seed=42+s, strategy='coordinated')
            
            learner_truth = constructor()
            truthful = run_single(learner_truth, attack, utilities, T, seed=42+s, strategy='truthful')
            
            result = {
                'strategic': strategic,
                'truthful': truthful,
                'target_boost': (strategic['target_recs'] - truthful['target_recs']) / T * 100,
                'mean_policy_strategic': np.mean(strategic['target_policy_trace']) * 100,
                'mean_policy_truthful': np.mean(truthful['target_policy_trace']) * 100
            }
            
            print(f"Boost: {result['target_boost']:.2f}%")
            method_results.append(result)
        
        all_results[method_name] = method_results
    
    # Summary
    print("\n" + "="*70)
    print(f"RESULTS: {exp_name}")
    print("="*70)
    
    for method in methods.keys():
        boosts = [r['target_boost'] for r in all_results[method]]
        policy_strat = [r['mean_policy_strategic'] for r in all_results[method]]
        policy_truth = [r['mean_policy_truthful'] for r in all_results[method]]
        
        print(f"\n{method.upper()}:")
        print(f"  Target boost: {np.mean(boosts):.2f}% ± {np.std(boosts):.2f}%")
        print(f"  Policy (strategic): {np.mean(policy_strat):.3f}%")
        print(f"  Policy (truthful): {np.mean(policy_truth):.3f}%")
    
    ern_boost = np.mean([r['target_boost'] for r in all_results['ern']])
    bsm_boost = np.mean([r['target_boost'] for r in all_results['bsm']])
    reduction = (1 - ern_boost / bsm_boost) * 100 if bsm_boost > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"ER-N REDUCES MANIPULATION BY: {reduction:.1f}%")
    print(f"{'='*70}\n")
    
    # Save
    os.makedirs('results', exist_ok=True)
    dataset_suffix = 'movielens' if use_real_data else 'synthetic'
    outpath = f'results/{exp_name}_{dataset_suffix}_T{T}_seeds{n_seeds}.json'
    
    # Add metadata
    metadata = {
        'collusion_rate': collusion_rate,
        'n_users': n_users,
        'n_items': n_items,
        'T': T,
        'n_seeds': n_seeds,
        'beta': beta,
        'sigma': sigma,
        'eta0': eta0,
        'alpha': alpha,
        'target_item': target_item
    }
    
    output = {
        'metadata': metadata,
        'results': all_results
    }
    
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved to: {outpath}\n")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='ER-N Experiments')
    
    # Experiment type
    parser.add_argument('--experiment', type=str, default='main',
                       choices=['main', 'ablation-sigma', 'ablation-beta', 
                               'ablation-collusion', 'ablation-alpha', 'all'],
                       help='Which experiment to run')
    
    # Basic params
    parser.add_argument('--T', type=int, default=10000, help='Number of rounds')
    parser.add_argument('--n_seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--real-data', action='store_true', help='Use MovieLens data')
    parser.add_argument('--K', type=int, default=None, help='Top-K items filter')
    
    # Algorithm params
    parser.add_argument('--beta', type=float, default=10.0, help='Exploration temperature')
    parser.add_argument('--sigma', type=float, default=0.3, help='ER-N noise parameter')
    parser.add_argument('--eta0', type=float, default=0.2, help='Learning rate')
    
    # Attack params
    parser.add_argument('--collusion', type=float, default=0.05, help='Collusion rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='Attack strength')
    
    args = parser.parse_args()
    
    if args.experiment == 'main':
        # Main experiment: vary collusion rate
        print("="*70)
        print("MAIN EXPERIMENT: Varying Collusion Rates")
        print("="*70 + "\n")
        
        for rate in [0.01, 0.05, 0.10, 0.20]:
            run_experiment(
                collusion_rate=rate,
                n_seeds=args.n_seeds,
                T=args.T,
                K=args.K,
                use_real_data=args.real_data,  # FIX: Now passing this
                beta=args.beta,
                sigma=args.sigma,
                eta0=args.eta0,
                alpha=args.alpha,
                exp_name=f'main_collusion{int(rate*100)}pct'
            )
    
    elif args.experiment == 'ablation-sigma':
        # Ablate sigma (most important!)
        print("="*70)
        print("ABLATION: σ (Robustness Noise)")
        print("="*70 + "\n")
        
        for sigma in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            run_experiment(
                collusion_rate=args.collusion,
                n_seeds=args.n_seeds,
                T=args.T,
                K=args.K,
                use_real_data=args.real_data,  # FIX
                beta=args.beta,
                sigma=sigma,
                eta0=args.eta0,
                alpha=args.alpha,
                exp_name=f'ablation_sigma{sigma:.1f}'
            )
    
    elif args.experiment == 'ablation-beta':
        # Ablate beta
        print("="*70)
        print("ABLATION: β (Exploration Temperature)")
        print("="*70 + "\n")
        
        for beta in [1.0, 5.0, 10.0, 20.0, 50.0]:
            run_experiment(
                collusion_rate=args.collusion,
                n_seeds=args.n_seeds,
                T=args.T,
                K=args.K,
                use_real_data=args.real_data,  # FIX
                beta=beta,
                sigma=args.sigma,
                eta0=args.eta0,
                alpha=args.alpha,
                exp_name=f'ablation_beta{beta:.1f}'
            )
    
    elif args.experiment == 'ablation-collusion':
        # Same as main but explicit
        print("="*70)
        print("ABLATION: Collusion Rate")
        print("="*70 + "\n")
        
        for rate in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]:
            run_experiment(
                collusion_rate=rate,
                n_seeds=args.n_seeds,
                T=args.T,
                K=args.K,
                use_real_data=args.real_data,  # FIX
                beta=args.beta,
                sigma=args.sigma,
                eta0=args.eta0,
                alpha=args.alpha,
                exp_name=f'ablation_collusion{int(rate*100)}pct'
            )
    
    elif args.experiment == 'ablation-alpha':
        # Ablate attack strength
        print("="*70)
        print("ABLATION: α (Attack Coordination Strength)")
        print("="*70 + "\n")
        
        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            run_experiment(
                collusion_rate=args.collusion,
                n_seeds=args.n_seeds,
                T=args.T,
                K=args.K,
                use_real_data=args.real_data,  # FIX
                beta=args.beta,
                sigma=args.sigma,
                eta0=args.eta0,
                alpha=alpha,
                exp_name=f'ablation_alpha{alpha:.1f}'
            )
    
    elif args.experiment == 'all':
        # Run everything
        print("="*70)
        print("RUNNING ALL EXPERIMENTS")
        print("="*70 + "\n")
        
        # Main - FIX: Now passing all args
        for rate in [0.01, 0.05, 0.10, 0.20]:
            run_experiment(
                collusion_rate=rate, 
                n_seeds=args.n_seeds, 
                T=args.T,
                K=args.K,
                use_real_data=args.real_data,  # FIX: Added
                beta=args.beta,
                sigma=args.sigma,
                eta0=args.eta0,
                alpha=args.alpha,
                exp_name=f'main_collusion{int(rate*100)}pct'
            )
        
        # Sigma ablation - FIX: Now passing all args
        for sigma in [0.0, 0.1, 0.2, 0.3, 0.5]:
            run_experiment(
                collusion_rate=0.05, 
                n_seeds=args.n_seeds, 
                T=args.T,
                K=args.K,
                use_real_data=args.real_data,  # FIX: Added
                beta=args.beta,
                sigma=sigma, 
                eta0=args.eta0,
                alpha=args.alpha,
                exp_name=f'ablation_sigma{sigma:.1f}'
            )
        
        # Beta ablation - FIX: Now passing all args
        for beta in [1.0, 5.0, 10.0, 20.0]:
            run_experiment(
                collusion_rate=0.05, 
                n_seeds=args.n_seeds, 
                T=args.T,
                K=args.K,
                use_real_data=args.real_data,  # FIX: Added
                beta=beta, 
                sigma=args.sigma,
                eta0=args.eta0,
                alpha=args.alpha,
                exp_name=f'ablation_beta{beta:.1f}'
            )
        
        print("="*70)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*70)


if __name__ == '__main__':
    main()
