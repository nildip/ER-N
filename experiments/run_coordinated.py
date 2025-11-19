"""
Run coordinated manipulation experiments.
"""
import argparse
import json
import os
import numpy as np
import random

from src.ern import ERNLearner
from src.baselines import BaselineSoftmaxModel, RobustMF
from src.strategic_users import generate_user_utilities
from src.coordinated_manipulation import CoordinatedAttack


def run_coordinated_single(learner, attack, utilities, T, seed, strategy='coordinated'):
    """
    Run single experiment with coordinated manipulation.
    
    Returns dict with:
    - true_rewards: List of true rewards received
    - target_recs: Number of times target was recommended
    - target_policy_trace: Policy mass on target item over time
    """
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
        # Sample user
        u_idx = np.random.randint(n_users)
        
        # Get action from learner
        if hasattr(learner, 'get_policy'):
            try:
                # Try user-aware first
                policy = learner.get_policy(u_idx)
            except:
                # Fall back to user-agnostic
                policy = learner.get_policy()
            
            # Normalize policy
            policy = np.asarray(policy, dtype=float)
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                policy = np.ones(n_items) / n_items
            
            action = int(np.random.choice(n_items, p=policy))
            
            # Track policy on target every 100 rounds
            if t % 100 == 0:
                results['target_policy_trace'].append(float(policy[attack.target_item]))
        else:
            action = np.random.randint(n_items)
            if t % 100 == 0:
                results['target_policy_trace'].append(1.0 / n_items)
        
        # Track target recommendations
        if action == attack.target_item:
            results['target_recs'] += 1
        
        # Get true reward
        true_reward = float(utilities[u_idx, action])
        results['true_rewards'].append(true_reward)
        
        # Get feedback (potentially manipulated)
        feedback = attack.get_feedback(u_idx, action, true_reward, strategy=strategy)
        
        # Update learner
        if isinstance(learner, RobustMF):
            learner.update(u_idx, action, feedback)
        else:
            learner.update(action, feedback)
    
    return results


def run_coordinated_experiment(collusion_rate=0.05, n_seeds=5, T=10000, K=None, use_real_data=False):
    """
    Run full coordinated manipulation experiment.
    
    Args:
        collusion_rate: Fraction of users colluding
        n_seeds: Number of random seeds
        T: Number of rounds
        K: Number of items (for synthetic) or top-K filter (for MovieLens)
        use_real_data: If True, load MovieLens; if False, use synthetic
    """
    
    # Load dataset
    if use_real_data:
        try:
            if K is None:
                # Load full MovieLens
                from data.preprocess import load_movielens
                data = load_movielens()
                n_users = data['n_users']
                n_items = data['n_items']
                print(f"Loaded MovieLens (full): {n_users} users, {n_items} items")
            else:
                # Load MovieLens top-K
                from data.preprocess import load_movielens_topK
                data = load_movielens_topK(K=K)
                n_users = data['n_users']
                n_items = data['n_items']
                print(f"Loaded MovieLens (top-{K}): {n_users} users, {n_items} items")
        except Exception as e:
            print(f"⚠️  Failed to load MovieLens: {e}")
            print(f"   Falling back to synthetic data")
            n_users = 1000
            n_items = K if K is not None else 200
            print(f"Using synthetic: {n_users} users, {n_items} items")
    else:
        # Synthetic data (default)
        n_users = 1000
        n_items = K if K is not None else 200
        print(f"Using synthetic: {n_users} users, {n_items} items")
    
    # Generate utilities
    utilities = generate_user_utilities(n_users, n_items, seed=42)
    
    # Choose target: pick item at 25th percentile popularity
    target_item = n_items // 4
    
    print(f"\nExperiment Configuration:")
    print(f"  Collusion rate: {collusion_rate*100:.0f}%")
    print(f"  Colluding users: {int(n_users * collusion_rate)}/{n_users}")
    print(f"  Target item: {target_item}")
    print(f"  Rounds: {T:,}")
    print(f"  Seeds: {n_seeds}\n")
    
    # Create attack
    attack = CoordinatedAttack(
        n_users=n_users,
        collusion_rate=collusion_rate,
        target_item=target_item,
        alpha=0.8,
        seed=42
    )
    
    # Methods (REMOVED RobustMF)
    methods = {
        'ern': lambda: ERNLearner(n_items=n_items, beta=10.0, sigma=0.3, eta0=0.2, seed=42),
        'bsm': lambda: BaselineSoftmaxModel(n_items=n_items, beta=10.0, eta0=0.2, seed=42),
    }
    
    all_results = {}
    
    for method_name, constructor in methods.items():
        print(f"Running {method_name.upper()}...")
        method_results = []
        
        for s in range(n_seeds):
            print(f"  Seed {s+1}/{n_seeds}...", end=' ')
            
            # Strategic (coordinated attack)
            learner_strat = constructor()
            strategic = run_coordinated_single(
                learner_strat, attack, utilities, T, 
                seed=42+s, strategy='coordinated'
            )
            
            # Truthful (no coordination)
            learner_truth = constructor()
            truthful = run_coordinated_single(
                learner_truth, attack, utilities, T, 
                seed=42+s, strategy='truthful'
            )
            
            # Compute metrics
            result = {
                'strategic': strategic,
                'truthful': truthful,
                'target_boost': (strategic['target_recs'] - truthful['target_recs']) / T * 100,
                'mean_policy_strategic': np.mean(strategic['target_policy_trace']) * 100,
                'mean_policy_truthful': np.mean(truthful['target_policy_trace']) * 100
            }
            
            print(f"Target boost: {result['target_boost']:.2f}%")
            method_results.append(result)
        
        all_results[method_name] = method_results
    
    # Print summary
    print("\n" + "="*70)
    print(f"RESULTS SUMMARY: {collusion_rate*100:.0f}% Collusion")
    print("="*70)
    
    for method in methods.keys():
        boosts = [r['target_boost'] for r in all_results[method]]
        policy_strat = [r['mean_policy_strategic'] for r in all_results[method]]
        policy_truth = [r['mean_policy_truthful'] for r in all_results[method]]
        
        print(f"\n{method.upper()}:")
        print(f"  Target item recommendation boost: {np.mean(boosts):.2f}% ± {np.std(boosts):.2f}%")
        print(f"  Policy mass on target (strategic): {np.mean(policy_strat):.3f}%")
        print(f"  Policy mass on target (truthful):  {np.mean(policy_truth):.3f}%")
        print(f"  Net manipulation effect: {np.mean(policy_strat) - np.mean(policy_truth):.3f}%")
    
    # Calculate ER-N vs BSM reduction
    ern_boost = np.mean([r['target_boost'] for r in all_results['ern']])
    bsm_boost = np.mean([r['target_boost'] for r in all_results['bsm']])
    reduction = (1 - ern_boost / bsm_boost) * 100 if bsm_boost > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"ER-N REDUCES MANIPULATION BY: {reduction:.1f}%")
    print(f"{'='*70}\n")
    
    # Save
    os.makedirs('results', exist_ok=True)
    dataset_suffix = 'movielens' if use_real_data else 'synthetic'
    outpath = f'results/coordinated_{int(collusion_rate*100)}pct_{dataset_suffix}_T{T}_seeds{n_seeds}.json'
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {outpath}")
    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collusion', type=float, default=0.05, 
                       help='Collusion rate (e.g., 0.05 for 5%%)')
    parser.add_argument('--T', type=int, default=10000,
                       help='Number of rounds')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of random seeds')
    parser.add_argument('--K', type=int, default=None, 
                       help='Number of items (synthetic) or top-K filter (MovieLens)')
    parser.add_argument('--real-data', action='store_true',
                       help='Use MovieLens data instead of synthetic')
    args = parser.parse_args()
    
    run_coordinated_experiment(
        collusion_rate=args.collusion,
        n_seeds=args.n_seeds,
        T=args.T,
        K=args.K,
        use_real_data=args.real_data
    )
