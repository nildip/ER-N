"""
Statistical significance tests for ER-N paper.
Tests if ER-N's manipulation reduction is statistically significant vs Baselines.
"""
import json
import os
import numpy as np
from scipy import stats
from glob import glob


def paired_ttest(a, b):
    """Paired t-test comparing two matched samples."""
    t_stat, p_value = stats.ttest_rel(a, b)
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'mean_diff': float(np.mean(a) - np.mean(b))
    }


def test_main_results():
    """Test significance of ER-N vs BSM across collusion rates."""
    
    print("="*70)
    print("STATISTICAL TESTS: Main Results")
    print("="*70 + "\n")
    
    for rate in [1, 5, 10, 20]:
        pattern = f'results/main_collusion{rate}pct_*_T10000_seeds5.json'
        files = glob(pattern)
        
        if not files:
            print(f"⚠️  No results found for {rate}% collusion")
            continue
        
        with open(files[0]) as f:
            data = json.load(f)
        
        # Extract target boosts
        ern_boosts = np.array([r['target_boost'] for r in data['results']['ern']])
        bsm_boosts = np.array([r['target_boost'] for r in data['results']['bsm']])
        
        # Paired t-test (ER-N should have LOWER boost)
        result = paired_ttest(ern_boosts, bsm_boosts)
        
        print(f"Collusion Rate: {rate}%")
        print(f"  ER-N boost: {np.mean(ern_boosts):.2f} ± {np.std(ern_boosts):.2f}")
        print(f"  BSM boost:  {np.mean(bsm_boosts):.2f} ± {np.std(bsm_boosts):.2f}")
        print(f"  Mean difference: {result['mean_diff']:.2f}")
        print(f"  t-statistic: {result['t_statistic']:.3f}")
        print(f"  p-value: {result['p_value']:.4f}")
        print(f"  Significant (p<0.05): {'✓ YES' if result['significant'] else '✗ NO'}")
        print()


def test_ablation_sigma():
    """Test if sigma=0 (no noise) is significantly worse than optimal sigma."""
    
    print("="*70)
    print("STATISTICAL TESTS: Sigma Ablation")
    print("="*70 + "\n")
    
    # Load sigma=0.0 (no robustness)
    files_0 = glob('results/ablation_sigma0.0_*_T10000_seeds5.json')
    # Load sigma=0.3 (default)
    files_3 = glob('results/ablation_sigma0.3_*_T10000_seeds5.json')
    
    if not files_0 or not files_3:
        print("⚠️  Missing sigma ablation results")
        return
    
    with open(files_0[0]) as f:
        data_0 = json.load(f)
    with open(files_3[0]) as f:
        data_3 = json.load(f)
    
    # Compare ER-N with sigma=0.0 vs sigma=0.3
    boosts_0 = np.array([r['target_boost'] for r in data_0['results']['ern']])
    boosts_3 = np.array([r['target_boost'] for r in data_3['results']['ern']])
    
    result = paired_ttest(boosts_0, boosts_3)
    
    print("ER-N with σ=0.0 vs σ=0.3:")
    print(f"  σ=0.0 (no noise): {np.mean(boosts_0):.2f} ± {np.std(boosts_0):.2f}")
    print(f"  σ=0.3 (default):  {np.mean(boosts_3):.2f} ± {np.std(boosts_3):.2f}")
    print(f"  Mean difference: {result['mean_diff']:.2f}")
    print(f"  t-statistic: {result['t_statistic']:.3f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Significant (p<0.05): {'✓ YES' if result['significant'] else '✗ NO'}")
    print()


def test_ablation_beta():
    """Test if beta matters for robustness."""
    
    print("="*70)
    print("STATISTICAL TESTS: Beta Ablation")
    print("="*70 + "\n")
    
    # Load beta=1.0 (low exploration)
    files_1 = glob('results/ablation_beta1.0_*_T10000_seeds5.json')
    # Load beta=10.0 (default)
    files_10 = glob('results/ablation_beta10.0_*_T10000_seeds5.json')
    
    if not files_1 or not files_10:
        print("⚠️  Missing beta ablation results")
        return
    
    with open(files_1[0]) as f:
        data_1 = json.load(f)
    with open(files_10[0]) as f:
        data_10 = json.load(f)
    
    boosts_1 = np.array([r['target_boost'] for r in data_1['results']['ern']])
    boosts_10 = np.array([r['target_boost'] for r in data_10['results']['ern']])
    
    result = paired_ttest(boosts_1, boosts_10)
    
    print("ER-N with β=1.0 vs β=10.0:")
    print(f"  β=1.0 (low):     {np.mean(boosts_1):.2f} ± {np.std(boosts_1):.2f}")
    print(f"  β=10.0 (default): {np.mean(boosts_10):.2f} ± {np.std(boosts_10):.2f}")
    print(f"  Mean difference: {result['mean_diff']:.2f}")
    print(f"  t-statistic: {result['t_statistic']:.3f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Significant (p<0.05): {'✓ YES' if result['significant'] else '✗ NO'}")
    print()


def run_all_tests():
    """Run all statistical tests."""
    
    print("\n" + "="*70)
    print("RUNNING ALL STATISTICAL SIGNIFICANCE TESTS")
    print("="*70 + "\n")
    
    test_main_results()
    test_ablation_sigma()
    test_ablation_beta()
    
    print("="*70)
    print("TESTS COMPLETE")
    print("="*70)
    print("\nInterpretation:")
    print("  - p < 0.05: Statistically significant difference")
    print("  - ✓ YES: ER-N significantly outperforms baseline")
    print("  - ✗ NO: Difference not statistically significant\n")


if __name__ == '__main__':
    run_all_tests()
```

**Add to `requirements.txt`:**
```
scipy
