"""
Statistical significance tests for ER-N paper.
Tests if ER-N's manipulation reduction is statistically significant vs BSM.
"""
import json
import os
import numpy as np
from scipy import stats
from glob import glob


def paired_ttest(a, b, alpha=0.05):
    """
    Paired t-test comparing two matched samples.
    
    Args:
        a: Array of metric values for method A
        b: Array of metric values for method B (same length as A)
        alpha: Significance level (default 0.05)
    
    Returns:
        dict with t_statistic, p_value, significance, mean_diff, cohen_d
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    if len(a) != len(b):
        raise ValueError("Samples must have same length")
    
    if len(a) < 2:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'mean_diff': float(np.mean(a) - np.mean(b)),
            'cohen_d': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(a, b)
    
    # Mean difference
    mean_diff = float(np.mean(a) - np.mean(b))
    
    # Cohen's d (effect size)
    diff = a - b
    cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    
    # 95% Confidence interval for mean difference
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
    t_critical = stats.t.ppf(1 - alpha/2, len(diff) - 1)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'mean_diff': mean_diff,
        'cohen_d': float(cohen_d),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }


def load_experiment_results(pattern):
    """Load experiment results matching pattern."""
    files = glob(f'results/{pattern}')
    if not files:
        return None
    
    # Sort to ensure consistent ordering
    files = sorted(files)
    
    results = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                results.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️  Error loading {f}: {e}")
            continue
    
    return results if results else None


def test_main_results():
    """Test significance of ER-N vs BSM across collusion rates."""
    
    print("="*70)
    print("STATISTICAL TESTS: Main Results")
    print("="*70 + "\n")
    
    has_results = False
    
    for rate in [1, 5, 10, 20]:
        # Try both synthetic and movielens
        for dataset in ['movielens', 'synthetic']:
            pattern = f'main_collusion{rate}pct_{dataset}_T10000_seeds*.json'
            files = glob(f'results/{pattern}')
            
            if not files:
                continue
            
            # Load first matching file
            try:
                with open(files[0], 'r') as f:
                    data = json.load(f)
            except:
                continue
            
            has_results = True
            
            # Check if results exist
            if 'results' not in data or 'ern' not in data['results'] or 'bsm' not in data['results']:
                print(f"⚠️  Collusion Rate {rate}% ({dataset}): Invalid data structure")
                continue
            
            # Extract target boosts
            ern_boosts = np.array([r['target_boost'] for r in data['results']['ern']])
            bsm_boosts = np.array([r['target_boost'] for r in data['results']['bsm']])
            
            # Check for valid data
            if len(ern_boosts) == 0 or len(bsm_boosts) == 0:
                print(f"⚠️  Collusion Rate {rate}% ({dataset}): No data")
                continue
            
            # Paired t-test (ER-N should have LOWER boost)
            result = paired_ttest(ern_boosts, bsm_boosts)
            
            # Calculate reduction
            mean_reduction = (np.mean(bsm_boosts) - np.mean(ern_boosts)) / (np.mean(bsm_boosts) + 1e-10) * 100
            
            print(f"Collusion Rate: {rate}% ({dataset})")
            print(f"  ER-N boost: {np.mean(ern_boosts):.2f} ± {np.std(ern_boosts):.2f}")
            print(f"  BSM boost:  {np.mean(bsm_boosts):.2f} ± {np.std(bsm_boosts):.2f}")
            print(f"  Mean difference: {result['mean_diff']:.2f}")
            print(f"  95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
            print(f"  Reduction: {mean_reduction:.1f}%")
            print(f"  t-statistic: {result['t_statistic']:.3f}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Cohen's d: {result['cohen_d']:.3f}")
            
            if result['significant']:
                print(f"  ✓ Significant at p<0.05")
            else:
                print(f"  ✗ Not significant (p={result['p_value']:.3f})")
            
            print()
            break  # Only show first found (prefer movielens)
    
    if not has_results:
        print("⚠️  No main experiment results found.")
        print("     Run: python experiments/run_experiments.py --experiment main")
        print()


def test_ablation_sigma():
    """Test if sigma=0 (no noise) is significantly worse than optimal sigma."""
    
    print("="*70)
    print("STATISTICAL TESTS: Sigma Ablation")
    print("="*70 + "\n")
    
    # Try to find sigma ablation results
    sigma_0_files = glob('results/ablation_sigma0.0_*_T10000_seeds*.json')
    sigma_opt_files = glob('results/ablation_sigma0.3_*_T10000_seeds*.json')
    
    if not sigma_0_files or not sigma_opt_files:
        print("⚠️  Missing sigma ablation results")
        print("     Run: python experiments/run_experiments.py --experiment ablation-sigma")
        print()
        return
    
    try:
        with open(sigma_0_files[0]) as f:
            data_0 = json.load(f)
        with open(sigma_opt_files[0]) as f:
            data_opt = json.load(f)
    except:
        print("⚠️  Error loading sigma ablation data")
        return
    
    # Extract ER-N boosts (compare ER-N with different sigmas)
    if 'results' not in data_0 or 'ern' not in data_0['results']:
        print("⚠️  Invalid data structure in sigma ablation")
        return
    
    boosts_0 = np.array([r['target_boost'] for r in data_0['results']['ern']])
    boosts_opt = np.array([r['target_boost'] for r in data_opt['results']['ern']])
    
    if len(boosts_0) == 0 or len(boosts_opt) == 0:
        print("⚠️  No data in sigma ablation")
        return
    
    result = paired_ttest(boosts_0, boosts_opt)
    
    # Calculate reduction from adding noise
    reduction = (np.mean(boosts_0) - np.mean(boosts_opt)) / (np.mean(boosts_0) + 1e-10) * 100
    
    print("ER-N with σ=0.0 vs σ=0.3:")
    print(f"  σ=0.0 (no noise): {np.mean(boosts_0):.2f} ± {np.std(boosts_0):.2f}")
    print(f"  σ=0.3 (default):  {np.mean(boosts_opt):.2f} ± {np.std(boosts_opt):.2f}")
    print(f"  Mean difference: {result['mean_diff']:.2f}")
    print(f"  95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
    print(f"  Improvement from noise: {reduction:.1f}%")
    print(f"  t-statistic: {result['t_statistic']:.3f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Cohen's d: {result['cohen_d']:.3f}")
    
    if result['significant']:
        print(f"  ✓ Significant at p<0.05")
    else:
        print(f"  ✗ Not significant (p={result['p_value']:.3f})")
    
    print()


def test_ablation_beta():
    """Test if beta matters for robustness."""
    
    print("="*70)
    print("STATISTICAL TESTS: Beta Ablation")
    print("="*70 + "\n")
    
    # Find beta ablation results
    beta_low_files = glob('results/ablation_beta1.0_*_T10000_seeds*.json')
    beta_opt_files = glob('results/ablation_beta10.0_*_T10000_seeds*.json')
    
    if not beta_low_files or not beta_opt_files:
        print("⚠️  Missing beta ablation results")
        print("     Run: python experiments/run_experiments.py --experiment ablation-beta")
        print()
        return
    
    try:
        with open(beta_low_files[0]) as f:
            data_low = json.load(f)
        with open(beta_opt_files[0]) as f:
            data_opt = json.load(f)
    except:
        print("⚠️  Error loading beta ablation data")
        return
    
    # Extract ER-N boosts
    if 'results' not in data_low or 'ern' not in data_low['results']:
        print("⚠️  Invalid data structure in beta ablation")
        return
    
    boosts_low = np.array([r['target_boost'] for r in data_low['results']['ern']])
    boosts_opt = np.array([r['target_boost'] for r in data_opt['results']['ern']])
    
    if len(boosts_low) == 0 or len(boosts_opt) == 0:
        print("⚠️  No data in beta ablation")
        return
    
    result = paired_ttest(boosts_low, boosts_opt)
    
    print("ER-N with β=1.0 vs β=10.0:")
    print(f"  β=1.0 (low):      {np.mean(boosts_low):.2f} ± {np.std(boosts_low):.2f}")
    print(f"  β=10.0 (default): {np.mean(boosts_opt):.2f} ± {np.std(boosts_opt):.2f}")
    print(f"  Mean difference: {result['mean_diff']:.2f}")
    print(f"  95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
    print(f"  t-statistic: {result['t_statistic']:.3f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Cohen's d: {result['cohen_d']:.3f}")
    
    if result['significant']:
        print(f"  ✓ Significant at p<0.05")
    else:
        print(f"  ✗ Not significant (p={result['p_value']:.3f})")
    
    print()


def generate_summary_statistics():
    """Generate summary statistics across all experiments."""
    
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70 + "\n")
    
    # Find all result files
    all_files = glob('results/*.json')
    
    if not all_files:
        print("⚠️  No results found in results/ directory")
        print("     Run experiments first")
        return
    
    print(f"Total result files found: {len(all_files)}\n")
    
    # Categorize by experiment type
    main_files = [f for f in all_files if 'main_collusion' in f]
    sigma_files = [f for f in all_files if 'ablation_sigma' in f]
    beta_files = [f for f in all_files if 'ablation_beta' in f]
    
    print(f"Main experiments: {len(main_files)}")
    print(f"Sigma ablations: {len(sigma_files)}")
    print(f"Beta ablations: {len(beta_files)}")
    print()
    
    # Analyze main experiments
    if main_files:
        print("Main Experiment Coverage:")
        for rate in [1, 5, 10, 20]:
            matching = [f for f in main_files if f'collusion{rate}pct' in f]
            status = "✓" if matching else "✗"
            print(f"  {status} {rate}% collusion: {len(matching)} file(s)")
        print()
    
    # Analyze sigma ablations
    if sigma_files:
        print("Sigma Ablation Coverage:")
        for sigma in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            matching = [f for f in sigma_files if f'sigma{sigma:.1f}' in f]
            status = "✓" if matching else "✗"
            print(f"  {status} σ={sigma:.1f}: {len(matching)} file(s)")
        print()
    
    # Analyze beta ablations
    if beta_files:
        print("Beta Ablation Coverage:")
        for beta in [1.0, 5.0, 10.0, 20.0, 50.0]:
            matching = [f for f in beta_files if f'beta{beta:.1f}' in f]
            status = "✓" if matching else "✗"
            print(f"  {status} β={beta:.1f}: {len(matching)} file(s)")
        print()


def check_data_quality():
    """Check for data quality issues (negative values, extreme variance, etc.)."""
    
    print("="*70)
    print("DATA QUALITY CHECKS")
    print("="*70 + "\n")
    
    all_files = glob('results/*.json')
    
    if not all_files:
        print("⚠️  No results to check")
        return
    
    issues_found = False
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        try:
            with open(filepath) as f:
                data = json.load(f)
        except:
            print(f"⚠️  {filename}: Cannot load file")
            issues_found = True
            continue
        
        if 'results' not in data:
            print(f"⚠️  {filename}: Missing 'results' key")
            issues_found = True
            continue
        
        for method in ['ern', 'bsm']:
            if method not in data['results']:
                continue
            
            boosts = [r.get('target_boost', np.nan) for r in data['results'][method]]
            
            # Check for NaN values
            if any(np.isnan(boosts)):
                print(f"⚠️  {filename} ({method}): Contains NaN values")
                issues_found = True
            
            # Check for negative values
            if any(b < 0 for b in boosts if not np.isnan(b)):
                print(f"⚠️  {filename} ({method}): Contains negative boost values")
                issues_found = True
            
            # Check for extreme variance (CV > 200%)
            mean_boost = np.mean([b for b in boosts if not np.isnan(b)])
            std_boost = np.std([b for b in boosts if not np.isnan(b)])
            if mean_boost > 0 and std_boost / mean_boost > 2.0:
                print(f"⚠️  {filename} ({method}): Very high variance (CV={std_boost/mean_boost*100:.0f}%)")
                issues_found = True
    
    if not issues_found:
        print("✓ No data quality issues detected")
    
    print()


def run_all_tests():
    """Run all statistical tests."""
    
    print("\n" + "="*70)
    print("RUNNING ALL STATISTICAL SIGNIFICANCE TESTS")
    print("="*70 + "\n")
    
    # Check if scipy is available
    try:
        from scipy import stats
    except ImportError:
        print("ERROR: scipy is not installed")
        print("Install with: pip install scipy")
        return
    
    # Summary statistics
    generate_summary_statistics()
    
    # Data quality checks
    check_data_quality()
    
    # Main tests
    test_main_results()
    test_ablation_sigma()
    test_ablation_beta()
    
    print("="*70)
    print("TESTS COMPLETE")
    print("="*70)
    print("\nInterpretation Guide:")
    print("  - p < 0.05: Statistically significant difference")
    print("  - p < 0.01: Highly significant")
    print("  - p < 0.001: Very highly significant")
    print("  - Cohen's d > 0.8: Large effect size")
    print("  - Cohen's d > 0.5: Medium effect size")
    print("  - Cohen's d > 0.2: Small effect size")
    print("\n✓ YES: ER-N significantly outperforms baseline")
    print("✗ NO: Difference not statistically significant")
    print("\nNote: Non-significant results may indicate:")
    print("  - High variance (need more seeds)")
    print("  - Weak attack (increase α or change target)")
    print("  - Insufficient effect size (check experimental design)")
    print()


if __name__ == '__main__':
    run_all_tests()
