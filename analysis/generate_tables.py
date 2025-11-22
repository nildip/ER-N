"""
Generate LaTeX tables for ER-N paper results.
"""
import json
import os
import numpy as np
from glob import glob


def load_experiment_results(pattern):
    """Load all results matching pattern."""
    files = glob(f'results/{pattern}')
    if not files:
        print(f'⚠️  No files found matching: results/{pattern}')
        return None
    
    results = []
    for f in sorted(files):
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)
    return results


def table_main_results(output_path='results/table1_main_results.tex'):
    """Table 1: Main results - ER-N vs BSM across collusion rates."""
    
    print("Generating Table 1: Main Results...")
    
    # Load all main experiment results
    experiments = []
    for rate in [1, 5, 10, 20]:
        pattern = f'main_collusion{rate}pct_*_T10000_seeds5.json'
        files = glob(f'results/{pattern}')
        if files:
            with open(files[0]) as f:
                experiments.append(json.load(f))
    
    if not experiments:
        print("⚠️  No main experiment results found. Run: python run_experiments.py --experiment main")
        return
    
    # Build table
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Collusion Rate & \multicolumn{2}{c}{Target Boost (\%)} & \multicolumn{2}{c}{Manipulation Reduction} \\",
        r" & BSM & ER-N & Absolute & Relative \\",
        r"\hline"
    ]
    
    for exp in experiments:
        rate = exp['metadata']['collusion_rate']
        
        # Get boosts
        bsm_boosts = [r['target_boost'] for r in exp['results']['bsm']]
        ern_boosts = [r['target_boost'] for r in exp['results']['ern']]
        
        bsm_mean = np.mean(bsm_boosts)
        bsm_std = np.std(bsm_boosts)
        ern_mean = np.mean(ern_boosts)
        ern_std = np.std(ern_boosts)
        
        # Calculate reduction
        absolute_reduction = bsm_mean - ern_mean
        relative_reduction = (1 - ern_mean / bsm_mean) * 100 if bsm_mean > 0 else 0
        
        lines.append(
            f"{rate*100:.0f}\\% & "
            f"{bsm_mean:.2f}$\\pm${bsm_std:.2f} & "
            f"{ern_mean:.2f}$\\pm${ern_std:.2f} & "
            f"{absolute_reduction:.2f}\\% & "
            f"{relative_reduction:.1f}\\% \\\\"
        )
    
    lines.extend([
        r"\hline",
        r"\end{tabular}"
    ])
    
    # Write
    os.makedirs('results', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved to: {output_path}\n")


def table_ablation_sigma(output_path='results/table2_ablation_sigma.tex'):
    """Table 2: Ablation study - effect of σ parameter."""
    
    print("Generating Table 2: Sigma Ablation...")
    
    # Load ablation results
    experiments = []
    for sigma in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        pattern = f'ablation_sigma{sigma:.1f}_*_T10000_seeds5.json'
        files = glob(f'results/{pattern}')
        if files:
            with open(files[0]) as f:
                experiments.append((sigma, json.load(f)))
    
    if not experiments:
        print("⚠️  No sigma ablation results found. Run: python run_experiments.py --experiment ablation-sigma")
        return
    
    # Build table
    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"$\sigma$ & Target Boost (\%) & Manipulation Reduction (\%) \\",
        r"\hline"
    ]
    
    for sigma, exp in experiments:
        bsm_boosts = [r['target_boost'] for r in exp['results']['bsm']]
        ern_boosts = [r['target_boost'] for r in exp['results']['ern']]
        
        bsm_mean = np.mean(bsm_boosts)
        ern_mean = np.mean(ern_boosts)
        ern_std = np.std(ern_boosts)
        
        reduction = (1 - ern_mean / bsm_mean) * 100 if bsm_mean > 0 else 0
        
        lines.append(
            f"{sigma:.1f} & "
            f"{ern_mean:.2f}$\\pm${ern_std:.2f} & "
            f"{reduction:.1f}\\% \\\\"
        )
    
    lines.extend([
        r"\hline",
        r"\end{tabular}"
    ])
    
    # Write
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved to: {output_path}\n")


def table_ablation_beta(output_path='results/table3_ablation_beta.tex'):
    """Table 3: Ablation study - effect of β parameter."""
    
    print("Generating Table 3: Beta Ablation...")
    
    experiments = []
    for beta in [1.0, 5.0, 10.0, 20.0, 50.0]:
        pattern = f'ablation_beta{beta:.1f}_*_T10000_seeds5.json'
        files = glob(f'results/{pattern}')
        if files:
            with open(files[0]) as f:
                experiments.append((beta, json.load(f)))
    
    if not experiments:
        print("⚠️  No beta ablation results found. Run: python run_experiments.py --experiment ablation-beta")
        return
    
    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"$\beta$ & Target Boost (\%) & Manipulation Reduction (\%) \\",
        r"\hline"
    ]
    
    for beta, exp in experiments:
        bsm_boosts = [r['target_boost'] for r in exp['results']['bsm']]
        ern_boosts = [r['target_boost'] for r in exp['results']['ern']]
        
        bsm_mean = np.mean(bsm_boosts)
        ern_mean = np.mean(ern_boosts)
        ern_std = np.std(ern_boosts)
        
        reduction = (1 - ern_mean / bsm_mean) * 100 if bsm_mean > 0 else 0
        
        lines.append(
            f"{beta:.1f} & "
            f"{ern_mean:.2f}$\\pm${ern_std:.2f} & "
            f"{reduction:.1f}\\% \\\\"
        )
    
    lines.extend([
        r"\hline",
        r"\end{tabular}"
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved to: {output_path}\n")


def generate_all_tables():
    """Generate all LaTeX tables for the paper."""
    print("="*70)
    print("GENERATING LATEX TABLES")
    print("="*70 + "\n")
    
    table_main_results()
    table_ablation_sigma()
    table_ablation_beta()
    
    print("="*70)
    print("ALL TABLES GENERATED")
    print("="*70)
    print("\nLaTeX tables saved in results/:")
    print("  - table1_main_results.tex")
    print("  - table2_ablation_sigma.tex")
    print("  - table3_ablation_beta.tex")
    print("\nCopy these into your paper's LaTeX source.")


if __name__ == '__main__':
    generate_all_tables()
