"""
Plot per-category FROC curves for tuberculosis detection.
Visualizes sensitivity vs FPI for each TB category separately.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def plot_per_category_froc(froc_results_file, output_dir=None):
    """
    Plot per-category FROC curves from computed results.
    
    Args:
        froc_results_file: Path to JSON file with FROC results
        output_dir: Directory to save plots (default: same as input file)
    """
    # Load results
    with open(froc_results_file, 'r') as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = Path(froc_results_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color scheme
    colors = {
        'overall': '#2E86AB',
        'ActiveTuberculosis': '#A23B72',
        'ObsoletePulmonaryTuberculosis': '#F18F01',
        'PulmonaryTuberculosis': '#C73E1D'
    }
    
    markers = {
        'overall': 'o',
        'ActiveTuberculosis': 's',
        'ObsoletePulmonaryTuberculosis': '^',
        'PulmonaryTuberculosis': 'D'
    }
    
    max_fpi = results['max_fpi']
    fpi_points = results['fpi_points']
    
    # Plot overall curve
    overall_curve = results['overall']['full_curve']
    fps = np.array(overall_curve['fps'])
    sens = np.array(overall_curve['sensitivity'])
    
    # Limit to max_fpi for cleaner visualization
    mask = fps <= max_fpi * 1.5
    fps_plot = fps[mask]
    sens_plot = sens[mask]
    
    ax.plot(fps_plot, sens_plot * 100, 
            color=colors['overall'], linewidth=2.5, 
            label='Overall (All TB)', alpha=0.8)
    
    # Mark FPI points on overall curve
    for fpi in fpi_points:
        sens_at_fpi = results['overall']['sensitivity_at_fpi'][str(fpi)]
        ax.plot(fpi, sens_at_fpi * 100, 
                marker=markers['overall'], 
                color=colors['overall'], 
                markersize=8, markeredgewidth=1.5, markeredgecolor='white')
    
    # Plot per-category curves
    for cat_name, cat_results in results['per_category'].items():
        if cat_results['num_annotations'] == 0:
            continue
        
        cat_curve = cat_results['full_curve']
        cat_fps = np.array(cat_curve['fps'])
        cat_sens = np.array(cat_curve['sensitivity'])
        
        # Limit to max_fpi
        cat_mask = cat_fps <= max_fpi * 1.5
        cat_fps_plot = cat_fps[cat_mask]
        cat_sens_plot = cat_sens[cat_mask]
        
        # Plot curve
        color = colors.get(cat_name, '#666666')
        marker = markers.get(cat_name, 'x')
        
        ax.plot(cat_fps_plot, cat_sens_plot * 100, 
                color=color, linewidth=2.5, 
                label=f'{cat_name} (n={cat_results["num_annotations"]})', 
                alpha=0.8, linestyle='--')
        
        # Mark FPI points
        for fpi in fpi_points:
            sens_at_fpi = cat_results['sensitivity_at_fpi'][str(fpi)]
            ax.plot(fpi, sens_at_fpi * 100, 
                    marker=marker, 
                    color=color, 
                    markersize=8, markeredgewidth=1.5, markeredgecolor='white')
    
    # Add vertical line at FPI=2.0 (key clinical threshold)
    ax.axvline(x=2.0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(2.0, 5, 'FPI = 2.0\n(Clinical threshold)', 
            ha='center', va='bottom', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('False Positives per Image (FPI)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sensitivity (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Category FROC Curves: TB Detection Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, max_fpi * 1.2)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    # Add text box with key metrics
    textstr = 'Key Metrics at FPI ≤ 2.0:\n'
    textstr += f'Overall: {results["overall"]["sensitivity_at_fpi"]["2.0"]*100:.1f}%\n'
    for cat_name, cat_results in results['per_category'].items():
        if cat_results['num_annotations'] > 0:
            sens = cat_results['sensitivity_at_fpi']['2.0']
            textstr += f'{cat_name}: {sens*100:.1f}%\n'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save figure
    output_base = output_dir / 'per_category_froc_curve'
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    print(f"Saved FROC curves to {output_base}.png and {output_base}.pdf")
    
    plt.close()
    
    # Create comparison bar chart for sensitivity at different FPI
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data
    categories = ['Overall']
    cat_data = {str(fpi): [results['overall']['sensitivity_at_fpi'][str(fpi)] * 100] 
                for fpi in fpi_points}
    
    for cat_name, cat_results in results['per_category'].items():
        if cat_results['num_annotations'] > 0:
            categories.append(f"{cat_name}\n(n={cat_results['num_annotations']})")
            for fpi in fpi_points:
                cat_data[str(fpi)].append(cat_results['sensitivity_at_fpi'][str(fpi)] * 100)
    
    # Bar positions
    x = np.arange(len(categories))
    width = 0.15
    
    # Plot bars for each FPI
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, fpi in enumerate(fpi_points):
        offset = (i - len(fpi_points)/2) * width + width/2
        ax.bar(x + offset, cat_data[str(fpi)], width, 
               label=f'FPI ≤ {fpi}', color=bar_colors[i], alpha=0.8)
    
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sensitivity (%)', fontsize=14, fontweight='bold')
    ax.set_title('Sensitivity Comparison Across FPI Thresholds', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=len(fpi_points))
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save bar chart
    output_base = output_dir / 'per_category_sensitivity_comparison'
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    print(f"Saved comparison chart to {output_base}.png and {output_base}.pdf")
    
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot per-category FROC curves')
    parser.add_argument('--froc-results', type=str, required=True,
                        help='Path to per-category FROC results JSON')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots (default: same as input)')
    
    args = parser.parse_args()
    
    plot_per_category_froc(args.froc_results, args.output_dir)

