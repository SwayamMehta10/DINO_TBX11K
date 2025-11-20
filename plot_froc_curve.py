"""
Plot FROC curve from saved evaluation results
Computes and visualizes Free-Response ROC (FROC) curve for TB detection
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pycocotools.coco import COCO
from util.froc_eval import compute_froc_from_coco_results


def plot_froc_curve(froc_results, output_path, title="FROC Curve - TB Detection"):
    """
    Plot FROC curve from computed results
    
    Args:
        froc_results: Dictionary containing FROC metrics
        output_path: Path to save the plot
        title: Title for the plot
    """
    # Extract full FROC curve data
    fp_per_image = np.array(froc_results['froc_curve']['fp_per_image'])
    sensitivity = np.array(froc_results['froc_curve']['sensitivity'])
    
    # Extract specific FPI points for marking
    fpi_values = froc_results['fpi_values']
    sensitivities_at_fpi = froc_results['sensitivities_at_fpi']
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Plot full FROC curve
    plt.plot(fp_per_image, sensitivity, 'b-', linewidth=2, label='FROC Curve')
    
    # Mark specific FPI points
    colors = ['red', 'orange', 'yellow', 'green', 'cyan']
    for i, (fpi, sens) in enumerate(zip(fpi_values, sensitivities_at_fpi)):
        color = colors[i % len(colors)]
        plt.plot(fpi, sens, 'o', color=color, markersize=10, 
                label=f'FPI={fpi:.3f}, Sens={sens:.3f}')
    
    # Highlight FPI <= 2.0 (paper's primary metric)
    plt.axvline(x=2.0, color='red', linestyle='--', linewidth=1.5, 
                label=f'FPI=2.0', alpha=0.7)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and title
    plt.xlabel('False Positives Per Image (FPI)', fontsize=12)
    plt.ylabel('Sensitivity (Recall)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set axis limits
    max_fpi = max(2.0, max(fpi_values) * 1.1) if len(fpi_values) > 0 else 8.0
    plt.xlim([0, min(max_fpi, 8.0)])  # Limit to max FPI of 8.0
    plt.ylim([0, 1.05])
    
    # Add legend
    plt.legend(loc='lower right', fontsize=9)
    
    # Add text box with key metrics
    textstr = f'Mean Sensitivity: {froc_results["mean_sensitivity"]:.4f}\n'
    textstr += f'Sensitivity @ FPI≤2: {froc_results["sensitivity_at_2fpi"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"FROC curve saved to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"FROC curve (PDF) saved to: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot FROC curve from evaluation results')
    parser.add_argument('--coco_path', type=str, required=True,
                       help='Path to TBX11K dataset root')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to detection results JSON file (COCO format)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for plots and metrics')
    parser.add_argument('--ann_file', type=str, default='annotations/json/TBX11K_val.json',
                       help='Annotation file relative to coco_path')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching predictions to ground truth')
    parser.add_argument('--fpi_points', type=float, nargs='+', 
                       default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                       help='FPI points to evaluate (default: 0.125 0.25 0.5 1 2 4 8)')
    parser.add_argument('--title', type=str, default='FROC Curve - TB Detection (TBX11K)',
                       help='Title for the plot')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load ground truth
        ann_file = Path(args.coco_path) / args.ann_file
        print(f"Loading ground truth from: {ann_file}")
        coco_gt = COCO(str(ann_file))
        
        # Load detection results
        print(f"Loading detection results from: {args.results_file}")
        coco_dt = coco_gt.loadRes(args.results_file)
        
        print(f"Ground truth: {len(coco_gt.getImgIds())} images")
        print(f"Detections: loaded successfully")
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Compute FROC metrics
        print(f"\nComputing FROC metrics with FPI points: {args.fpi_points}")
        froc_results = compute_froc_from_coco_results(
            coco_gt, 
            coco_dt,
            iou_threshold=args.iou_threshold,
            fp_per_image_range=args.fpi_points
        )
        
        # Save results to JSON
        json_path = output_dir / 'froc_results.json'
        with open(json_path, 'w') as f:
            json.dump(froc_results, f, indent=2)
        print(f"\nFROC results saved to: {json_path}")
        
        # Print results
        print("\n" + "="*60)
        print("FROC Evaluation Results")
        print("="*60)
        
        fpi_values = froc_results['fpi_values']
        sensitivities = froc_results['sensitivities_at_fpi']
        
        print("\nSensitivity at different FPI values:")
        print("-" * 40)
        for fpi, sens in zip(fpi_values, sensitivities):
            marker = " ← Paper metric" if fpi == 2.0 else ""
            print(f"  FPI = {fpi:5.3f}:  Sensitivity = {sens:.4f}{marker}")
        
        print("\n" + "-" * 40)
        print(f"Mean Sensitivity: {froc_results['mean_sensitivity']:.4f}")
        print(f"Sensitivity at FPI≤2: {froc_results['sensitivity_at_2fpi']:.4f}")
        print("="*60 + "\n")
        
        # Plot FROC curve
        plot_path = output_dir / 'froc_curve.png'
        plot_froc_curve(froc_results, plot_path, title=args.title)
        
        print("\n" + "="*60)
        print("INTERPRETATION GUIDE:")
        print("="*60)
        print("- The FROC curve shows the trade-off between sensitivity")
        print("  (true positive rate) and false positives per image (FPI)")
        print("- For TB detection, sensitivity at FPI≤2 is the key metric")
        print("- Higher sensitivity at low FPI is better")
        print("- Marked points show sensitivity at clinically relevant FPI values")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR during FROC computation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
