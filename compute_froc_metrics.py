"""
Standalone script to compute FROC metrics from COCO evaluation results
Usage: python compute_froc_metrics.py --results_file results.pkl --output_dir eval_output
"""

import argparse
import pickle
import json
from pathlib import Path
from util.froc_eval import compute_froc_from_coco_results, print_froc_results
from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser(description='Compute FROC metrics for TBX11K evaluation')
    parser.add_argument('--coco_path', type=str, required=True,
                       help='Path to TBX11K dataset root')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to detection results JSON file (COCO format)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for FROC results')
    parser.add_argument('--ann_file', type=str, default='annotations/instances_val2017.json',
                       help='Annotation file relative to coco_path')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching predictions to ground truth')
    args = parser.parse_args()
    
    # Load ground truth
    ann_file = Path(args.coco_path) / args.ann_file
    print(f"Loading ground truth from: {ann_file}")
    coco_gt = COCO(str(ann_file))
    
    # Load detection results
    print(f"Loading detection results from: {args.results_file}")
    coco_dt = coco_gt.loadRes(args.results_file)
    
    # Compute FROC metrics
    print("\nComputing FROC metrics...")
    froc_results = compute_froc_from_coco_results(
        coco_gt, 
        coco_dt,
        iou_threshold=args.iou_threshold,
        fp_per_image_range=[0.125, 0.25, 0.5, 1, 2, 4, 8]
    )
    
    # Print results
    print_froc_results(froc_results)
    
    # Save results to JSON
    output_path = Path(args.output_dir) / 'froc_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(froc_results, f, indent=2)
    
    print(f"\nFROC results saved to: {output_path}")
    
    # Print key metric for TBX11K paper
    print("\n" + "="*60)
    print("KEY METRIC FOR TBX11K PAPER:")
    print(f"Sensitivity at FPI < 2: {froc_results['sensitivity_at_2fpi']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
