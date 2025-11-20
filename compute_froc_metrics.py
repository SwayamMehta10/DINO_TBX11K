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
    parser.add_argument('--max_fpi', type=float, default=2.0,
                       help='Maximum FPI to evaluate (default: 2.0 for paper compliance, use 8.0 for full range)')
    args = parser.parse_args()
    
    try:
        # Load ground truth
        ann_file = Path(args.coco_path) / args.ann_file
        print(f"Loading ground truth from: {ann_file}")
        coco_gt = COCO(str(ann_file))
        
        print(f"Ground truth has {len(coco_gt.getImgIds())} images")
        print(f"Ground truth has {len(coco_gt.getCatIds())} categories: {coco_gt.getCatIds()}")
        print(f"Sample image IDs: {coco_gt.getImgIds()[:5]}")
        
        # Load detection results
        print(f"\nLoading detection results from: {args.results_file}")
        
        # First check what's in the results file
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
        
        print(f"Results file has {len(results_data)} detections")
        if len(results_data) > 0:
            print(f"Sample detection: {results_data[0]}")
            
            # Check unique image IDs in results
            result_img_ids = set(det['image_id'] for det in results_data)
            print(f"Results have {len(result_img_ids)} unique image IDs")
            print(f"Sample result image IDs: {list(result_img_ids)[:5]}")
            
            # Check which image IDs don't match
            gt_img_ids = set(coco_gt.getImgIds())
            missing_in_gt = result_img_ids - gt_img_ids
            if missing_in_gt:
                print(f"WARNING: {len(missing_in_gt)} image IDs in results not in ground truth!")
                print(f"Sample missing IDs: {list(missing_in_gt)[:5]}")
            
            # Check category IDs
            result_cat_ids = set(det['category_id'] for det in results_data)
            print(f"Results have category IDs: {result_cat_ids}")
            gt_cat_ids = set(coco_gt.getCatIds())
            if not result_cat_ids.issubset(gt_cat_ids):
                print(f"WARNING: Some category IDs in results not in ground truth!")
                print(f"Invalid categories: {result_cat_ids - gt_cat_ids}")
        else:
            print("WARNING: Results file is empty!")
            return
        
        coco_dt = coco_gt.loadRes(args.results_file)
        print("Successfully loaded detection results into COCO format")
        
    except Exception as e:
        print(f"ERROR loading COCO data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Compute FROC metrics
        print(f"\nComputing FROC metrics (FPI range: 0 to {args.max_fpi})...")
        
        # Standard clinical FPI evaluation points for medical imaging
        # Paper reports sensitivity at FPI<=2.0, but compute full curve for visualization
        fp_per_image_range = [0.125, 0.25, 0.5, 1.0, 2.0]
        
        # Extend range if max_fpi > 2.0
        if args.max_fpi > 2.0:
            fp_per_image_range.extend([4.0, 8.0])
        
        print(f"Evaluating at FPI points: {fp_per_image_range}")
        
        froc_results = compute_froc_from_coco_results(
            coco_gt, 
            coco_dt,
            iou_threshold=args.iou_threshold,
            fp_per_image_range=fp_per_image_range
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
        print("KEY METRIC FOR TBX11K PAPER (TB Area Detection):")
        print(f"Sensitivity at FPI <= 2.0: {froc_results['sensitivity_at_2fpi']:.4f}")
        print("\nNote: This is the primary localization metric used in the paper.")
        print("The paper evaluates TB detection using only TB X-rays in the test set.")
        print("Use --tb_only_eval flag in main.py for TB-only evaluation.")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR during FROC computation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
