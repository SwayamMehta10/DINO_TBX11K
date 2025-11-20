"""
Extract per-category metrics from COCO evaluation results.
This computes AP, AR for each category separately.
"""

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def compute_per_category_metrics(gt_anno_file, dt_results_file, output_file=None):
    """
    Compute per-category AP and AR metrics.
    
    Args:
        gt_anno_file: Path to ground truth annotations (COCO format)
        dt_results_file: Path to detection results (COCO format)
        output_file: Optional path to save results JSON
    """
    # Load ground truth and detections
    coco_gt = COCO(gt_anno_file)
    coco_dt = coco_gt.loadRes(dt_results_file)
    
    # Get all categories
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    
    print(f"\nFound {len(cats)} categories:")
    for cat in cats:
        print(f"  - ID {cat['id']}: {cat['name']}")
    
    # Compute overall metrics first
    print("\n" + "="*80)
    print("OVERALL METRICS (all categories)")
    print("="*80)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    overall_stats = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
        'AR_max1': coco_eval.stats[6],
        'AR_max10': coco_eval.stats[7],
        'AR_max100': coco_eval.stats[8],
        'AR_small': coco_eval.stats[9],
        'AR_medium': coco_eval.stats[10],
        'AR_large': coco_eval.stats[11],
    }
    
    # Compute per-category metrics
    per_category_results = {}
    
    for cat in cats:
        cat_id = cat['id']
        cat_name = cat['name']
        
        print("\n" + "="*80)
        print(f"CATEGORY: {cat_name} (ID: {cat_id})")
        print("="*80)
        
        # Get annotations for this category
        cat_anns = coco_gt.getAnnIds(catIds=[cat_id])
        print(f"Number of ground truth annotations: {len(cat_anns)}")
        
        # Get images with this category
        cat_img_ids = list(set([ann['image_id'] for ann in coco_gt.loadAnns(cat_anns)]))
        print(f"Number of images with this category: {len(cat_img_ids)}")
        
        # Evaluate only this category
        coco_eval_cat = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()
        coco_eval_cat.summarize()
        
        per_category_results[cat_name] = {
            'category_id': cat_id,
            'num_annotations': len(cat_anns),
            'num_images': len(cat_img_ids),
            'AP': coco_eval_cat.stats[0],
            'AP50': coco_eval_cat.stats[1],
            'AP75': coco_eval_cat.stats[2],
            'AP_small': coco_eval_cat.stats[3],
            'AP_medium': coco_eval_cat.stats[4],
            'AP_large': coco_eval_cat.stats[5],
            'AR_max1': coco_eval_cat.stats[6],
            'AR_max10': coco_eval_cat.stats[7],
            'AR_max100': coco_eval_cat.stats[8],
            'AR_small': coco_eval_cat.stats[9],
            'AR_medium': coco_eval_cat.stats[10],
            'AR_large': coco_eval_cat.stats[11],
        }
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Category':<30} {'AP':<8} {'AP50':<8} {'AP75':<8} {'AR@100':<8} {'#Anns':<8} {'#Imgs':<8}")
    print("-"*80)
    print(f"{'OVERALL':<30} {overall_stats['AP']*100:>6.2f}% {overall_stats['AP50']*100:>6.2f}% "
          f"{overall_stats['AP75']*100:>6.2f}% {overall_stats['AR_max100']*100:>6.2f}% {'N/A':<8} {'N/A':<8}")
    for cat_name, results in per_category_results.items():
        print(f"{cat_name:<30} {results['AP']*100:>6.2f}% {results['AP50']*100:>6.2f}% "
              f"{results['AP75']*100:>6.2f}% {results['AR_max100']*100:>6.2f}% "
              f"{results['num_annotations']:<8} {results['num_images']:<8}")
    
    # Save results to JSON
    results_dict = {
        'overall': overall_stats,
        'per_category': per_category_results
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute per-category COCO metrics')
    parser.add_argument('--gt-anno', type=str, required=True,
                        help='Path to ground truth annotations (COCO JSON format)')
    parser.add_argument('--dt-results', type=str, required=True,
                        help='Path to detection results (COCO JSON format)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    
    args = parser.parse_args()
    
    compute_per_category_metrics(args.gt_anno, args.dt_results, args.output)

