"""
Compute per-category FROC metrics for tuberculosis detection.
This computes sensitivity at various FPI thresholds for each TB category separately.
"""

import json
import numpy as np
from pycocotools.coco import COCO
import argparse
from collections import defaultdict


def compute_froc_curve_data(coco_gt, dt_results, num_images, iou_threshold=0.5):
    """
    Compute FROC curve data (FPs per image vs sensitivity).
    
    Args:
        coco_gt: COCO ground truth object
        dt_results: List of detection results in COCO format
        num_images: Total number of images
        iou_threshold: IoU threshold for matching
    
    Returns:
        fps_per_image, sensitivities, thresholds
    """
    # Load detections as COCO object
    if len(dt_results) == 0:
        return np.array([0]), np.array([0]), np.array([0])
    
    coco_dt = coco_gt.loadRes(dt_results)
    
    # Get all image IDs
    img_ids = sorted(coco_gt.getImgIds())
    num_gt_total = len(coco_gt.getAnnIds())
    
    # Collect all detections with scores
    all_detections = []
    for det in dt_results:
        all_detections.append({
            'image_id': det['image_id'],
            'bbox': det['bbox'],  # [x, y, w, h]
            'score': det['score'],
            'category_id': det['category_id']
        })
    
    # Sort by score descending
    all_detections.sort(key=lambda x: x['score'], reverse=True)
    
    # Track matches
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    scores = np.array([d['score'] for d in all_detections])
    
    gt_matched = {img_id: set() for img_id in img_ids}
    
    # Process each detection
    for det_idx, det in enumerate(all_detections):
        img_id = det['image_id']
        det_bbox = det['bbox']  # [x, y, w, h]
        det_cat = det['category_id']
        
        # Get GT for this image and category
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[det_cat])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        
        if len(gt_anns) == 0:
            fp[det_idx] = 1
            continue
        
        # Compute IoU with all GT boxes
        max_iou = 0
        max_gt_idx = -1
        
        det_x1, det_y1 = det_bbox[0], det_bbox[1]
        det_x2, det_y2 = det_bbox[0] + det_bbox[2], det_bbox[1] + det_bbox[3]
        det_area = det_bbox[2] * det_bbox[3]
        
        for gt_idx, gt_ann in enumerate(gt_anns):
            gt_bbox = gt_ann['bbox']
            gt_x1, gt_y1 = gt_bbox[0], gt_bbox[1]
            gt_x2, gt_y2 = gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]
            gt_area = gt_bbox[2] * gt_bbox[3]
            
            # Compute intersection
            ix1 = max(det_x1, gt_x1)
            iy1 = max(det_y1, gt_y1)
            ix2 = min(det_x2, gt_x2)
            iy2 = min(det_y2, gt_y2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = det_area + gt_area - inter
                iou = inter / union
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_ann['id']
        
        # Check if matched
        if max_iou >= iou_threshold:
            if max_gt_idx not in gt_matched[img_id]:
                tp[det_idx] = 1
                gt_matched[img_id].add(max_gt_idx)
            else:
                fp[det_idx] = 1
        else:
            fp[det_idx] = 1
    
    # Compute cumulative
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute sensitivity and FPI
    sensitivities = tp_cumsum / max(num_gt_total, 1)
    fps_per_image = fp_cumsum / num_images
    
    return fps_per_image, sensitivities, scores


def compute_per_category_froc(gt_anno_file, dt_results_file, max_fpi=2.0, output_file=None):
    """
    Compute per-category FROC metrics.
    
    Args:
        gt_anno_file: Path to ground truth annotations (COCO format)
        dt_results_file: Path to detection results (COCO format)
        max_fpi: Maximum false positives per image to evaluate
        output_file: Optional path to save results JSON
    """
    # Load ground truth and detections
    coco_gt = COCO(gt_anno_file)
    with open(dt_results_file, 'r') as f:
        dt_results = json.load(f)
    
    # Get all categories
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    
    print(f"\nFound {len(cats)} categories:")
    for cat in cats:
        print(f"  - ID {cat['id']}: {cat['name']}")
    
    # Standard clinical FPI points
    fpi_points = [0.125, 0.25, 0.5, 1.0, 2.0]
    if max_fpi > 2.0:
        fpi_points.extend([4.0, 8.0])
    
    print(f"\nComputing FROC at FPI points: {fpi_points}")
    print(f"Maximum FPI: {max_fpi}")
    
    # Get all image IDs
    all_img_ids = coco_gt.getImgIds()
    num_images = len(all_img_ids)
    print(f"\nTotal images: {num_images}")
    
    # Compute overall FROC first
    print("\n" + "="*80)
    print("OVERALL FROC (all categories)")
    print("="*80)
    
    overall_fps, overall_sens, overall_thresholds = compute_froc_curve_data(
        coco_gt, dt_results, num_images
    )
    
    # Compute sensitivity at FPI points
    overall_sens_at_fpi = {}
    for fpi in fpi_points:
        idx = np.where(overall_fps <= fpi)[0]
        if len(idx) > 0:
            sens = overall_sens[idx[-1]]
            overall_sens_at_fpi[fpi] = sens
            print(f"  Sensitivity @ FPI ≤ {fpi:5.3f}: {sens*100:6.2f}%")
        else:
            overall_sens_at_fpi[fpi] = 0.0
            print(f"  Sensitivity @ FPI ≤ {fpi:5.3f}:   0.00%")
    
    # Compute per-category FROC
    per_category_results = {}
    
    for cat in cats:
        cat_id = cat['id']
        cat_name = cat['name']
        
        print("\n" + "="*80)
        print(f"CATEGORY: {cat_name} (ID: {cat_id})")
        print("="*80)
        
        # Get annotations for this category
        cat_ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        cat_anns = coco_gt.loadAnns(cat_ann_ids)
        
        if len(cat_anns) == 0:
            print(f"No annotations found for category {cat_name}. Skipping.")
            per_category_results[cat_name] = {
                'category_id': cat_id,
                'num_annotations': 0,
                'num_images': 0,
                'sensitivity_at_fpi': {str(fpi): 0.0 for fpi in fpi_points}
            }
            continue
        
        # Get images with this category
        cat_img_ids = list(set([ann['image_id'] for ann in cat_anns]))
        print(f"Number of ground truth annotations: {len(cat_anns)}")
        print(f"Number of images with this category: {len(cat_img_ids)}")
        
        # Filter detections to only this category
        cat_dt_results = [det for det in dt_results if det['category_id'] == cat_id]
        print(f"Number of detections for this category: {len(cat_dt_results)}")
        
        # Create category-specific COCO GT (only this category)
        cat_coco_gt_dict = {
            'images': coco_gt.dataset['images'],
            'annotations': cat_anns,
            'categories': [cat],
            'info': coco_gt.dataset.get('info', {}),
            'licenses': coco_gt.dataset.get('licenses', [])
        }
        cat_coco_gt = COCO()
        cat_coco_gt.dataset = cat_coco_gt_dict
        cat_coco_gt.createIndex()
        
        # Compute FROC for this category
        cat_fps, cat_sens, cat_thresholds = compute_froc_curve_data(
            cat_coco_gt, cat_dt_results, num_images
        )
        
        # Compute sensitivity at FPI points
        cat_sens_at_fpi = {}
        for fpi in fpi_points:
            idx = np.where(cat_fps <= fpi)[0]
            if len(idx) > 0:
                sens = cat_sens[idx[-1]]
                cat_sens_at_fpi[fpi] = sens
                print(f"  Sensitivity @ FPI ≤ {fpi:5.3f}: {sens*100:6.2f}%")
            else:
                cat_sens_at_fpi[fpi] = 0.0
                print(f"  Sensitivity @ FPI ≤ {fpi:5.3f}:   0.00%")
        
        per_category_results[cat_name] = {
            'category_id': cat_id,
            'num_annotations': len(cat_anns),
            'num_images': len(cat_img_ids),
            'num_detections': len(cat_dt_results),
            'sensitivity_at_fpi': {str(fpi): float(cat_sens_at_fpi[fpi]) for fpi in fpi_points},
            'full_curve': {
                'fps': cat_fps.tolist() if isinstance(cat_fps, np.ndarray) else cat_fps,
                'sensitivity': cat_sens.tolist() if isinstance(cat_sens, np.ndarray) else cat_sens,
                'thresholds': cat_thresholds.tolist() if isinstance(cat_thresholds, np.ndarray) else cat_thresholds
            }
        }
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    header = f"{'Category':<35}"
    for fpi in fpi_points:
        header += f" Sens@{fpi:<4}"
    header += " #Anns #Imgs"
    print(header)
    print("-"*80)
    
    # Overall row
    overall_row = f"{'OVERALL':<35}"
    for fpi in fpi_points:
        overall_row += f" {overall_sens_at_fpi[fpi]*100:>6.2f}%"
    overall_row += f" {'N/A':<5} {'N/A':<5}"
    print(overall_row)
    
    # Per-category rows
    for cat_name, results in per_category_results.items():
        if results['num_annotations'] == 0:
            continue
        row = f"{cat_name:<35}"
        for fpi in fpi_points:
            sens = results['sensitivity_at_fpi'][str(fpi)]
            row += f" {sens*100:>6.2f}%"
        row += f" {results['num_annotations']:<5} {results['num_images']:<5}"
        print(row)
    
    # Save results to JSON
    results_dict = {
        'max_fpi': max_fpi,
        'fpi_points': fpi_points,
        'num_images': num_images,
        'overall': {
            'sensitivity_at_fpi': {str(fpi): float(overall_sens_at_fpi[fpi]) for fpi in fpi_points},
            'full_curve': {
                'fps': overall_fps.tolist() if isinstance(overall_fps, np.ndarray) else overall_fps,
                'sensitivity': overall_sens.tolist() if isinstance(overall_sens, np.ndarray) else overall_sens,
                'thresholds': overall_thresholds.tolist() if isinstance(overall_thresholds, np.ndarray) else overall_thresholds
            }
        },
        'per_category': per_category_results
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute per-category FROC metrics')
    parser.add_argument('--gt-anno', type=str, required=True,
                        help='Path to ground truth annotations (COCO JSON format)')
    parser.add_argument('--dt-results', type=str, required=True,
                        help='Path to detection results (COCO JSON format)')
    parser.add_argument('--max-fpi', type=float, default=2.0,
                        help='Maximum false positives per image (default: 2.0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    
    args = parser.parse_args()
    
    compute_per_category_froc(args.gt_anno, args.dt_results, args.max_fpi, args.output)

