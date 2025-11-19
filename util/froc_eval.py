"""
FROC (Free-Response Receiver Operating Characteristic) evaluation for TBX11K
Computes sensitivity vs false positives per image (FPI) for tuberculosis detection
"""

import numpy as np
from typing import List, Dict, Tuple
import torch


def compute_froc(
    all_boxes: List[np.ndarray],
    all_scores: List[np.ndarray],
    all_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    iou_threshold: float = 0.5,
    fp_per_image_range: List[float] = [0.125, 0.25, 0.5, 1, 2, 4, 8]
) -> Dict[str, float]:
    """
    Compute FROC curve and metrics for object detection.
    
    Args:
        all_boxes: List of predicted boxes per image, each shape [N, 4] (x1, y1, x2, y2)
        all_scores: List of confidence scores per image, each shape [N]
        all_labels: List of predicted labels per image, each shape [N]
        gt_boxes: List of ground truth boxes per image, each shape [M, 4]
        gt_labels: List of ground truth labels per image, each shape [M]
        iou_threshold: IoU threshold for considering a detection as correct
        fp_per_image_range: FPI values to evaluate at
    
    Returns:
        Dictionary containing:
            - 'sensitivities_at_fpi': List of sensitivities at each FPI value
            - 'mean_sensitivity': Average sensitivity across FPI values
            - 'fpi_values': List of FPI values used
            - 'sensitivity_at_2fpi': Sensitivity at 2 FPI (key metric for TBX11K)
    """
    num_images = len(all_boxes)
    num_gt_total = sum(len(gt) for gt in gt_boxes)
    
    # Collect all detections with image index
    all_detections = []
    for img_idx in range(num_images):
        boxes = all_boxes[img_idx]
        scores = all_scores[img_idx]
        labels = all_labels[img_idx]
        
        for box, score, label in zip(boxes, scores, labels):
            all_detections.append({
                'image_id': img_idx,
                'bbox': box,
                'score': score,
                'label': label,
                'matched': False
            })
    
    # Sort detections by confidence score (descending)
    all_detections.sort(key=lambda x: x['score'], reverse=True)
    
    # Initialize arrays to track TP and FP
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in gt_boxes]
    
    # Process each detection in order of confidence
    for det_idx, detection in enumerate(all_detections):
        img_idx = detection['image_id']
        det_box = detection['bbox']
        det_label = detection['label']
        
        gt_box_img = gt_boxes[img_idx]
        gt_label_img = gt_labels[img_idx]
        
        if len(gt_box_img) == 0:
            fp[det_idx] = 1
            continue
        
        # Compute IoU with all ground truth boxes in this image
        ious = compute_iou(det_box.reshape(1, 4), gt_box_img)
        
        # Find best matching GT box
        max_iou_idx = ious.argmax()
        max_iou = ious[0, max_iou_idx]
        
        # Check if detection matches a GT box
        # Convert to scalar for comparison
        gt_label_at_idx = int(gt_label_img[max_iou_idx])
        det_label_int = int(det_label)
        
        if max_iou >= iou_threshold and det_label_int == gt_label_at_idx:
            if not gt_matched[img_idx][max_iou_idx]:
                # True positive
                tp[det_idx] = 1
                gt_matched[img_idx][max_iou_idx] = True
            else:
                # GT already matched, count as FP
                fp[det_idx] = 1
        else:
            # False positive
            fp[det_idx] = 1
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute sensitivity (recall) = TP / total_GT
    sensitivities = tp_cumsum / max(num_gt_total, 1)
    
    # Compute FP per image
    fp_per_image = fp_cumsum / num_images
    
    # Compute sensitivity at specified FPI values
    sensitivities_at_fpi = []
    for target_fpi in fp_per_image_range:
        # Find the detection index where FPI is closest to target
        idx = np.searchsorted(fp_per_image, target_fpi, side='left')
        if idx < len(sensitivities):
            sensitivities_at_fpi.append(sensitivities[idx])
        else:
            sensitivities_at_fpi.append(sensitivities[-1] if len(sensitivities) > 0 else 0.0)
    
    # Compute mean sensitivity
    mean_sensitivity = np.mean(sensitivities_at_fpi)
    
    # Get sensitivity at 2 FPI specifically (important for TBX11K)
    if 2 in fp_per_image_range:
        fpi_2_idx = fp_per_image_range.index(2)
        sensitivity_at_2fpi = sensitivities_at_fpi[fpi_2_idx]
    else:
        # Interpolate
        idx = np.searchsorted(fp_per_image, 2.0, side='left')
        sensitivity_at_2fpi = sensitivities[idx] if idx < len(sensitivities) else sensitivities[-1]
    
    results = {
        'sensitivities_at_fpi': sensitivities_at_fpi,
        'mean_sensitivity': mean_sensitivity,
        'fpi_values': fp_per_image_range,
        'sensitivity_at_2fpi': sensitivity_at_2fpi,
        'froc_curve': {
            'fp_per_image': fp_per_image.tolist(),
            'sensitivity': sensitivities.tolist()
        }
    }
    
    return results


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Array of shape [N, 4] in format (x1, y1, x2, y2)
        boxes2: Array of shape [M, 4] in format (x1, y1, x2, y2)
    
    Returns:
        IoU matrix of shape [N, M]
    """
    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersection
    x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute union
    union = area1[:, np.newaxis] + area2 - intersection
    
    # Compute IoU
    iou = intersection / np.maximum(union, 1e-10)
    
    return iou


def compute_froc_from_coco_results(
    coco_gt,
    coco_dt,
    iou_threshold: float = 0.5,
    fp_per_image_range: List[float] = [0.125, 0.25, 0.5, 1, 2, 4, 8]
) -> Dict[str, float]:
    """
    Compute FROC metrics from COCO format ground truth and detections.
    
    Args:
        coco_gt: COCO ground truth object
        coco_dt: COCO detections object
        iou_threshold: IoU threshold for matching
        fp_per_image_range: FPI values to evaluate at
    
    Returns:
        Dictionary with FROC metrics
    """
    img_ids = sorted(coco_gt.getImgIds())
    num_images = len(img_ids)
    
    all_boxes = []
    all_scores = []
    all_labels = []
    gt_boxes = []
    gt_labels = []
    
    for img_id in img_ids:
        # Get predictions
        det_ids = coco_dt.getAnnIds(imgIds=[img_id])
        dets = coco_dt.loadAnns(det_ids)
        
        if len(dets) > 0:
            boxes = np.array([det['bbox'] for det in dets])  # [x, y, w, h]
            # Convert to [x1, y1, x2, y2]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            scores = np.array([det['score'] for det in dets])
            labels = np.array([det['category_id'] for det in dets])
        else:
            boxes = np.zeros((0, 4))
            scores = np.zeros(0)
            labels = np.zeros(0)
        
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        
        # Get ground truth
        gt_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gts = coco_gt.loadAnns(gt_ids)
        
        if len(gts) > 0:
            gt_box = np.array([gt['bbox'] for gt in gts])  # [x, y, w, h]
            # Convert to [x1, y1, x2, y2]
            gt_box[:, 2] += gt_box[:, 0]
            gt_box[:, 3] += gt_box[:, 1]
            gt_label = np.array([gt['category_id'] for gt in gts])
        else:
            gt_box = np.zeros((0, 4))
            gt_label = np.zeros(0)
        
        gt_boxes.append(gt_box)
        gt_labels.append(gt_label)
    
    return compute_froc(all_boxes, all_scores, all_labels, gt_boxes, gt_labels, 
                       iou_threshold, fp_per_image_range)


def print_froc_results(froc_results: Dict[str, float]) -> None:
    """Print FROC results in a formatted way."""
    print("\n" + "="*60)
    print("FROC Evaluation Results")
    print("="*60)
    
    fpi_values = froc_results['fpi_values']
    sensitivities = froc_results['sensitivities_at_fpi']
    
    print("\nSensitivity at different FPI values:")
    print("-" * 40)
    for fpi, sens in zip(fpi_values, sensitivities):
        print(f"  FPI = {fpi:5.3f}:  Sensitivity = {sens:.4f}")
    
    print("\n" + "-" * 40)
    print(f"Mean Sensitivity: {froc_results['mean_sensitivity']:.4f}")
    print(f"Sensitivity at 2 FPI: {froc_results['sensitivity_at_2fpi']:.4f}")
    print("="*60 + "\n")
