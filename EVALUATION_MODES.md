# EVALUATION MODES - TBX11K Dataset

## Overview
The TBX11K paper describes two evaluation protocols:
1. **All X-rays**: Evaluate on all validation images (healthy + sick + TB)
2. **TB X-rays only**: Evaluate only on images containing TB annotations

This codebase now supports both modes via the `--tb_only_eval` flag.

## Quick Start

### Evaluate on All Images (Default)
```bash
# Original behavior - evaluates on all 1800 validation images
sbatch scripts/eval_tbx11k.sbatch
```

### Evaluate on TB-Only Images (Paper Protocol)
```bash
# Add --tb_only_eval flag in the script
# Already configured in updated eval_tbx11k.sbatch
```

## FROC Curve Interpretation

### What FPI Values Do You Need?

**Short Answer**: Yes, you need multiple FPI values (0.125, 0.25, 0.5, 1, 2, 4, 8) to plot a smooth FROC curve.

**Why**:
- FROC curve shows trade-off between sensitivity and false positives per image
- Multiple points are needed to visualize the full curve
- Paper emphasizes **sensitivity at FPI ≤ 2.0** as the primary metric
- But showing full curve (up to FPI=8) provides context

### Standard FPI Points

```python
# These are the standard FPI evaluation points
fpi_points = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# Paper's primary metric
paper_metric = "Sensitivity at FPI ≤ 2.0"
```

### Plotting FROC from Previous Evaluation

Your previous evaluation (Job 39748262) ran on **all 1800 validation images**. To plot the FROC curve:

#### On Windows (PowerShell):
```powershell
.\scripts\plot_froc_from_previous_eval.ps1
```

#### On Linux/Sol:
```bash
bash scripts/plot_froc_from_previous_eval.sh
```

This will generate:
- `outputs/evaluation_39748262/froc_curve.png` - High-res PNG plot
- `outputs/evaluation_39748262/froc_curve.pdf` - Publication-quality PDF
- `outputs/evaluation_39748262/froc_results.json` - Raw metrics

### Understanding FROC Results

The FROC curve visualization includes:
- **Full curve**: Continuous line showing sensitivity vs FPI trade-off
- **Marked points**: Specific FPI values (0.125, 0.25, 0.5, 1, 2, 4, 8) highlighted
- **Vertical line at FPI=2.0**: Paper's threshold for primary metric
- **Text box**: Key metrics (mean sensitivity, sensitivity @ FPI≤2)

### Comparison: Before vs After Implementation

| Aspect | Before (Job 39748262) | After (with --tb_only_eval) |
|--------|----------------------|----------------------------|
| Images evaluated | All 1800 val images | ~1200 TB-positive images |
| Includes healthy X-rays | Yes | No |
| Matches paper protocol | No (uses all data) | Yes (TB-only option) |
| FROC max FPI | 8.0 | 2.0 (paper metric) |

## Updated Scripts

### 1. Evaluation Script
**File**: `scripts/eval_tbx11k.sbatch`

**Changes**:
- Added `--tb_only_eval` flag to evaluation command
- Updated documentation to explain evaluation modes
- FROC computation already configured with `--max_fpi 2.0`

**Usage**:
```bash
# Submit updated evaluation job
sbatch scripts/eval_tbx11k.sbatch

# The script will automatically:
# 1. Filter to TB-only images
# 2. Compute COCO metrics (AP, AR)
# 3. Compute FROC metrics (sensitivity at various FPI)
# 4. Save results to outputs/evaluation_<jobid>/
```

### 2. Training Scripts
**Files**: 
- `scripts/train_tbx11k.sbatch` (2 GPUs)
- `scripts/train_tbx11k_1gpu.sbatch` (1 GPU)

**Status**: No changes needed - training uses full dataset (all images including healthy)

**Why**: Training on all images (healthy + TB) provides more diverse examples and improves model robustness. Only evaluation needs to be filtered to TB-only for precise TB detection metrics.

## Manual Evaluation Commands

### All Images (Default)
```bash
python main.py \
    --config_file config/DINO/DINO_4scale_tbx11k.py \
    --coco_path TBX11K \
    --output_dir outputs/evaluation_all \
    --eval \
    --resume outputs/tbx11k_run_39722751/checkpoint_best_regular.pth \
    --save_results
```

### TB-Only Images (Paper Protocol)
```bash
python main.py \
    --config_file config/DINO/DINO_4scale_tbx11k.py \
    --coco_path TBX11K \
    --output_dir outputs/evaluation_tb_only \
    --eval \
    --resume outputs/tbx11k_run_39722751/checkpoint_best_regular.pth \
    --save_results \
    --tb_only_eval  # ← Add this flag
```

### Compute FROC Metrics
```bash
python compute_froc_metrics.py \
    --coco_path TBX11K \
    --results_file outputs/evaluation_tb_only/results0.json \
    --output_dir outputs/evaluation_tb_only \
    --ann_file annotations/json/TBX11K_val.json \
    --iou_threshold 0.5 \
    --max_fpi 2.0
```

### Plot FROC Curve
```bash
python plot_froc_curve.py \
    --coco_path TBX11K \
    --results_file outputs/evaluation_tb_only/results0.json \
    --output_dir outputs/evaluation_tb_only \
    --ann_file annotations/json/TBX11K_val.json \
    --iou_threshold 0.5 \
    --fpi_points 0.125 0.25 0.5 1.0 2.0 4.0 8.0 \
    --title "FROC Curve - TB Detection (TB X-rays Only)"
```

## Key Metrics

### COCO Metrics (Standard Object Detection)
- **AP @ IoU=0.5:0.95**: Primary COCO metric (mean AP across IoU thresholds)
- **AP @ IoU=0.5**: Lenient metric (bounding box overlap ≥ 50%)
- **AR @ maxDets=100**: Average recall (max 100 detections per image)

### FROC Metrics (Medical Imaging)
- **Sensitivity @ FPI ≤ 2.0**: Paper's primary metric for TB detection
- **Mean Sensitivity**: Average sensitivity across all FPI thresholds
- **FROC Curve**: Full visualization of sensitivity vs FPI trade-off

## Expected Results

### Previous Evaluation (Job 39748262)
- **Dataset**: All 1800 validation images
- **AP @ IoU=0.5:0.95**: ~0.XXX (check log for exact value)
- **AP @ IoU=0.5**: ~0.XXX
- **FROC Sensitivity @ FPI≤2**: To be computed by plotting script

### TB-Only Evaluation (New)
- **Dataset**: ~1200 TB-positive images (exact count depends on annotations)
- **Expected**: Higher AP and sensitivity (no dilution from healthy images)
- **Use case**: Precise TB detection analysis as per paper protocol

## File Locations

```
outputs/
├── evaluation_39748262/          # Previous evaluation (all images)
│   ├── results0.json             # Detection results (COCO format)
│   ├── log.txt                   # COCO metrics
│   ├── froc_curve.png            # FROC visualization (after plotting)
│   ├── froc_curve.pdf            # Publication-quality plot
│   └── froc_results.json         # FROC metrics
│
└── evaluation_<new_jobid>/       # New TB-only evaluation
    ├── results0.json
    ├── log.txt
    ├── froc_curve.png
    └── froc_results.json
```

## Troubleshooting

### Issue: "No images found in val_tb_only"
**Cause**: Annotation file may not have images with TB labels
**Solution**: Verify `TBX11K/annotations/json/TBX11K_val.json` contains annotations for categories 1, 2, 3

### Issue: FROC metrics show 0 sensitivity
**Cause**: No true positive detections or incorrect IoU threshold
**Solution**: Check detection results and lower IoU threshold if needed

### Issue: Plot script fails with "results0.json not found"
**Cause**: Evaluation didn't save results
**Solution**: Ensure `--save_results` flag is used in evaluation command

## References

- **Paper**: TBX11K: A Large-scale Benchmark for Tuberculosis Detection
- **FROC Analysis**: FROC curve is standard for medical imaging CAD systems
- **Paper Metric**: Sensitivity at FPI ≤ 2.0 balances detection accuracy and false positive rate
