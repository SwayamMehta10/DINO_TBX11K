# ============================================================================
# Plot FROC Curve from Previous Evaluation (Job 39748262) - Windows Version
# ============================================================================
# This script generates FROC curve visualization from existing detection results
# Uses results from evaluation run before TB-only filtering implementation
# ============================================================================

$PROJECT_ROOT = "D:\GitHub\DINO_TBX11K"
$DATA_PATH = "$PROJECT_ROOT\TBX11K"
$EVAL_DIR = "$PROJECT_ROOT\outputs\evaluation_39748262"
$RESULTS_FILE = "$EVAL_DIR\results0.json"

# Check if results file exists
if (-not (Test-Path $RESULTS_FILE)) {
    Write-Host "ERROR: Detection results file not found at $RESULTS_FILE" -ForegroundColor Red
    Write-Host "Expected results from evaluation job 39748262"
    exit 1
}

Write-Host "================================================================"
Write-Host "Plotting FROC Curve from Previous Evaluation"
Write-Host "================================================================"
Write-Host "Data Path: $DATA_PATH"
Write-Host "Results File: $RESULTS_FILE"
Write-Host "Output Directory: $EVAL_DIR"
Write-Host "================================================================"

# Run FROC plotting with standard FPI points
# Paper emphasizes FPI≤2, but we compute full curve for context
python "$PROJECT_ROOT\plot_froc_curve.py" `
    --coco_path "$DATA_PATH" `
    --results_file "$RESULTS_FILE" `
    --output_dir "$EVAL_DIR" `
    --ann_file "annotations/json/TBX11K_val.json" `
    --iou_threshold 0.5 `
    --fpi_points 0.125 0.25 0.5 1.0 2.0 4.0 8.0 `
    --title "FROC Curve - Previous Evaluation (All Validation Images)"

Write-Host ""
Write-Host "================================================================"
Write-Host "FROC curve generated!"
Write-Host "================================================================"
Write-Host "Plot saved to: $EVAL_DIR\froc_curve.png"
Write-Host "Plot (PDF) saved to: $EVAL_DIR\froc_curve.pdf"
Write-Host "Metrics saved to: $EVAL_DIR\froc_results.json"
Write-Host "================================================================"
