#!/bin/bash

# ============================================================================
# Plot FROC Curve from Previous Evaluation (Job 39748262)
# ============================================================================
# This script generates FROC curve visualization from existing detection results
# Uses results from evaluation run before TB-only filtering implementation
# ============================================================================

# Set paths
PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
DATA_PATH="${PROJECT_ROOT}/TBX11K"
EVAL_DIR="${PROJECT_ROOT}/outputs/evaluation_39748262"
RESULTS_FILE="${EVAL_DIR}/results0.json"

# Check if results file exists
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "ERROR: Detection results file not found at ${RESULTS_FILE}"
    echo "Expected results from evaluation job 39748262"
    exit 1
fi

echo "================================================================"
echo "Plotting FROC Curve from Previous Evaluation"
echo "================================================================"
echo "Data Path: ${DATA_PATH}"
echo "Results File: ${RESULTS_FILE}"
echo "Output Directory: ${EVAL_DIR}"
echo "================================================================"

# Run FROC plotting with standard FPI points
# Paper emphasizes FPI≤2, but we compute full curve for context
python "${PROJECT_ROOT}/plot_froc_curve.py" \
    --coco_path "${DATA_PATH}" \
    --results_file "${RESULTS_FILE}" \
    --output_dir "${EVAL_DIR}" \
    --ann_file "annotations/json/TBX11K_val.json" \
    --iou_threshold 0.5 \
    --fpi_points 0.125 0.25 0.5 1.0 2.0 4.0 8.0 \
    --title "FROC Curve - Previous Evaluation (All Validation Images)"

echo ""
echo "================================================================"
echo "FROC curve generated!"
echo "================================================================"
echo "Plot saved to: ${EVAL_DIR}/froc_curve.png"
echo "Plot (PDF) saved to: ${EVAL_DIR}/froc_curve.pdf"
echo "Metrics saved to: ${EVAL_DIR}/froc_results.json"
echo "================================================================"
