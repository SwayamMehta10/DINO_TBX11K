#!/bin/bash
#SBATCH -J dino_tbx11k_eval
#SBATCH -p public
#SBATCH -q class
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_tbx11k_%j.out
#SBATCH --error=logs/eval_tbx11k_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

# ============================================================================
# DINO Evaluation Script for TBX11K on ASU Sol Supercomputer
# ============================================================================
# This script evaluates trained DINO model on TBX11K validation set
# Computes standard COCO metrics (AP, AR) and FROC score (sensitivity at FPI<2)
# ============================================================================

echo "================================================================"
echo "DINO Evaluation on TBX11K Dataset"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

# Load required modules
module purge
module load gcc-12.1.0-gcc-11.2.0
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest

# Activate conda environment
source activate DINO_ENV

# Set paths - MODIFY THESE
export PROJECT_ROOT=$SLURM_SUBMIT_DIR
export DATA_PATH=${PROJECT_ROOT}/TBX11K
export OUTPUT_DIR=${PROJECT_ROOT}/outputs/evaluation_${SLURM_JOB_ID}
export CONFIG_FILE=${PROJECT_ROOT}/config/DINO/DINO_4scale_tbx11k.py

# Checkpoint to evaluate - MODIFY THIS
# Use either checkpoint_best_regular.pth or checkpoint_best_ema.pth
export CHECKPOINT_PATH=${PROJECT_ROOT}/outputs/tbx11k_run_39722751/checkpoint_best_regular.pth

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROJECT_ROOT}/logs

# Navigate to project directory
cd ${PROJECT_ROOT}

echo "Project Root: ${PROJECT_ROOT}"
echo "Data Path: ${DATA_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Config File: ${CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "================================================================"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

echo "================================================================"
echo "Starting Evaluation..."
echo "================================================================"

# Run evaluation (disable distributed mode for single GPU)
# Unset SLURM distributed environment variables to force single-GPU mode
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NTASKS
unset WORLD_SIZE
unset LOCAL_RANK
unset RANK

python main.py \
    --config_file ${CONFIG_FILE} \
    --coco_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --eval \
    --resume ${CHECKPOINT_PATH} \
    --world_size 1 \
    --options \
        num_classes=4 \
        dn_labelbook_size=4 \
    --save_results

# Check evaluation exit status
if [ $? -eq 0 ]; then
    echo "================================================================"
    echo "Evaluation completed successfully!"
    echo "================================================================"
    
    # Parse and display COCO metrics in readable format
    if [ -f "${OUTPUT_DIR}/log.txt" ]; then
        echo ""
        echo "================================================================"
        echo "COCO EVALUATION METRICS"
        echo "================================================================"
        
        # Extract the test_coco_eval_bbox array from log.txt
        python3 << EOF
import json
import sys
import os

output_dir = os.environ.get('OUTPUT_DIR', '.')
try:
    with open(f"{output_dir}/log.txt", "r") as f:
        log_data = json.load(f)
    
    if "test_coco_eval_bbox" in log_data:
        stats = log_data["test_coco_eval_bbox"]
        
        print("\nBounding Box Detection Metrics:")
        print("-" * 60)
        print(f"AP @ IoU=0.50:0.95 (primary metric): {stats[0]:.3f}")
        print(f"AP @ IoU=0.50:              {stats[1]:.3f}")
        print(f"AP @ IoU=0.75:              {stats[2]:.3f}")
        print(f"AP @ small:                 {stats[3]:.3f}")
        print(f"AP @ medium:                {stats[4]:.3f}")
        print(f"AP @ large:                 {stats[5]:.3f}")
        print("-" * 60)
        print(f"AR @ maxDets=1:             {stats[6]:.3f}")
        print(f"AR @ maxDets=10:            {stats[7]:.3f}")
        print(f"AR @ maxDets=100:           {stats[8]:.3f}")
        print(f"AR @ small:                 {stats[9]:.3f}")
        print(f"AR @ medium:                {stats[10]:.3f}")
        print(f"AR @ large:                 {stats[11]:.3f}")
        print("-" * 60)
    else:
        print("No COCO evaluation metrics found in log.txt")
        
except Exception as e:
    print(f"Error parsing metrics: {e}")
    print("\nRaw log content:")
    try:
        with open(f"{output_dir}/log.txt", "r") as f:
            print(f.read())
    except FileNotFoundError:
        print(f"Log file not found at {output_dir}/log.txt")
EOF
    fi
else
    echo "================================================================"
    echo "Evaluation failed with error code $?"
    echo "================================================================"
    exit 1
fi

echo ""
echo "================================================================"
echo "COMPUTING FROC METRICS"
echo "================================================================"

# FROC computation uses results extracted from COCO evaluator
JSON_FILE="${OUTPUT_DIR}/results0.json"

if [ -f "${JSON_FILE}" ]; then
    echo "Detection results found, computing FROC..."
    
    python compute_froc_metrics.py \
        --coco_path ${DATA_PATH} \
        --results_file ${JSON_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --ann_file annotations/json/TBX11K_val.json \
        --iou_threshold 0.5 \
        --max_fpi 2.0
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "FROC metrics computed successfully!"
        if [ -f "${OUTPUT_DIR}/froc_results.json" ]; then
            echo "FROC results saved to: ${OUTPUT_DIR}/froc_results.json"
        fi
    else
        echo "Warning: FROC computation failed"
    fi
else
    echo "Warning: Detection results file not found at ${JSON_FILE}"
    echo "Results are saved after COCO evaluation completes"
fi

echo ""
echo "End Time: $(date)"
echo "================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs saved to: ${PROJECT_ROOT}/logs/eval_tbx11k_${SLURM_JOB_ID}.out"
echo "================================================================"
