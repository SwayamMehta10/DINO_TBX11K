#!/bin/bash
#SBATCH -J dino_tbx11k_eval
#SBATCH -p public
#SBATCH -q class
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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
export CHECKPOINT_PATH=${PROJECT_ROOT}/outputs/tbx11k_run_39543743/checkpoint_best_regular.pth

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
python main.py \
    --config_file ${CONFIG_FILE} \
    --coco_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --eval \
    --resume ${CHECKPOINT_PATH} \
    --world_size 1 \
    --options \
        num_classes=4 \
        dn_labelbook_size=4

# Check evaluation exit status
if [ $? -eq 0 ]; then
    echo "================================================================"
    echo "Evaluation completed successfully!"
    echo "================================================================"
    
    # Display results if log file exists
    if [ -f "${OUTPUT_DIR}/log.txt" ]; then
        echo "Evaluation Results:"
        cat ${OUTPUT_DIR}/log.txt
    fi
else
    echo "================================================================"
    echo "Evaluation failed with error code $?"
    echo "================================================================"
    exit 1
fi

echo "End Time: $(date)"
echo "================================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs saved to: ${PROJECT_ROOT}/logs/eval_tbx11k_${SLURM_JOB_ID}.out"
echo "================================================================"

# Optional: Run FROC evaluation separately if needed
echo ""
echo "================================================================"
echo "Note: To compute FROC metrics, use the compute_froc.py script"
echo "with the evaluation results from ${OUTPUT_DIR}"
echo "================================================================"
