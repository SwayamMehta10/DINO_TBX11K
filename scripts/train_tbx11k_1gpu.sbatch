#!/bin/bash
#SBATCH -J dino_1gpu_train
#SBATCH -p public
#SBATCH -q class
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_tbx11k_%j.out
#SBATCH --error=logs/train_tbx11k_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

# ============================================================================
# DINO Training Script for TBX11K - 1 GPU VERSION (FASTER QUEUE)
# ============================================================================
# This script trains DINO with 1 GPU but simulates batch_size=2 using
# gradient accumulation (2 forward passes before optimizer step)
# 
# RESOURCE USAGE: 1 GPU × 8 hours = 480 GPU-minutes (50% of budget)
# QUEUE TIME: Much faster allocation than 2 GPU jobs
# PERFORMANCE: Same as 2 GPU training (effective batch_size=2)
# TRADE-OFF: Slightly slower per epoch (~1.2x) but starts immediately
# ============================================================================

# Set environment for single-GPU training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=0

# Load required modules
module purge
module load gcc-12.1.0-gcc-11.2.0
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate DINO_ENV

# Set NCCL settings
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

echo "================================================================"
echo "DINO Training on TBX11K Dataset - 1 GPU with Gradient Accumulation"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "GPUs: 1 × A100 (simulating effective batch_size=2)"
echo "================================================================"

# Set paths
export PROJECT_ROOT=$SLURM_SUBMIT_DIR
export DATA_PATH=${PROJECT_ROOT}/TBX11K
export OUTPUT_DIR=${PROJECT_ROOT}/outputs/tbx11k_run_${SLURM_JOB_ID}
export CONFIG_FILE=${PROJECT_ROOT}/config/DINO/DINO_4scale_tbx11k_1gpu.py

# Find most recent checkpoint from 1-GPU training sessions
# We look for checkpoints that have gradient_accumulation_steps in their config
# This ensures we only resume from compatible checkpoints
export PREV_CHECKPOINT_DIR=""
for dir in $(ls -td ${PROJECT_ROOT}/outputs/tbx11k_run_* 2>/dev/null | grep -v "tbx11k_run_${SLURM_JOB_ID}"); do
    # Check if this checkpoint has gradient_accumulation_steps in config (1-GPU training)
    if [ -f "${dir}/config_args_all.json" ] && grep -q "gradient_accumulation_steps" "${dir}/config_args_all.json" 2>/dev/null; then
        if [ -f "${dir}/checkpoint.pth" ]; then
            export PREV_CHECKPOINT_DIR="${dir}"
            echo "Found compatible 1-GPU checkpoint: ${dir}"
            break
        fi
    fi
done

# If no compatible checkpoint found, will start fresh with pretrained weights

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROJECT_ROOT}/logs

cd ${PROJECT_ROOT}

echo "Project Root: ${PROJECT_ROOT}"
echo "Data Path: ${DATA_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Config File: ${CONFIG_FILE}"
if [ ! -z "${PREV_CHECKPOINT_DIR}" ]; then
    echo "Previous Checkpoint Directory: ${PREV_CHECKPOINT_DIR}"
fi
echo "================================================================"

# Check for checkpoint resume
if [ ! -z "${PREV_CHECKPOINT_DIR}" ] && [ -f "${PREV_CHECKPOINT_DIR}/checkpoint.pth" ]; then
    echo "Resuming from previous job checkpoint: ${PREV_CHECKPOINT_DIR}/checkpoint.pth"
    RESUME_FLAG="--resume ${PREV_CHECKPOINT_DIR}/checkpoint.pth"
    cp -r ${PREV_CHECKPOINT_DIR}/* ${OUTPUT_DIR}/
elif [ -f "${OUTPUT_DIR}/checkpoint.pth" ]; then
    echo "Found checkpoint in current output directory, will resume training..."
    RESUME_FLAG="--resume ${OUTPUT_DIR}/checkpoint.pth"
else
    echo "No checkpoint found, starting fresh training..."
    RESUME_FLAG=""
fi

# Pretrained model handling
PRETRAIN_MODEL="/scratch/$USER/dino_pretrained/checkpoint0011_4scale.pth"
if [ -z "${RESUME_FLAG}" ] && [ -f "${PRETRAIN_MODEL}" ]; then
    echo "Using pretrained weights: ${PRETRAIN_MODEL}"
    PRETRAIN_FLAG="--pretrain_model_path ${PRETRAIN_MODEL} --finetune_ignore label_enc class_embed dn_labelbook tgt_embed"
else
    if [ ! -z "${RESUME_FLAG}" ]; then
        echo "Resuming from checkpoint - pretrained weights not needed"
    else
        echo "No pretrained weights found, training from scratch"
    fi
    PRETRAIN_FLAG=""
fi

echo "================================================================"
echo "Starting Training with Gradient Accumulation..."
echo "Effective Batch Size: 2 (1 per GPU × 2 accumulation steps)"
echo "NOTE: Starting FRESH training (not resuming from old batch_size=1 checkpoint)"
echo "Old checkpoint was incompatible due to different training strategy"
echo "================================================================"

# Run training with single GPU
# Config uses batch_size=1 with gradient accumulation to simulate batch_size=2
python main.py \
    --config_file ${CONFIG_FILE} \
    --coco_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 4 \
    ${RESUME_FLAG} \
    ${PRETRAIN_FLAG}

# Check exit status
if [ $? -eq 0 ]; then
    echo "================================================================"
    echo "Training completed successfully!"
    echo "================================================================"
else
    echo "================================================================"
    echo "Training failed with error code $?"
    echo "================================================================"
    exit 1
fi

echo "End Time: $(date)"
echo "================================================================"
echo "Output saved to: ${OUTPUT_DIR}"
echo "Logs saved to: ${PROJECT_ROOT}/logs/train_tbx11k_${SLURM_JOB_ID}.out"
echo "================================================================"
echo ""
echo "NEXT STEPS:"
echo "1. Check training metrics: grep 'Averaged stats' logs/train_tbx11k_${SLURM_JOB_ID}.out"
echo "2. To continue training, resubmit: sbatch scripts/train_tbx11k_1gpu.sbatch"
echo "3. Expected epochs per session: ~15-20 epochs"
echo "================================================================"
