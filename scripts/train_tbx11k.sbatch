#!/bin/bash
#SBATCH -J dino_tbx11k_train
#SBATCH -p public
#SBATCH -q class
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_tbx11k_%j.out
#SBATCH --error=logs/train_tbx11k_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

# ============================================================================
# DINO Training Script for TBX11K on ASU Sol Supercomputer
# ============================================================================
# This script trains DINO object detector on TBX11K tuberculosis dataset
# Designed for 2x A100 GPUs with 8-hour walltime limit (480 GPU-minutes)
# Supports automatic checkpoint resume for multi-session training
# Resource-optimized: Leaves 480 GPU-minutes for other concurrent jobs
# ============================================================================

# Set Slurm environment variables for multi-GPU training
# These are required by util/misc.py init_distributed_mode()
# SLURM will automatically set these, but we ensure they're correct
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Note: SLURM_PROCID, SLURM_LOCALID, SLURM_NPROCS are automatically set by SLURM
# when using multiple GPUs with srun or the job environment

# Load required modules (adjust module names based on Sol configuration)
module purge
module load gcc-12.1.0-gcc-11.2.0
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
# Activate conda environment
source activate DINO_ENV

# Set NCCL settings for single-GPU distributed mode
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

echo "================================================================"
echo "DINO Training on TBX11K Dataset"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
echo "================================================================"

# Set paths
export PROJECT_ROOT=$SLURM_SUBMIT_DIR
export DATA_PATH=${PROJECT_ROOT}/TBX11K
export OUTPUT_DIR=${PROJECT_ROOT}/outputs/tbx11k_run_${SLURM_JOB_ID}
export CONFIG_FILE=${PROJECT_ROOT}/config/DINO/DINO_4scale_tbx11k.py

# Automatically find the most recent checkpoint directory
# This finds the latest tbx11k_run_* directory (excluding current job)
export PREV_CHECKPOINT_DIR=$(ls -td ${PROJECT_ROOT}/outputs/tbx11k_run_* 2>/dev/null | grep -v "tbx11k_run_${SLURM_JOB_ID}" | head -1)

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROJECT_ROOT}/logs

# Navigate to project directory
cd ${PROJECT_ROOT}

echo "Project Root: ${PROJECT_ROOT}"
echo "Data Path: ${DATA_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Config File: ${CONFIG_FILE}"
if [ ! -z "${PREV_CHECKPOINT_DIR}" ]; then
    echo "Previous Checkpoint Directory: ${PREV_CHECKPOINT_DIR}"
fi
echo "================================================================"

# Check if checkpoint exists for resume
# Priority 1: Check if resuming from previous job (when PREV_CHECKPOINT_DIR is set and checkpoint exists)
# Priority 2: Check current output directory (for same-job resume after restart)
if [ ! -z "${PREV_CHECKPOINT_DIR}" ] && [ -f "${PREV_CHECKPOINT_DIR}/checkpoint.pth" ]; then
    echo "Resuming from previous job checkpoint: ${PREV_CHECKPOINT_DIR}/checkpoint.pth"
    RESUME_FLAG="--resume ${PREV_CHECKPOINT_DIR}/checkpoint.pth"
    # Also copy previous checkpoints to new output directory to maintain history
    echo "Copying previous checkpoints to new output directory..."
    cp -r ${PREV_CHECKPOINT_DIR}/* ${OUTPUT_DIR}/
elif [ -f "${OUTPUT_DIR}/checkpoint.pth" ]; then
    echo "Found checkpoint in current output directory, will resume training..."
    RESUME_FLAG="--resume ${OUTPUT_DIR}/checkpoint.pth"
else
    echo "No checkpoint found, starting fresh training..."
    RESUME_FLAG=""
fi

# Optional: Download pretrained COCO weights if not exists
# Only use pretrained weights if NOT resuming from checkpoint
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
echo "Starting Training..."
echo "================================================================"

# Run training with 2 GPUs using PyTorch DistributedDataParallel
# Each GPU processes batch_size=1, effective batch_size=2
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    main.py \
    --config_file ${CONFIG_FILE} \
    --coco_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 4 \
    ${RESUME_FLAG} \
    ${PRETRAIN_FLAG}

# Check training exit status
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

# Optional: Copy important files to permanent storage
# cp ${OUTPUT_DIR}/checkpoint_best*.pth /scratch/$USER/dino_checkpoints/
