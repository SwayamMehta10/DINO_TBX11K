#!/bin/bash
#SBATCH -J dino_tbx11k_train
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
# DINO Training Script for TBX11K on ASU Sol Supercomputer
# ============================================================================
# This script trains DINO object detector on TBX11K tuberculosis dataset
# Designed for single A100 GPU with 8-hour walltime limit
# Supports automatic checkpoint resume for multi-session training
# ============================================================================

# Set Slurm environment variables for single-GPU training
# These are required by util/misc.py init_distributed_mode()
export SLURM_PROCID=0
export SLURM_LOCALID=0
export SLURM_NPROCS=1

# Set PyTorch distributed environment variables for single-GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500

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
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

# Set paths
export PROJECT_ROOT=$SLURM_SUBMIT_DIR
export DATA_PATH=${PROJECT_ROOT}/TBX11K
export OUTPUT_DIR=${PROJECT_ROOT}/outputs/tbx11k_run_${SLURM_JOB_ID}
export CONFIG_FILE=${PROJECT_ROOT}/config/DINO/DINO_4scale_tbx11k.py

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROJECT_ROOT}/logs

# Navigate to project directory
cd ${PROJECT_ROOT}

echo "Project Root: ${PROJECT_ROOT}"
echo "Data Path: ${DATA_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Config File: ${CONFIG_FILE}"
echo "================================================================"

# Check if checkpoint exists for resume
if [ -f "${OUTPUT_DIR}/checkpoint.pth" ]; then
    echo "Found existing checkpoint, will resume training..."
    RESUME_FLAG="--resume ${OUTPUT_DIR}/checkpoint.pth"
else
    echo "No checkpoint found, starting fresh training..."
    RESUME_FLAG=""
fi

# Optional: Download pretrained COCO weights if not exists
PRETRAIN_MODEL="/scratch/$USER/dino_pretrained/checkpoint0011_4scale.pth"
if [ -f "${PRETRAIN_MODEL}" ]; then
    echo "Using pretrained weights: ${PRETRAIN_MODEL}"
    PRETRAIN_FLAG="--pretrain_model_path ${PRETRAIN_MODEL} --finetune_ignore label_enc class_embed dn_labelbook"
else
    echo "No pretrained weights found, training from scratch"
    PRETRAIN_FLAG=""
fi

echo "================================================================"
echo "Starting Training..."
echo "================================================================"

# Run training
python main.py \
    --config_file ${CONFIG_FILE} \
    --coco_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 4 \
    --options \
        num_classes=4 \
        dn_labelbook_size=4 \
        epochs=50 \
        lr_drop=40 \
        save_checkpoint_interval=5 \
        batch_size=1 \
        use_ema=True \
        ema_decay=0.9997 \
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
