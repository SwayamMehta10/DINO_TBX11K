#!/bin/bash

# QUICK FIX: Submit fresh 1-GPU training without resuming from bad checkpoint
# This bypasses the checkpoint resume issue

cd /scratch/$USER/DINO_TBX11K

# Rename the old failed output directory to prevent auto-resume
if [ -d "outputs/tbx11k_run_39543743" ]; then
    mv outputs/tbx11k_run_39543743 outputs/tbx11k_run_39543743_FAILED_BATCH1
    echo "Renamed old checkpoint to prevent auto-resume"
fi

# Submit fresh 1-GPU training
echo "Submitting fresh 1-GPU training with gradient accumulation..."
sbatch scripts/train_tbx11k_1gpu.sbatch

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/train_tbx11k_<JOB_ID>.out"
