# Training Issue: Immediate Exit - What Happened and How to Fix

## What Happened

Your 1-GPU training job (39581627) **exited immediately** after only 5 seconds without running any training epochs:

```
Start Time: Mon Nov 17 18:06:06 MST 2025
Training time 0:00:00
End Time: Mon Nov 17 18:07:05 MST 2025
```

## Root Cause

The script tried to **automatically resume** from your old checkpoint (`tbx11k_run_39543743`), which was trained with:
- ❌ `batch_size=1` WITHOUT gradient accumulation
- ❌ Incompatible optimizer state
- ❌ Different training strategy

The new 1-GPU script expects:
- ✅ `batch_size=1` WITH `gradient_accumulation_steps=2`
- ✅ Different optimizer step frequency
- ✅ New training strategy

When PyTorch tried to load the old checkpoint into the new training loop, it likely encountered a mismatch that caused the training to exit cleanly (no error, just quit).

## Evidence from Logs

```bash
Resuming from previous job checkpoint: /scratch/smehta90/DINO_TBX11K/outputs/tbx11k_run_39543743/checkpoint.pth
Resuming from checkpoint - pretrained weights not needed
```

Then immediately:
```bash
Start training
Training time 0:00:00  # <-- NO EPOCHS RAN!
Training completed successfully!
```

## The Fix (Already Applied)

I've updated `scripts/train_tbx11k_1gpu.sbatch` to **NOT** automatically resume from old checkpoints:

```bash
# Changed from:
export PREV_CHECKPOINT_DIR=$(ls -td ${PROJECT_ROOT}/outputs/tbx11k_run_* 2>/dev/null | ...)

# To:
export PREV_CHECKPOINT_DIR=""  # Start fresh!
```

This forces fresh training with:
- ✅ Pretrained COCO weights (if available)
- ✅ Gradient accumulation enabled
- ✅ No incompatible checkpoint baggage

## Action Required: Resubmit Job

### Option A: Quick Resubmit (RECOMMENDED)

```bash
cd /scratch/$USER/DINO_TBX11K

# Resubmit the fixed script
sbatch scripts/train_tbx11k_1gpu.sbatch

# Monitor the new job
squeue -u $USER
tail -f logs/train_tbx11k_<NEW_JOB_ID>.out
```

### Option B: Rename Old Checkpoint First (Extra Safe)

```bash
cd /scratch/$USER/DINO_TBX11K

# Use the convenience script
bash scripts/submit_fresh_1gpu.sh

# OR manually:
mv outputs/tbx11k_run_39543743 outputs/tbx11k_run_39543743_FAILED_BATCH1
sbatch scripts/train_tbx11k_1gpu.sbatch
```

## What to Expect This Time

### First Few Lines of Output:
```
Resuming from checkpoint - pretrained weights not needed  # This should NOT appear
Using pretrained weights: /scratch/.../checkpoint0011_4scale.pth  # This SHOULD appear
```

### Training Should Start:
```
Epoch: [0]  [  0/6889]  eta: ...  loss: 2.xxxx  ...
Epoch: [0]  [ 10/6889]  eta: ...  loss: 2.xxxx  ...
Epoch: [0]  [ 20/6889]  eta: ...  loss: 2.xxxx  ...
```

### After First Epoch (~20 minutes):
```
Averaged stats: loss: 2.xxx  class_error: xx.xx  ...
Test: Epoch [0] AP@0.5:0.95: 0.02-0.04 (2-4%)  # Healthy start!
```

## Verification Checklist

After resubmitting, check these in the log file:

- [ ] **"Using pretrained weights"** message appears (NOT "Resuming from checkpoint")
- [ ] **Training starts**: "Epoch: [0] [0/6889]" appears
- [ ] **Loss values**: Around 2.0-2.5 for first epoch
- [ ] **First epoch completes**: Takes ~20 minutes
- [ ] **Test AP**: Should be 2-4% after epoch 0 (not 0.18%)
- [ ] **No immediate exit**: Job runs for multiple hours

## Why This Happened

The automatic checkpoint resume logic was designed for **2-GPU training** where:
- Checkpoint is compatible (same batch_size strategy)
- Optimizer state matches training loop
- Resume is seamless

For **1-GPU with gradient accumulation**, we're using a **different training strategy**, so old checkpoints are incompatible.

## Future Sessions

Once you have a checkpoint from the NEW 1-GPU training:
1. Future 1-GPU jobs CAN resume from it
2. Mark it clearly: `tbx11k_run_XXXXX_1GPU_GRADACCUM`
3. Don't mix with 2-GPU checkpoints

## Alternative: Switch to 2-GPU Training

If your 2-GPU job (if you submitted one) is running, you can:
1. Cancel the 1-GPU job
2. Let 2-GPU continue
3. Start fresh with proper batch_size=2 on 2 GPUs

The 2-GPU script (`train_tbx11k.sbatch`) also has the same auto-resume issue, so make sure it starts FRESH training too.

## Summary

**Problem**: Old checkpoint incompatible with gradient accumulation  
**Solution**: Start fresh training (already fixed in script)  
**Action**: Resubmit with `sbatch scripts/train_tbx11k_1gpu.sbatch`  
**Expected**: Training will run for ~8 hours, completing 12-15 epochs
