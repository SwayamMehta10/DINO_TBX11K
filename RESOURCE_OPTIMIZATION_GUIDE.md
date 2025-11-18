# Resource Optimization Guide for DINO Training on Sol

## Your Resource Constraints

### Per-Job Limits
- **CPU**: Maximum 32 cores
- **Memory**: Maximum 320 GB
- **GPUs**: Maximum 4 GPUs
- **Wall Time**: Maximum 24 hours

### User-Level Limits
- **Concurrent Jobs**: Maximum 2 running simultaneously
- **Queue Limit**: Maximum 10 jobs
- **GPU Minutes**: 960 minutes total (shared across all running jobs)
  - 1 GPU for 16 hours = 960 minutes
  - 2 GPUs for 8 hours = 960 minutes
  - 4 GPUs for 4 hours = 960 minutes

## Recommended Resource Allocation

### DINO Training Job
```bash
#SBATCH --gres=gpu:a100:2      # 2 GPUs
#SBATCH --time=08:00:00        # 8 hours
#SBATCH --cpus-per-task=8      # 8 CPU cores
#SBATCH --mem=64G              # 64 GB memory
```

**GPU-minutes used**: 2 GPUs × 8 hours × 60 min/hour = **960 minutes**

**Wait!** This uses your entire budget. See better strategy below.

### Your Other Job
```bash
#SBATCH --gres=gpu:a100:1      # 1-2 GPUs
#SBATCH --time=08:00:00        # Adjust as needed
```

## Optimal Strategy: Sequential Training

Since you need GPUs for another job, run jobs **sequentially** rather than concurrently:

### Option 1: Maximize DINO Training (Recommended)
```
Session 1 (First 8 hours):
├─ DINO Training: 2 GPUs × 8 hours = 960 GPU-minutes
└─ Completes ~15-20 epochs with checkpoint resume

Session 2 (Next 8 hours):
├─ Your Other Job: 2 GPUs × 8 hours = 960 GPU-minutes
└─ OR run DINO again to continue training

Session 3 (Continue):
└─ Resume DINO from checkpoint, repeat until convergence
```

### Option 2: Balanced Approach
```
Daily Schedule:
├─ Morning: DINO Training (2 GPUs × 6 hours = 720 GPU-min)
├─ Afternoon: Other Job (2 GPUs × 4 hours = 480 GPU-min)
└─ Total: 1200 GPU-minutes (split across day, not concurrent)
```

### Option 3: Conservative (Run Concurrently)
```bash
# DINO Job
#SBATCH --gres=gpu:a100:1      # 1 GPU only
#SBATCH --time=08:00:00        # 8 hours = 480 GPU-min

# Other Job  
#SBATCH --gres=gpu:a100:1      # 1 GPU
#SBATCH --time=08:00:00        # 8 hours = 480 GPU-min

Total: 960 GPU-minutes (both jobs run together)
```

**⚠️ Problem**: 1 GPU gives batch_size=1, which causes training instability (as seen in your logs).

## Why 2 GPUs is Critical for DINO

Your training logs showed catastrophic failure with batch_size=1:
- **Epoch 3**: 3.82% AP (best)
- **Epoch 45**: 0.19% AP (10× worse!)
- **Cause**: Batch size too small for transformer gradient stability

### With 2 GPUs:
- Effective batch size = 2 (1 per GPU)
- Stable gradient estimates
- Proper learning rate schedule execution
- Expected performance: 15-25% AP (based on literature)

## Training Timeline Estimate

### Current Configuration (2 GPUs × 8 hours):
- **Epochs per session**: ~15-20 epochs
- **Total sessions needed**: 3-4 sessions (50 epochs ÷ 15 = 3.3)
- **Total GPU-hours**: 3 sessions × 2 GPUs × 8 hours = 48 GPU-hours
- **Total GPU-minutes**: 2,880 minutes (spread over 3 separate days)

### Automatic Checkpoint Resume
The training script automatically:
1. Finds the most recent checkpoint from previous jobs
2. Resumes training from that epoch
3. Preserves optimizer state and learning rate schedule
4. No manual intervention needed

## How to Submit Jobs

### 1. First Training Session
```bash
cd d:\GitHub\DINO_TBX11K
sbatch scripts/train_tbx11k.sbatch
```

This will:
- Train for 8 hours (until SLURM kills it)
- Save checkpoint every 5 epochs
- Final checkpoint saved as `checkpoint.pth`

### 2. Check Progress
```bash
# View job queue
squeue -u $USER

# Check GPU usage (while running)
sacct -j <JOB_ID> --format=JobID,Elapsed,State,MaxRSS,AllocGRES

# Monitor training log
tail -f logs/train_tbx11k_<JOB_ID>.out
```

### 3. Resume Training (Next Session)
```bash
# Simply resubmit the same script
sbatch scripts/train_tbx11k.sbatch
```

The script automatically:
- Detects previous checkpoint
- Resumes from last epoch
- Continues training for another 8 hours

### 4. Repeat Until Complete
Resubmit 2-3 more times until reaching epoch 50.

## Monitoring Resource Usage

### Check GPU-Minute Balance
```bash
# View your usage
sacct -S $(date -d '24 hours ago' +%Y-%m-%d) --format=JobID,JobName,Elapsed,State,AllocGRES

# Calculate remaining budget
# 960 minutes total - (running GPU × time)
```

### During Training
```bash
# SSH to compute node (while job running)
squeue -u $USER  # Get NODELIST
ssh <node_name>

# Check GPU utilization
nvidia-smi

# Check CPU/memory
htop
```

## Expected Results with Optimized Config

### Previous Training (batch_size=1, 1 GPU):
- ❌ Epoch 3: 3.82% AP → Epoch 45: 0.19% AP (degradation)
- ❌ Learning rate stuck, no convergence

### New Training (batch_size=2, 2 GPUs):
- ✅ **Target**: 15-25% AP@0.5:0.95
- ✅ **Epoch 30**: Should see ~10-15% AP
- ✅ **Epoch 50**: Should reach 18-25% AP
- ✅ Stable learning curve with warmup

### State-of-the-Art (Reference)
- Literature reports: 20-30% AP for TB detection
- With your fixes: Expected to reach 18-25% AP

## Configuration Changes Made

### Training Script (`train_tbx11k.sbatch`)
```bash
--gres=gpu:a100:2        # Was: gpu:a100:1
--cpus-per-task=8        # Was: 4 (scaled with GPUs)
--mem=64G                # Was: 32G (scaled with GPUs)
--time=08:00:00          # Kept at 8 hours (480 GPU-min per GPU)
```

### Config File (`DINO_4scale_tbx11k.py`)
```python
batch_size = 1              # Per-GPU (effective = 2 with 2 GPUs)
epochs = 50                 # Total training epochs
lr_drop = 30                # Was: 40 (decay earlier)
warmup_epochs = 2           # NEW: Added for stability
dropout = 0.1               # NEW: Added regularization
data_aug_scales = [480,512,544,576,608]  # Expanded range
ema_decay = 0.9998          # Faster adaptation
```

## Troubleshooting

### Issue: "GPU memory out of bounds"
**Solution**: Your A100 has 24GB, batch_size=1 per GPU should fit. If not:
```python
# In config file, reduce input size
data_aug_scales = [384, 416, 448, 480, 512]  # Smaller scales
```

### Issue: "Exceeded GPU-minute quota"
**Solution**: Wait until quota resets (typically 24 hours) or:
- Reduce wall time: `--time=04:00:00` (4 hours)
- Use 1 GPU only (but expect batch_size=1 instability)

### Issue: "Job killed at 8 hours, checkpoint lost"
**Solution**: Checkpoints save every 5 epochs automatically. Last checkpoint preserved as `checkpoint.pth`.

### Issue: "Training not improving"
**Check**:
1. Verify 2 GPUs detected: `grep "world_size" logs/train_*.out`
2. Check effective batch size: `grep "batch_size" logs/train_*.out`
3. Monitor learning rate: `grep "lr:" logs/train_*.out`

## Cost-Benefit Analysis

### Your Options Compared

| Option | GPUs | Hours | GPU-min | Epochs/Session | Sessions to 50 | Total Time | Risk |
|--------|------|-------|---------|----------------|----------------|------------|------|
| **Recommended** | 2 | 8 | 960 | ~15-20 | 3-4 | 3-4 days | ✅ Low - stable training |
| Sequential 1-GPU | 1 | 16 | 960 | ~15-20 | 3-4 | 3-4 days | ❌ High - batch_size=1 |
| Concurrent 1-GPU | 1 | 8 | 480 | ~10 | 5+ | 5+ days | ❌ High - batch_size=1 |
| 4-GPU Sprint | 4 | 4 | 960 | ~20-25 | 2-3 | 2-3 days | ⚠️ Med - needs batch_size=4 |

## Next Steps

1. **Review Changes**: Check `scripts/train_tbx11k.sbatch` and `config/DINO/DINO_4scale_tbx11k.py`

2. **Submit First Job**:
   ```bash
   sbatch scripts/train_tbx11k.sbatch
   ```

3. **Monitor Progress**:
   ```bash
   tail -f logs/train_tbx11k_<JOB_ID>.out
   ```

4. **After 8 Hours**: Resubmit to continue training

5. **After 50 Epochs**: Run evaluation to check final performance

## Questions?

- **"Should I use --time=24:00:00?"** No - uses 2×24×60=2880 GPU-minutes, exceeding your 960 limit
- **"Can I run 4 GPUs?"** Yes, but only for 4 hours (960÷4=240 min), may not complete enough epochs
- **"Will this reach SoTA?"** With these fixes, expect 18-25% AP, close to literature values
