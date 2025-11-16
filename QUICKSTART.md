# Quick Start Guide - DINO Training on TBX11K

## Prerequisites Checklist

- [ ] ASU Sol account with access to public partition
- [ ] SSH access to Sol configured
- [ ] Basic familiarity with SLURM job submission
- [ ] ~100GB free space in `/scratch/$USER/`

## 5-Minute Quick Start

### 1. Login and Navigate
```bash
ssh your_asurite@sol.asu.edu
cd /scratch/$USER
```

### 2. Setup Environment (One-Time)
```bash
# Load and create conda environment
module load mamba/latest
mamba create -n dino_env python=3.9 -y
source activate dino_env

# Install dependencies
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
cd DINO_TBX11K
pip install -r requirements.txt

# Build operators
cd models/dino/ops && python setup.py build install && cd ../../..
```

### 3. Download Pretrained Weights
```bash
mkdir -p /scratch/$USER/dino_pretrained
cd /scratch/$USER/dino_pretrained
wget https://github.com/IDEA-Research/DINO/releases/download/v0.1/checkpoint0011_4scale.pth
```

### 4. Configure and Launch Training
```bash
cd /scratch/$USER/DINO_TBX11K

# Edit email in training script
sed -i 's/your_email@asu.edu/YOUR_ASURITE@asu.edu/g' scripts/train_tbx11k.sbatch

# Create logs directory
mkdir -p logs

# Submit job
sbatch scripts/train_tbx11k.sbatch
```

### 5. Monitor Training
```bash
# Check job status
squeue -u $USER

# View live training log
tail -f logs/train_tbx11k_*.out

# Check progress (replace JOBID)
grep "Train Epoch" logs/train_tbx11k_JOBID.out | tail -5
```

## After Training Completes

### Run Evaluation
```bash
# Edit checkpoint path in eval script
nano scripts/eval_tbx11k.sbatch
# Change CHECKPOINT_PATH to your trained model

# Submit evaluation
sbatch scripts/eval_tbx11k.sbatch

# View results
cat outputs/evaluation_*/log.txt
```

### Compute FROC Metrics
```bash
python compute_froc_metrics.py \
    --coco_path TBX11K \
    --results_file outputs/evaluation_*/results.json \
    --output_dir outputs/evaluation_*
```

## Expected Timeline

- **Environment Setup**: 15-20 minutes (one-time)
- **Pretrained Download**: 2-3 minutes
- **Training (50 epochs)**: 5-7 hours on A100
- **Evaluation**: 10-15 minutes
- **FROC Computation**: 2-3 minutes

## Need More Details?

See [TRAINING_SETUP.md](TRAINING_SETUP.md) for:
- Detailed troubleshooting
- Multi-session training for 8-hour limit
- Configuration explanations
- Expected performance metrics

## Quick Commands Reference

```bash
# Job management
squeue -u $USER              # Check your jobs
scancel <JOBID>              # Cancel a job
scontrol show job <JOBID>    # Job details

# Monitor
tail -f logs/train_*.out     # Watch training
nvidia-smi                   # GPU usage (on compute node)

# Files
ls -lh outputs/*/checkpoint*.pth  # List checkpoints
du -sh /scratch/$USER/*           # Check disk usage
```

## Common Issues & Quick Fixes

### "Out of Memory"
Already using minimum batch size (1). Cannot reduce further. Ensure no other jobs running on GPU.

### "Module not found"
```bash
cd models/dino/ops
python setup.py build install
cd ../../..
```

### "Job exceeded walltime"
Expected! Just resubmit - automatic resume will continue training:
```bash
sbatch scripts/train_tbx11k.sbatch
```

### "Cannot find checkpoint"
Check output directory:
```bash
ls -lh outputs/tbx11k_run_*/
```

## Getting Help

1. Check [TRAINING_SETUP.md](TRAINING_SETUP.md) for detailed docs
2. Review error logs: `cat logs/train_tbx11k_JOBID.err`
3. ASU RC Support: rcsupport@asu.edu
4. DINO Issues: https://github.com/IDEA-Research/DINO/issues
