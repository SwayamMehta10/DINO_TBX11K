# Training DINO on TBX11K Dataset - Complete Setup Guide

## Overview
This guide provides complete instructions for training the DINO object detector on the TBX11K tuberculosis detection dataset using ASU's Sol Supercomputer.

**Dataset**: TBX11K - 11,200 chest X-ray images for TB detection  
**Model**: DINO (DETR with Improved Denoising Anchor Boxes)  
**Hardware**: ASU Sol - 1x NVIDIA A100 GPU  
**Official Dataset Splits**: `all_train.json` (6,889 images) and `all_val.json` (2,089 images)

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Configuration](#dataset-configuration)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [FROC Metrics](#froc-metrics)
6. [Multi-Session Training](#multi-session-training)
7. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### 1. Connect to ASU Sol Supercomputer
```bash
ssh your_asurite@login.sol.rc.asu.edu
```

### 2. Clone the Repository
```bash
cd /scratch/$USER
git clone <your-dino-repo-url> DINO_TBX11K
cd DINO_TBX11K
```

### 3. Create Conda Environment
```bash
interactive -t 60 -p htc
$ module load cuda-12.6.1-gcc-12.1.0
$ module load mamba/latest
$ mamba create -n myENV_pytorch -c conda-forge python=3.12
$ source activate myENV_pytorch
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install dependencies (install numpy first, then pycocotools from PyPI)
pip install numpy cython
pip install pycocotools scipy submitit termcolor addict yapf timm

# Install panopticapi (optional, for panoptic segmentation)
pip install git+https://github.com/cocodataset/panopticapi.git

# Build deformable attention operators
# IMPORTANT: Load GCC 12 to match PyTorch's compiler requirements
module load gcc-12.1.0-gcc-11.2.0
gcc --version  # Verify GCC 12.1.0 is loaded

cd models/dino/ops
python setup.py build install
cd ../../..

# Test installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Dataset Configuration

### 1. Verify Dataset Structure
```
TBX11K/
├── imgs/
│   ├── tb/           # TB X-rays
│   ├── health/       # Healthy X-rays
│   ├── sick/         # Sick non-TB X-rays
│   ├── extra/        # Additional datasets
│   └── test/         # Test images
├── annotations/
│   ├── json/
│   │   ├── all_train.json    # 6,889 training images
│   │   ├── all_val.json      # 2,089 validation images
│   │   ├── all_trainval.json # Combined for final training
│   │   └── all_test.json     # Test set
│   └── xml/          # Original XML annotations
├── lists/
│   ├── all_train.txt
│   ├── all_val.txt
│   └── all_trainval.txt
└── code/
    └── make_json_anno.py
```

### 2. Verify JSON Annotations
The JSON files should already be generated. Verify they exist:
```bash
cd TBX11K/annotations/json
ls -lh all_*.json
```

If JSON files are missing or empty, regenerate them:
```bash
cd TBX11K
python code/make_json_anno.py --list_path lists/all_train.txt
python code/make_json_anno.py --list_path lists/all_val.txt
python code/make_json_anno.py --list_path lists/all_trainval.txt
```

### 3. Dataset Classes
TBX11K has **3 tuberculosis detection classes**:
- **ActiveTuberculosis** (category_id: 1) - Active TB
- **ObsoletePulmonaryTuberculosis** (category_id: 2) - Latent TB  
- **PulmonaryTuberculosis** (category_id: 3) - Uncertain TB

**Note**: DINO `num_classes = 4` (3 classes + 1 for background/padding)

---

## Training

### 1. Download Pretrained Weights (Optional but Recommended)
```bash
mkdir -p /scratch/$USER/dino_pretrained
cd /scratch/$USER/dino_pretrained

# Download COCO pretrained ResNet-50 4-scale checkpoint (12 epochs, 49.0 AP)
# Official model zoo: https://github.com/IDEA-Research/DINO#model-zoo

# Recommended: Use 4-scale checkpoint (faster training, good accuracy)
pip install gdown
gdown 1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_  # 4-scale, 12 epochs, 49.0 AP

# Alternative: 5-scale checkpoint (slightly better accuracy, slower)
# gdown <file_id_for_5scale>  # 5-scale, 12 epochs, 49.4 AP

# Note: Use 4-scale for faster training (23 FPS) with similar accuracy
# Use 5-scale only if you need the extra 0.4 AP and can afford slower training
```

### 2. Modify Training Script
Edit `scripts/train_tbx11k.sbatch`:
```bash
# Update email address (line 12)
#SBATCH --mail-user=your_asurite@asu.edu

# Update pretrained model path (line 44-50)
PRETRAIN_MODEL="/scratch/$USER/dino_pretrained/checkpoint0011_4scale.pth"
```

### 3. Submit Training Job
```bash
cd /scratch/$USER/DINO_TBX11K

# Create logs directory
mkdir -p logs

# Submit job
sbatch scripts/train_tbx11k.sbatch
```

### 4. Monitor Training
```bash
# Check job status
squeue -u $USER

# Monitor training log (replace JOBID with actual job ID)
tail -f logs/train_tbx11k_JOBID.out

# Check GPU usage (if job is running)
ssh <node_name>  # Get from squeue
nvidia-smi
```

### 5. Training Configuration
Training parameters in `config/DINO/DINO_4scale_tbx11k.py`:
- **Epochs**: 50 (training takes ~5-7 hours on A100)
- **Learning rate drop**: Epoch 40
- **Batch size**: 1 (per GPU)
- **Checkpoints**: Saved every 5 epochs
- **Best model**: Automatically saved based on validation mAP

Expected output structure:
```
outputs/tbx11k_run_JOBID/
├── checkpoint.pth              # Latest checkpoint (auto-resumes from this)
├── checkpoint0004.pth          # Epoch 4 checkpoint
├── checkpoint0009.pth          # Epoch 9 checkpoint
├── checkpoint0039.pth          # Before LR drop
├── checkpoint_best_regular.pth # Best model (regular)
├── checkpoint_best_ema.pth     # Best model (EMA)
├── config_cfg.py               # Config backup
├── config_args_raw.json        # Args backup
└── log.txt                     # Training log
```

---

## Evaluation

### 1. Modify Evaluation Script
Edit `scripts/eval_tbx11k.sbatch`:
```bash
# Update email (line 12)
#SBATCH --mail-user=your_asurite@asu.edu

# Update checkpoint path (line 38)
export CHECKPOINT_PATH=${PROJECT_ROOT}/outputs/tbx11k_run_XXXXX/checkpoint_best_regular.pth
# Replace XXXXX with your actual training job ID
```

### 2. Submit Evaluation Job
```bash
sbatch scripts/eval_tbx11k.sbatch
```

### 3. View Results
```bash
# Check evaluation output
cat outputs/evaluation_JOBID/log.txt
```

Standard COCO metrics reported:
- **AP** @ IoU=0.50:0.95 (primary metric)
- **AP50** @ IoU=0.50
- **AP75** @ IoU=0.75
- **APsmall**, **APmedium**, **APlarge** (by object size)
- **AR** (Average Recall) @ different detection limits

---

## FROC Metrics

FROC (Free-Response Receiver Operating Characteristic) measures sensitivity vs false positives per image (FPI), which is important for medical imaging evaluation.

### 1. Generate Detection Results for FROC
After evaluation, save detection results in COCO format:
```bash
# Modify main.py evaluation section to save detections
# Or extract from existing evaluation output
```

### 2. Compute FROC Score
```bash
python compute_froc_metrics.py \
    --coco_path TBX11K \
    --results_file outputs/evaluation_JOBID/results.json \
    --output_dir outputs/evaluation_JOBID \
    --ann_file annotations/json/all_val.json
```

### 3. Key Metrics for TBX11K Paper
- **Sensitivity at FPI < 2**: Primary metric reported in TBX11K paper
- **Mean Sensitivity**: Across FPI values [0.125, 0.25, 0.5, 1, 2, 4, 8]

Example output:
```
================================================================
FROC Evaluation Results
================================================================

Sensitivity at different FPI values:
----------------------------------------
  FPI = 0.125:  Sensitivity = 0.6234
  FPI = 0.250:  Sensitivity = 0.7123
  FPI = 0.500:  Sensitivity = 0.7891
  FPI = 1.000:  Sensitivity = 0.8456
  FPI = 2.000:  Sensitivity = 0.8892
  FPI = 4.000:  Sensitivity = 0.9234
  FPI = 8.000:  Sensitivity = 0.9501

----------------------------------------
Mean Sensitivity: 0.8190
Sensitivity at 2 FPI: 0.8892
================================================================
```

---

## Multi-Session Training

Due to the 8-hour walltime limit on Sol's public partition, training may require multiple job submissions.

### Automatic Resume Process

The training script automatically detects existing checkpoints:

1. **First Training Session**:
   ```bash
   sbatch scripts/train_tbx11k.sbatch
   # Trains for ~7.5 hours, saves checkpoint at OUTPUT_DIR/checkpoint.pth
   ```

2. **Continue Training** (before job ends or after timeout):
   ```bash
   # Same script detects checkpoint and resumes automatically
   sbatch scripts/train_tbx11k.sbatch
   ```

3. **Monitor Progress**:
   ```bash
   # Check which epoch training reached
   cat outputs/tbx11k_run_JOBID/log.txt | grep "epoch:"
   ```

### Manual Resume (if needed)
If automatic resume doesn't work, manually specify checkpoint:
```bash
python main.py \
    --config_file config/DINO/DINO_4scale_tbx11k.py \
    --coco_path TBX11K \
    --output_dir outputs/continue_training \
    --resume outputs/tbx11k_run_XXXXX/checkpoint.pth \
    --options num_classes=4 dn_labelbook_size=4
```

### Checkpoint Contents
Each checkpoint contains:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Current epoch number
- EMA model weights (if using EMA)

This ensures seamless continuation of training from the exact state.

---

## Troubleshooting

### Issue 1: Module Not Found Errors
```bash
# Rebuild deformable attention
cd models/dino/ops
python setup.py build install
cd ../../..

# Reinstall requirements
pip install numpy cython pycocotools scipy submitit termcolor addict yapf timm
```

### Issue 1a: GCC Version Error When Building Operators
If you see "You're trying to build PyTorch with a too old version of GCC":
```bash
# Load GCC 12 to match PyTorch's compiler
module load gcc-12.1.0
gcc --version  # Should show 12.1.0

# Rebuild operators
cd models/dino/ops
python setup.py clean --all  # Clean previous build attempts
python setup.py build install
cd ../../..
```

### Issue 2: CUDA Out of Memory
```bash
# Reduce batch size in config
# Edit config/DINO/DINO_4scale_tbx11k.py
batch_size = 1  # Already minimum, cannot reduce further

# Or reduce image size
data_aug_scales = [448, 480, 512]  # Smaller than default
```

### Issue 3: Job Killed Due to Time Limit
```bash
# This is expected for 50 epochs in 8 hours
# Simply resubmit the same script - it will auto-resume
sbatch scripts/train_tbx11k.sbatch

# Estimate progress: ~6-8 minutes per epoch on A100
# 50 epochs * 7 min/epoch ≈ 5.8 hours (should fit in 8 hours)
```

### Issue 4: No Checkpoint Found for Resume
```bash
# Check if OUTPUT_DIR contains checkpoint
ls -lh outputs/tbx11k_run_*/checkpoint.pth

# If checkpoint exists but not detected, manually set in script:
RESUME_FLAG="--resume /full/path/to/checkpoint.pth"
```

### Issue 5: Slow Data Loading
```bash
# Copy dataset to local scratch (faster I/O)
cp -r TBX11K /tmp/TBX11K_$SLURM_JOB_ID
export DATA_PATH=/tmp/TBX11K_$SLURM_JOB_ID

# Don't forget to update paths in script
```

### Issue 6: Import Error - datasets/coco.py
The code has been modified to auto-detect TBX11K structure. If issues persist:
```bash
# Verify the modification was applied
grep -A 10 "Check if this is TBX11K" datasets/coco.py

# Fallback: Create symlinks to match COCO structure
cd TBX11K
ln -s imgs train2017
ln -s imgs val2017
mkdir -p annotations
cd annotations
ln -s ../annotations/json/all_train.json instances_train2017.json
ln -s ../annotations/json/all_val.json instances_val2017.json
```

---

## Expected Results

Based on the paper and DINO's performance:

### COCO Metrics (Validation Set)
- **AP @ IoU=0.50:0.95**: ~0.45-0.55
- **AP50**: ~0.70-0.80
- **AP75**: ~0.50-0.60

### FROC Metrics
- **Sensitivity @ FPI < 2**: ~0.85-0.90 (target metric)
- **Mean Sensitivity**: ~0.80-0.85

**Note**: These are rough estimates. Actual performance depends on:
- Pretraining (using COCO weights significantly improves results)
- Training epochs
- Data augmentation strategy
- Hyperparameter tuning

---

## Final Training for Submission

Once you've validated your approach on `all_train`/`all_val`, train the final model:

### 1. Modify Config for Final Training
Create `config/DINO/DINO_4scale_tbx11k_final.py`:
```python
_base_ = ['DINO_4scale_tbx11k.py']

# Use trainval split (8,978 images)
# This will be handled by modifying the training script
```

### 2. Modify Dataset Path in Training Script
```bash
# Edit the build function call to use "trainval" split
# Or update JSON file paths to use all_trainval.json
```

### 3. Train on all_trainval and Evaluate on all_test
```bash
# Training
sbatch scripts/train_tbx11k.sbatch  # with updated paths

# Evaluation (if test labels available)
sbatch scripts/eval_tbx11k.sbatch
```

---

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{liu2020rethinking,
  title={Rethinking Computer-Aided Tuberculosis Diagnosis},
  author={Liu, Yun and Wu, Yu-Huan and Ban, Yunfeng and Wang, Huifang and Cheng, Ming-Ming},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}

@article{zhang2022dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel M and Shum, Heung-Yeung},
  journal={arXiv preprint arXiv:2203.03605},
  year={2022}
}
```

---

## Contact & Support

For ASU Sol-specific issues:
- ASU Research Computing: https://cores.research.asu.edu/research-computing
- Email: rcsupport@asu.edu

For DINO implementation questions:
- GitHub: https://github.com/IDEA-Research/DINO

---

## Quick Reference Commands

```bash
# Check job status
squeue -u $USER

# Cancel job
scancel <JOBID>

# View job details
scontrol show job <JOBID>

# Check GPU availability
sinfo -p public --Format=nodes,cpus,memory,gres,time,statelong

# Monitor training
tail -f logs/train_tbx11k_*.out

# Check disk usage
du -sh /scratch/$USER/*

# List checkpoints
ls -lht outputs/tbx11k_run_*/checkpoint*.pth
```
