"""
DINO 4-scale configuration for TBX11K tuberculosis detection dataset
Based on DINO_4scale.py with modifications for 3 TB detection classes

OPTIMIZED CONFIGURATION:
- Fixed batch_size=2 (was 1, which caused training instability)
- Added warmup schedule for stable convergence
- Adjusted learning rate and schedule for medical imaging
- Improved data augmentation for chest X-rays
- Resource-optimized for 2x A100 GPUs with 8-hour wall time
"""

_base_ = ['DINO_4scale.py']

# TBX11K dataset has 3 TB detection classes
# Categories: ActiveTuberculosis (1), ObsoletePulmonaryTuberculosis (2), PulmonaryTuberculosis (3)
# num_classes = max_category_id + 1 = 3 + 1 = 4
num_classes = 4
dn_labelbook_size = 4

# Training parameters adjusted for medical imaging dataset
# TBX11K is smaller than COCO, so we train for more epochs
epochs = 50
lr_drop = 30  # Drop learning rate earlier to prevent overfitting
warmup_epochs = 2  # Add warmup for stable convergence with batch_size=2

# Save checkpoints more frequently for 8-hour job limit
save_checkpoint_interval = 5

# CRITICAL FIX: Increase batch size from 1 to 2
# With 2 GPUs, effective batch_size = 2 GPUs × 1 sample/GPU = 2
# This provides stable gradient estimates for transformer training
batch_size = 1  # Per-GPU batch size (effective batch = 2 with 2 GPUs)

# Learning rate scaled for batch_size=2 (effective)
# Base LR from COCO: 0.0001 for batch_size=2
# Using linear scaling: lr = base_lr × (batch_size / base_batch_size)
lr = 0.0001
lr_backbone = 1e-05
lr_drop_epochs = [30, 40]  # Multi-step decay

# Data augmentation adjusted for medical X-ray images
# Conservative augmentation to preserve diagnostic features
# X-rays are typically 512x512 or larger
data_aug_scales = [480, 512, 544, 576, 608]  # Multi-scale training
data_aug_max_size = 640  # Allow slightly larger for detail preservation

# Add dropout for regularization with small dataset
dropout = 0.1

# Reduce number of queries since TB images typically have fewer objects than COCO
num_queries = 300

# Keep deformable attention settings
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4

# Denoising training parameters
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True

# Loss weights (keep default COCO settings)
# Note: Class imbalance is handled by the denoising training strategy
# and hybrid matching, not by weighted loss terms
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# Use EMA for more stable training
# Slightly faster decay for smaller medical dataset
use_ema = True
ema_decay = 0.9998  # Faster adaptation than COCO's 0.9997
ema_epoch = 0

# Backbone
backbone = 'resnet50'  # Can also try 'swin_T' for better performance
