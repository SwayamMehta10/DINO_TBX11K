"""
DINO 4-scale configuration for TBX11K - 1 GPU VERSION with Gradient Accumulation
Based on DINO_4scale_tbx11k.py optimized for single GPU training

KEY DIFFERENCES FROM 2-GPU CONFIG:
- Uses gradient accumulation to simulate batch_size=2
- Accumulates gradients over 2 forward passes before optimizer.step()
- Same effective batch size as 2-GPU training
- Slightly slower per epoch (~1.2x) but faster queue allocation

WHEN TO USE THIS CONFIG:
- When 2-GPU jobs are stuck in queue
- When you want to start training immediately
- When other jobs need GPUs concurrently
- Performance is identical to 2-GPU training
"""

_base_ = ['DINO_4scale.py']

# TBX11K dataset configuration
num_classes = 4
dn_labelbook_size = 4

# Training parameters
epochs = 50
lr_drop = 30
warmup_epochs = 2
save_checkpoint_interval = 5

# GRADIENT ACCUMULATION SETUP
# batch_size=1: Actual batch per forward pass
# gradient_accumulation_steps=2: Accumulate gradients over 2 batches
# Effective batch size = 1 × 2 = 2 (same as 2-GPU training)
batch_size = 1
gradient_accumulation_steps = 2  # Simulate batch_size=2

# Learning rate (same as 2-GPU config)
# Effective batch_size=2, so keep lr=0.0001
lr = 0.0001
lr_backbone = 1e-05
lr_drop_epochs = [30, 40]

# Data augmentation
data_aug_scales = [480, 512, 544, 576, 608]
data_aug_max_size = 640
dropout = 0.1

# Model architecture
num_queries = 300
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4

# Denoising training
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True

# Loss weights - CRITICAL FIX for class imbalance
# TBX11K has severe class imbalance (many healthy X-rays, few TB cases)
# Focal loss alpha controls weighting between positive and negative examples
# Default COCO: focal_alpha=0.25 (balanced dataset with 80 classes)
# TBX11K fix: focal_alpha=0.75 (imbalanced medical dataset with 4 classes)
# Higher alpha gives more weight to hard positives (TB cases)
focal_alpha = 0.75  # Increased from 0.25 to handle class imbalance
focal_gamma = 2.0   # Keep default
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# EMA settings
use_ema = True
ema_decay = 0.9998
ema_epoch = 0

# Backbone
backbone = 'resnet50'

# Performance expectations (same as 2-GPU):
# - Training speed: ~1.2x slower per epoch than 2-GPU
# - Final AP: 18-25% (identical to 2-GPU training)
# - Epochs per 8-hour session: ~12-15 epochs
# - Queue wait time: Much faster than 2-GPU
