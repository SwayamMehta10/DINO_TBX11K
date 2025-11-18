"""
DINO 4-scale OPTIMIZED configuration for TBX11K tuberculosis detection
Based on DINO_4scale.py with critical optimizations for better performance
"""

_base_ = ['DINO_4scale.py']

# TBX11K dataset configuration
num_classes = 4
dn_labelbook_size = 4

# ============================================================================
# CRITICAL: Training parameters optimized for single A100 GPU
# ============================================================================

# INCREASED EPOCHS - Medical datasets need more training
epochs = 150  # Increased from 50
lr_drop = 120  # Drop LR later to allow full convergence

# Save checkpoints frequently
save_checkpoint_interval = 10

# CRITICAL: Increase batch size using gradient accumulation
batch_size = 2  # Base batch size (what fits in memory)
# If you can fit more, try batch_size=4

# CRITICAL: Scale learning rate with effective batch size
# Original DINO uses batch_size=16 with lr=0.0001
# We're using effective batch_size=2, so scale down proportionally
lr = 0.0001 * (2 / 16)  # = 0.0000125 (1.25e-5)
lr_backbone = lr * 0.1  # = 1.25e-6

# Learning rate warmup - CRITICAL for transformer training
# Add these as options when running
# --options ... lr_warmup_epochs=5

# ============================================================================
# Data augmentation - More conservative for medical imaging
# ============================================================================
data_aug_scales = [480, 512, 544, 576, 608]  # More scale variation
data_aug_max_size = 640  # Larger max size

# ============================================================================
# Model architecture tweaks for better performance
# ============================================================================

# Increase number of queries for medical images (tend to have fewer but important objects)
num_queries = 300  # Keep default

# Deformable attention settings - keep default
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4

# Denoising training parameters - these help a lot!
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True

# ============================================================================
# Loss weights - tune for medical imaging
# ============================================================================
cls_loss_coef = 2.0  # Increase classification loss (was 1.0)
bbox_loss_coef = 5.0  # Keep bbox loss
giou_loss_coef = 2.0  # Keep giou loss

# ============================================================================
# EMA for more stable training - CRITICAL
# ============================================================================
use_ema = True
ema_decay = 0.9999  # Slightly higher than before (was 0.9997)
ema_epoch = 0

# ============================================================================
# Backbone - Consider upgrading for better features
# ============================================================================
backbone = 'resnet50'  
# For better performance, try: 'swin_T' or 'swin_S' if you have time
# These require downloading additional pretrained weights

# ============================================================================
# Dropout for regularization on small dataset
# ============================================================================
dropout = 0.1  # Add slight dropout (was 0.0)

# ============================================================================
# Advanced: Gradient clipping for stability
# ============================================================================
clip_max_norm = 0.1  # Keep default

