"""
DINO 4-scale configuration for TBX11K with Swin-Tiny backbone
Alternative to ResNet50 for potentially better performance

COMPARISON WITH RESNET50:
- Expected AP improvement: +2-4% over ResNet50
- Training speed: ~30% slower
- GPU memory: ~20% more
- Recommended: Use after establishing ResNet50 baseline

WHEN TO USE THIS:
1. After ResNet50 training shows stable convergence
2. When you have budget for longer training sessions
3. If ResNet50 plateaus below 20% AP

PRETRAINED WEIGHTS NEEDED:
Download Swin-Tiny ImageNet-1K pretrained:
  https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
Save to: /scratch/$USER/dino_pretrained/swin_tiny_patch4_window7_224.pth
"""

_base_ = ['DINO_4scale.py']

# TBX11K dataset configuration
num_classes = 4
dn_labelbook_size = 4

# Training parameters - same as ResNet50 config
epochs = 50
lr_drop = 30
warmup_epochs = 2
save_checkpoint_interval = 5

# Batch size per GPU (effective batch = 2 with 2 GPUs)
batch_size = 1

# Learning rate - same as ResNet50
lr = 0.0001
lr_backbone = 1e-05
lr_drop_epochs = [30, 40]

# Data augmentation
data_aug_scales = [480, 512, 544, 576, 608]
data_aug_max_size = 640
dropout = 0.1

# Reduce queries for medical images
num_queries = 300

# Deformable attention settings
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4

# Denoising training
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True

# Loss weights
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# EMA settings
use_ema = True
ema_decay = 0.9998
ema_epoch = 0

# SWIN TRANSFORMER BACKBONE
# Available options: 'swin_T_224_1k', 'swin_B_224_22k', 'swin_B_384_22k', 
#                    'swin_L_224_22k', 'swin_L_384_22k'
backbone = 'swin_T_224_1k'  # Swin-Tiny (28M params vs ResNet50 25M params)

# Enable gradient checkpointing to save memory
use_checkpoint = True  # Critical for Swin with limited GPU memory

# Backbone directory for pretrained weights
# Set this in training command or ensure weights are in default location
# backbone_dir = '/scratch/$USER/dino_pretrained'

# NOTES:
# 1. Swin-Tiny is comparable to ResNet50 in size but uses attention
# 2. Expected performance on TBX11K: 20-28% AP (vs 18-25% for ResNet50)
# 3. Training time per epoch: ~1.3× ResNet50
# 4. GPU memory: Fits in A100 24GB with batch_size=1 per GPU
# 5. Use this config AFTER establishing ResNet50 baseline
