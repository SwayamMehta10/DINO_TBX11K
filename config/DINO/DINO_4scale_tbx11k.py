"""
DINO 4-scale configuration for TBX11K tuberculosis detection dataset
Based on DINO_4scale.py with modifications for 3 TB detection classes
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
lr_drop = 40  # Drop learning rate at epoch 40

# Save checkpoints more frequently for 8-hour job limit
save_checkpoint_interval = 5

# Reduce batch size for single A100 GPU
batch_size = 1  # Adjust based on GPU memory availability

# Learning rate (keep default or slightly lower for medical images)
lr = 0.0001
lr_backbone = 1e-05

# Data augmentation adjusted for 512x512 medical X-ray images
# Be conservative with augmentation for medical images
data_aug_scales = [480, 512, 544, 576]
data_aug_max_size = 600

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
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# Use EMA for more stable training
use_ema = True
ema_decay = 0.9997
ema_epoch = 0

# Dataset
dataset_file = 'coco'  # Use COCO format loader

# Backbone
backbone = 'resnet50'  # Can also try 'swin_T' for better performance
