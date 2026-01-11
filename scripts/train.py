"""
Training script for road damage detection model.
"""

import os
import sys
import yaml
import logging
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_yolo(config: dict, model_variant: str = 'l'):
    """
    Train YOLOv8/v11 model for road damage detection.
    
    Args:
        config: Configuration dictionary
        model_variant: Model variant (n, s, m, l, x)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Install with: pip install ultralytics")
        return False
    
    logger.info("="*60)
    logger.info("Road Damage Detection - YOLO Training")
    logger.info("="*60)
    
    # Check if dataset.yaml exists
    data_dir = config['paths']['data_dir']
    dataset_yaml = os.path.join(data_dir, 'dataset.yaml')
    
    if not os.path.exists(dataset_yaml):
        logger.error(f"dataset.yaml not found at {dataset_yaml}")
        logger.error("Please ensure dataset is properly prepared and split.")
        return False
    
    # Create checkpoint directory
    checkpoint_dir = config['paths']['checkpoints_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model
    model_name = f"yolov8{model_variant}"
    logger.info(f"Loading {model_name} model...")
    model = YOLO(f"{model_name}.pt")
    
    # Training configuration
    train_config = config['training']
    device = 0 if config['device']['use_cuda'] else 'cpu'
    
    logger.info(f"Training configuration:")
    logger.info(f"  - Epochs: {train_config['epochs']}")
    logger.info(f"  - Batch size: {train_config['batch_size']}")
    logger.info(f"  - Learning rate: {train_config['learning_rate']}")
    logger.info(f"  - Image size: {config['data']['image_size']}")
    logger.info(f"  - Device: {device}")
    
    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=train_config['epochs'],
        imgsz=config['data']['image_size'],
        batch=train_config['batch_size'],
        device=device,
        patience=train_config['patience'],
        save=True,
        save_period=train_config['save_interval'],
        project=checkpoint_dir,
        name=f"rdd_yolo{model_variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        optimizer=train_config['optimizer'],
        lr0=train_config['learning_rate'],
        lrf=train_config['scheduler'],
        weight_decay=train_config['weight_decay'],
        warmup_epochs=train_config['warmup_epochs'],
        hsv_h=config['augmentation']['hsv_h'],
        hsv_s=config['augmentation']['hsv_s'],
        hsv_v=config['augmentation']['hsv_v'],
        degrees=config['augmentation']['degrees'],
        translate=config['augmentation']['translate'],
        scale=config['augmentation']['scale'],
        flipud=config['augmentation']['flipud'],
        fliplr=config['augmentation']['fliplr'],
        perspective=config['augmentation']['perspective'],
        mosaic=config['augmentation']['mosaic'],
        close_mosaic=5,  # Close mosaic in last 5 epochs
        verbose=True,
        seed=42
    )
    
    logger.info("="*60)
    logger.info("Training completed successfully!")
    logger.info("="*60)
    
    # Validate
    logger.info("\nRunning validation...")
    metrics = model.val()
    
    logger.info(f"Validation Results:")
    logger.info(f"  - mAP@0.5: {metrics.box.map50}")
    logger.info(f"  - mAP@0.5:0.95: {metrics.box.map}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Train road damage detection model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--variant', type=str, default='l',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model variant')
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Create directories
    for path_key in ['checkpoints_dir', 'logs_dir', 'results_dir']:
        os.makedirs(config['paths'][path_key], exist_ok=True)
    
    # Train
    success = train_yolo(config, model_variant=args.variant)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
