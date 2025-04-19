import os
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from data import get_dataloaders, show_image_and_mask, visualize_batch
from model import train_unet


def setup_directories(base_dir):
    """Create directory structure for the project."""
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    visualizations_dir = os.path.join(base_dir, 'visualizations')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    
    return {
        'base_dir': base_dir,
        'checkpoints_dir': checkpoints_dir,
        'visualizations_dir': visualizations_dir
    }


def get_training_config(args, dirs):
    """Create configuration dictionary for training."""
    return {
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_classes": args.num_classes,
        "image_size": (args.image_size, args.image_size),
        "checkpoint_dir": dirs['checkpoints_dir'],
        "visualization_dir": dirs['visualizations_dir'],
        "save_best_only": not args.save_all,
        "save_interval": args.save_interval
    }


def explore_dataset(args):
    """Explore and visualize dataset samples."""
    data_path = Path(args.data_path)
    images_path = data_path / 'Images'
    masks_path = data_path / 'Masks'
    
    images_paths = sorted(os.listdir(images_path))
    mask_paths = sorted(os.listdir(masks_path))
    
    print(f"Found {len(images_paths)} images and {len(mask_paths)} masks")
    
    # Show a sample image and mask
    vis_dir = Path(args.output_dir) / 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    for i in range(min(3, len(images_paths))):
        fig = show_image_and_mask(
            images_path / images_paths[i], 
            masks_path / mask_paths[i],
            class_count=args.num_classes
        )
        fig.savefig(vis_dir / f'sample_{i}.png')
        plt.close(fig)
    
    # Get dataloaders to explore batch
    train_loader, val_loader = get_dataloaders(
        images_path, 
        masks_path, 
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size
    )
    
    # Visualize a batch
    images, masks = next(iter(train_loader))
    fig = visualize_batch(images, masks, num_samples=4, class_count=args.num_classes)
    fig.savefig(vis_dir / 'batch_samples.png')
    plt.close(fig)
    
    return train_loader, val_loader


def train_model(args, config, train_loader, val_loader):
    """Train the model with the provided configuration."""
    print(f"Using device: {config['device']}")
    
    # Create UNet model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                         in_channels=1, out_channels=args.num_classes, 
                         init_features=64, pretrained=False)
    
    # Train the model
    history = train_unet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Marine Debris Segmentation Training')
    
    # Data and output paths
    parser.add_argument('--data_path', type=str, default='marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation',
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images to')
    parser.add_argument('--num_classes', type=int, default=12,
                        help='Number of segmentation classes')
    
    # Checkpoint options
    parser.add_argument('--save_all', action='store_true',
                        help='Save checkpoints for all epochs instead of best only')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup directory structure
    dirs = setup_directories(args.output_dir)
    
    # Get dataloaders and explore dataset
    train_loader, val_loader = explore_dataset(args)
    
    # Get training configuration
    config = get_training_config(args, dirs)
    
    history = train_model(args, config, train_loader, val_loader)
    print("Training completed successfully!")
    print(f"Best validation IoU: {history['best_val_iou']:.4f}")


if __name__ == "__main__":
    main()

