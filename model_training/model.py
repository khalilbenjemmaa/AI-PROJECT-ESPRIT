import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def compute_iou(preds, masks, num_classes):
    """Compute IoU metric between prediction and ground truth."""
    smooth = 1e-6
    if num_classes == 1:
        intersection = (preds & masks.bool()).float().sum((1, 2))
        union = (preds | masks.bool()).float().sum((1, 2))
    else:
        intersection = ((preds == masks) & (masks != 0)).float().sum()
        union = ((preds != 0) | (masks != 0)).float().sum()
    return ((intersection + smooth) / (union + smooth)).item()


def plot_training_metrics(train_losses, val_losses, val_ious, save_path=None):
    """Plot and optionally save training metrics."""
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ious, label='Val IoU')
    plt.title('IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training metrics plot saved to {save_path}")
    
    return plt.gcf()


def visualize_predictions(model, dataloader, device, num_samples=4, class_count=12, save_path=None):
    """
    Visualize predictions from model.
    Shows: input image, ground truth mask, predicted mask
    """
    model.eval()
    images, masks = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    # Move to CPU for plotting
    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    n = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))

    if n == 1:
        axes = [axes]  # Ensure consistent indexing for 1 sample

    for i in range(n):
        img = images[i].squeeze(0)  # [H, W]
        gt = masks[i]
        pred = preds[i]

        axes[i][0].imshow(img, cmap='gray')
        axes[i][0].set_title("Input Image")
        axes[i][1].imshow(gt, cmap='tab20', vmin=0, vmax=class_count-1)
        axes[i][1].set_title("Ground Truth")
        axes[i][2].imshow(pred, cmap='tab20', vmin=0, vmax=class_count-1)
        axes[i][2].set_title("Prediction")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction visualization saved to {save_path}")
    
    return fig


def train_unet(model, train_loader, val_loader, config):
    """Train UNet model with the provided configuration."""
    device = config["device"]
    epochs = config["epochs"]
    lr = config["learning_rate"]
    num_classes = config["num_classes"]
    checkpoint_dir = config["checkpoint_dir"]
    vis_dir = config["visualization_dir"]
    save_best_only = config["save_best_only"]
    save_interval = config.get("save_interval", 5)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    best_val_iou = 0.0
    model = model.to(device)

    # Loss function
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    train_losses, val_losses, val_ious = [], [], []
    epoch_checkpoints = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        batch_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training')
        
        for images, masks in batch_progress:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if num_classes == 1:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, masks.float())
            else:
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Update progress bar
            batch_progress.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss, total_iou = 0.0, 0.0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation')
        
        with torch.no_grad():
            for images, masks in val_progress:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                if num_classes == 1:
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, masks.float())
                    preds = torch.sigmoid(outputs) > 0.5
                else:
                    loss = criterion(outputs, masks)
                    preds = torch.argmax(outputs, dim=1)

                val_loss += loss.item()
                batch_iou = compute_iou(preds, masks, num_classes)
                total_iou += batch_iou
                
                # Update progress bar
                val_progress.set_postfix({'loss': loss.item(), 'iou': batch_iou})

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)

        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # Save checkpoint
        is_best = avg_val_iou > best_val_iou
        if is_best:
            best_val_iou = avg_val_iou
            
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_iou': avg_val_iou,
            'val_loss': avg_val_loss,
            'train_loss': epoch_train_loss,
            'config': config
        }
        
        # Save checkpoint based on strategy
        if (is_best and save_best_only) or not save_best_only:
            if is_best:
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pt')
                torch.save(checkpoint_data, checkpoint_path)
                print(f"✅ Best model saved at {checkpoint_path}")
            
            # Always save at regular intervals regardless of best_only
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                interval_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pt')
                torch.save(checkpoint_data, interval_path)
                epoch_checkpoints.append(interval_path)
                print(f"💾 Checkpoint saved at {interval_path}")
        
        # Visualize predictions periodically
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            vis_path = os.path.join(vis_dir, f'predictions_epoch_{epoch+1}.png')
            visualize_predictions(model, val_loader, device, num_samples=4, 
                                  class_count=num_classes, save_path=vis_path)

        # Update scheduler
        scheduler.step(avg_val_iou)

    # Save final metrics plot
    metrics_path = os.path.join(vis_dir, 'training_metrics.png')
    plot_training_metrics(train_losses, val_losses, val_ious, save_path=metrics_path)
    
    # Save metrics as CSV
    metrics_csv_path = os.path.join(vis_dir, 'training_metrics.csv')
    np.savetxt(metrics_csv_path, 
               np.column_stack((train_losses, val_losses, val_ious)), 
               delimiter=',', 
               header='train_loss,val_loss,val_iou')
    
    # Return training history
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'best_val_iou': best_val_iou,
        'checkpoints': epoch_checkpoints
    }
