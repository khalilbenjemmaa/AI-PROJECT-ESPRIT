import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(mask_dir, f))  # only keep if mask exists
        ])

        self.img_base_transform = T.Compose([
            T.Grayscale(),
            T.Resize(self.image_size),
            T.ToTensor(),
        ])

        self.mask_base_transform = T.Compose([
            T.Grayscale(),
            T.Resize(self.image_size, interpolation=Image.NEAREST),
            T.PILToTensor(),  # keep label values intact
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = self.img_base_transform(image)  # [1, H, W]
        mask = self.mask_base_transform(mask).float()  # [1, H, W], keep float to match transforms

        if self.augment:
            image, mask = self.augment_pair(image, mask)

        return image, mask.squeeze(0).long()  # Return mask as [H, W]

    def augment_pair(self, image, mask):
        # Both image and mask must be [1, H, W]

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random 90-degree rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)

        return image, mask


def get_dataloaders(image_dir, mask_dir, image_size=(256, 256), val_split=0.2, batch_size=8):
    """Create train and validation dataloaders."""
    full_dataset = SegmentationDataset(image_dir, mask_dir, image_size=image_size)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Apply augmentations to training set
    train_ds.dataset.augment = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    return train_loader, val_loader


def show_image_and_mask(image_path, mask_path, class_count=12):
    """Display an image and its mask side by side."""
    # Load image and mask
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path)

    # Convert to numpy arrays
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Ensure mask is 2D (height, width)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    # Create a colormap for classes
    cmap = plt.get_cmap('tab20', class_count)

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(image_np)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    im = axs[1].imshow(mask_np, cmap=cmap, vmin=0, vmax=class_count-1)
    axs[1].set_title('Segmentation Mask')
    axs[1].axis('off')

    # Add colorbar for class labels
    cbar = fig.colorbar(im, ax=axs[1], ticks=range(class_count))
    cbar.set_label('Class Index')

    plt.tight_layout()
    return fig


def visualize_batch(images, masks, num_samples=4, class_count=12):
    """
    Show images and masks side by side for a batch.
    images: Tensor [B, 1, H, W]
    masks:  Tensor [B, H, W]
    """
    images = images[:num_samples]
    masks = masks[:num_samples]

    fig, axs = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    cmap = plt.get_cmap('tab20', class_count)

    for i in range(num_samples):
        img = images[i].squeeze().cpu().numpy()
        mask = masks[i].cpu().numpy()

        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 0].set_title("Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(mask, cmap=cmap, vmin=0, vmax=class_count - 1)
        axs[i, 1].set_title("Mask")
        axs[i, 1].axis("off")

    plt.tight_layout()
    return fig