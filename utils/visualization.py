import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

def plot_images(images, title=None, nrow=8, figsize=(16, 4)):
    if isinstance(images, list):
        images = torch.stack(images)
    grid_img = make_grid(images, nrow=nrow, normalize=True)
    plt.figure(figsize=figsize)
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_loss_curves(loss_dict, title="Training Loss", xlabel="Epoch", ylabel="Loss", figsize=(8, 6)):
    plt.figure(figsize=figsize)
    for label, losses in loss_dict.items():
        plt.plot(losses, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_class_specific_images(generated_images, images_per_class, num_classes, class_names=None, figsize=None):
    if figsize is None:
        figsize = (images_per_class * 2, num_classes * 2)
    plt.figure(figsize=figsize)
    for class_idx in range(num_classes):
        class_imgs = generated_images[class_idx]
        for img_idx in range(images_per_class):
            plt_idx = class_idx * images_per_class + img_idx + 1
            plt.subplot(num_classes, images_per_class, plt_idx)
            img = class_imgs[img_idx] if isinstance(class_imgs, list) else class_imgs[img_idx]
            plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
            plt.axis('off')
            if img_idx == 0:
                if class_names:
                    plt.title(f"{class_names[class_idx]}", fontsize=8)
                else:
                    plt.title(f"Class {class_idx}", fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_trajectories_per_class(trajectories_dict, class_names=None, figsize=(15, 10)):
    num_classes = len(trajectories_dict)
    max_steps = max(len(traj) for traj in trajectories_dict.values())
    plt.figure(figsize=figsize)
    for class_idx, traj in trajectories_dict.items():
        for step_idx, img in enumerate(traj):
            plt_idx = class_idx * max_steps + step_idx + 1
            plt.subplot(num_classes, max_steps, plt_idx)
            img_np = (img.squeeze().cpu().numpy() + 1) / 2
            plt.imshow(img_np, cmap="gray")
            plt.axis("off")
            if step_idx == 0:
                if class_names:
                    label = class_names[class_idx] if isinstance(class_names, list) else class_names.get(class_idx, f"Class {class_idx}")
                else:
                    label = f"Class {class_idx}"
                plt.ylabel(label, rotation=0, labelpad=20, fontsize=8, va="center")
    plt.tight_layout()
    plt.show()