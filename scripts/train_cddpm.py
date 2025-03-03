import argparse
import os
import torch
from data.load_data import load_fashion_mnist
from models.cddpm import ConditionalDDPMManager
import torchvision.utils as vutils

def main():
    parser = argparse.ArgumentParser(description="Train Conditional DDPM on FashionMNIST")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--img_size', type=int, default=32, help="Image size")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device")
    parser.add_argument('--class_label', type=int, default=0, help="Target class label for generation")
    args = parser.parse_args()

    dataloader = load_fashion_mnist(args.batch_size, args.img_size)
    
    cddpm_manager = ConditionalDDPMManager(num_classes=10, lr=args.lr, device=args.device, img_size=args.img_size)
    
    history = cddpm_manager.train(dataloader, epochs=args.epochs)
    
    samples = cddpm_manager.sample(8, class_label=args.class_label)
    
    os.makedirs('results', exist_ok=True)
    
    vutils.save_image(samples, "results/cddpm_samples.png", normalize=True, nrow=4)
    print("Training complete. Generated samples saved to results/cddpm_samples.png")

if __name__ == "__main__":
    main()