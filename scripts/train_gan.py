import argparse
import os
import torch
from data.load_data import load_fashion_mnist
from models.gan import GANManager
import torchvision.utils as vutils

def main():
    parser = argparse.ArgumentParser(description="Train GAN on FashionMNIST")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--img_size', type=int, default=32, help="Image size")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device")
    args = parser.parse_args()

    dataloader = load_fashion_mnist(args.batch_size, args.img_size)
    
    gan_manager = GANManager(lr=args.lr, device=args.device)
    
    history = gan_manager.train(dataloader, epochs=args.epochs)
    
    samples = gan_manager.sample(8)
    
    os.makedirs('results', exist_ok=True)
    
    vutils.save_image(samples, "results/gan_samples.png", normalize=True, nrow=4)
    print("Training complete. Generated samples saved to results/gan_samples.png")

if __name__ == "__main__":
    main()