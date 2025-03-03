import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, feature_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_d=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d, feature_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 2, feature_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.net(img)
        return out.view(-1, 1).squeeze(1)

class GANManager():
    def __init__(self, z_dim=100, img_channels=1, feature_g=64, feature_d=64, lr=2e-5, device='cpu'):
        self.device = device
        self.z_dim = z_dim
        self.generator = Generator(z_dim=z_dim, img_channels=img_channels, feature_g=feature_g).to(device)
        self.discriminator = Discriminator(img_channels=img_channels, feature_d=feature_d).to(device)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train(self, dataloader, epochs=10):
        history = {"g_loss": [], "d_loss": []}
        for epoch in range(epochs):
            total_g_loss = []
            total_d_loss = []
            for batch_x, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_size = batch_x.size(0)
                batch_x = batch_x.to(self.device)
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)
                
                # Train Discriminator
                self.discriminator.zero_grad()
                outputs_real = self.discriminator(batch_x)
                d_loss_real = self.criterion(outputs_real, real_labels)
                
                z = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                fake_images = self.generator(z)
                outputs_fake = self.discriminator(fake_images.detach())
                d_loss_fake = self.criterion(outputs_fake, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_D.step()
                
                # Train Generator
                self.generator.zero_grad()
                z = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                fake_images = self.generator(z)
                outputs = self.discriminator(fake_images)
                g_loss = self.criterion(outputs, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

                total_g_loss.append(g_loss.item())
                total_d_loss.append(d_loss.item())
            
            mean_g_loss = np.mean(total_g_loss)
            mean_d_loss = np.mean(total_d_loss)
            print(f"[Epoch {epoch+1}/{epochs}] [D loss: {mean_d_loss:.4f}] [G loss: {mean_g_loss:.4f}]")
            history["g_loss"].append(mean_g_loss)
            history["d_loss"].append(mean_d_loss)
        return history

    def sample(self, n):
        z = torch.randn(n, self.z_dim, 1, 1, device=self.device)
        with torch.no_grad():
            gen_imgs = self.generator(z)
        return gen_imgs