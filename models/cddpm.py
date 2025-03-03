import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Import the common DDPM building blocks from ddpm.py (assumed to be in the same folder)
from ddpm import ConvResNet, Down, Up, Attention

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConditionalUNet, self).__init__()
        self.num_classes = num_classes
        self.inc = ConvResNet(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.attn = Attention(512)
        self.up1 = Up(512, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.outc = nn.Conv2d(128, 1, kernel_size=1)
        self.class_embedding = nn.Embedding(num_classes, 512)

    def forward(self, x, t, class_labels):
        class_emb = self.class_embedding(class_labels).unsqueeze(-1).unsqueeze(-1)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x4 = self.attn(x4)
        x4 = x4 + class_emb
        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        return self.outc

class ConditionalDDPMManager():
    def __init__(self, num_classes=10, T=1000, beta_start=1e-4, beta_end=0.02,
                 lr=2e-5, device='cpu', img_channels=1, img_size=32):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_channels = img_channels
        self.img_size = img_size

        self.unet = ConditionalUNet(num_classes).to(self.device)
        self.optimizer = optim.Adam(self.unet.parameters(), lr=lr)
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.T).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)

    def train(self, dataloader, epochs=30):
        history = {"loss": []}
        for epoch in range(epochs):
            total_loss = []
            for batch_x, batch_y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                t = torch.randint(0, self.T, (batch_x.size(0),), device=self.device).long()
                alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
                noise = torch.randn_like(batch_x)
                noisy_images = torch.sqrt(alpha_bar_t) * batch_x + torch.sqrt(1 - alpha_bar_t) * noise
                noise_pred = self.unet(noisy_images, t, batch_y)
                loss = F.mse_loss(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
            mean_loss = np.mean(total_loss)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {mean_loss:.4f}")
            history["loss"].append(mean_loss)
        return history

    def sample(self, n, class_label):
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size), device=self.device)
            class_labels = torch.tensor([class_label] * n, device=self.device)
            for t in reversed(range(self.T)):
                t_tensor = torch.tensor([t], device=self.device).long()
                alpha_bar_t = self.alpha_bars[t_tensor].view(-1, 1, 1, 1)
                if t > 0:
                    alpha_bar_t_prev = self.alpha_bars[t_tensor - 1].view(-1, 1, 1, 1)
                else:
                    alpha_bar_t_prev = torch.tensor(1.0, device=self.device).view(-1, 1, 1, 1)
                pred_noise = self.unet(x, t_tensor, class_labels)
                x = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                if t > 0:
                    x = x * alpha_bar_t_prev.sqrt() + torch.randn_like(x) * (1 - alpha_bar_t_prev).sqrt()
            return x

    def sample_trajectory(self, n, class_label):
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size), device=self.device)
            class_labels = torch.tensor([class_label] * n, device=self.device)
            trajectories = []
            for t in reversed(range(self.T)):
                t_tensor = torch.tensor([t], device=self.device).long()
                alpha_bar_t = self.alpha_bars[t_tensor].view(-1, 1, 1, 1)
                if t > 0:
                    alpha_bar_t_prev = self.alpha_bars[t_tensor - 1].view(-1, 1, 1, 1)
                else:
                    alpha_bar_t_prev = torch.tensor(1.0, device=self.device).view(-1, 1, 1, 1)
                pred_noise = self.unet(x, t_tensor, class_labels)
                x = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                if t > 0:
                    x = x * alpha_bar_t_prev.sqrt() + torch.randn_like(x) * (1 - alpha_bar_t_prev).sqrt()
                if t % (self.T // 10) == 0 or t == 0:
                    trajectories.append(x.clone())
            return trajectories
