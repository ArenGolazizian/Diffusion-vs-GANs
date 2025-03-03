import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# --- Building Blocks for UNet ---
class ConvResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=2):
        super(ConvResNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.residuals = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_residuals)]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x + self.residuals(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip, t):
        x = self.up(x)
        return torch.cat((x, skip), dim=1)

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(in_channels, num_heads=8)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x_reshaped = x.view(batch, channels, -1).permute(2, 0, 1)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.permute(1, 2, 0).view(batch, channels, height, width)
        return x + attn_out

# --- UNet Architecture ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = ConvResNet(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.attn = Attention(512)
        self.up1 = Up(512, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.outc = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x4 = self.attn(x4)
        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        return self.outc(x)

class DDPMManager():
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, lr=2e-5, device='cpu', img_channels=1, img_size=32):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_channels = img_channels
        self.img_size = img_size
        
        self.unet = UNet().to(self.device)
        self.optimizer = optim.Adam(self.unet.parameters(), lr=lr)
        
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.T).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)

    def train(self, dataloader, epochs=10):
        history = {"loss": []}
        for epoch in range(epochs):
            total_loss = []
            for batch_x, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_x = batch_x.to(self.device)
                t = torch.randint(0, self.T, (batch_x.size(0),), device=self.device).long()
                alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
                noise = torch.randn_like(batch_x)
                noisy_images = torch.sqrt(alpha_bar_t) * batch_x + torch.sqrt(1 - alpha_bar_t) * noise
                noise_pred = self.unet(noisy_images, t)
                loss = F.mse_loss(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
            mean_loss = np.mean(total_loss)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {mean_loss:.4f}")
            history["loss"].append(mean_loss)
        return history

    def sample(self, n):
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size), device=self.device)
            for t in reversed(range(self.T)):
                t_tensor = torch.tensor([t], device=self.device).long()
                alpha_bar_t = self.alpha_bars[t_tensor].view(-1, 1, 1, 1)
                if t > 0:
                    alpha_bar_t_prev = self.alpha_bars[t_tensor - 1].view(-1, 1, 1, 1)
                else:
                    alpha_bar_t_prev = torch.tensor(1.0, device=self.device).view(-1, 1, 1, 1)
                pred_noise = self.unet(x, t_tensor)
                x = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                if t > 0:
                    x = x * alpha_bar_t_prev.sqrt() + torch.randn_like(x) * (1 - alpha_bar_t_prev).sqrt()
            return x

