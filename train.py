
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Time embedding function
def get_time_embedding(time_steps, temb_dim, spatial_size):
    factor = 10000 ** (torch.arange(temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    t_emb = time_steps[:, None].float() / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    t_emb = t_emb.view(-1, temb_dim, 1, 1).expand(-1, temb_dim, spatial_size, spatial_size)
    return t_emb


# Dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = [(os.path.join(root_dir, 'LR', f), os.path.join(root_dir, 'HR', f)) for f in os.listdir(os.path.join(root_dir, 'LR'))]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        lr_image_path, hr_image_path = self.image_pairs[idx]
        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = Image.open(hr_image_path).convert('RGB')

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

dataset = CustomDataset(root_dir='path/to/your/dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Self-Attention module
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

# Residual Block with GroupNorm
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(4, out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# U-Net model
class ConditionalUNet(nn.Module):
    def __init__(self):
        super(ConditionalUNet, self).__init__()
        self.enc1 = nn.Sequential(
            ResidualBlock(6, 16),
            ResidualBlock(16, 16),
            SelfAttention(16)
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(16, 64, stride=2),
            ResidualBlock(64, 64),
            SelfAttention(64)
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            SelfAttention(128)
        )
        self.enc4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            SelfAttention(256)
        )

        # Middle
        self.middle = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            SelfAttention(256)
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU()
        )
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 128), 
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 64),  
            nn.ReLU()
        )
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 64), 
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16), 
            nn.ReLU()
        )
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16), 
            nn.ReLU()
        )
        self.out_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, cond, t):
        # Get time embeddings
        t_emb_16 = get_time_embedding(t, 16, 512)  # For enc1 and dec1
        t_emb_64 = get_time_embedding(t, 64, 256)  # For enc2 and dec2
        t_emb_128 = get_time_embedding(t, 128, 128)  # For enc3 and dec3
        t_emb_256 = get_time_embedding(t, 256, 64)  # For enc4 and dec4

        x = torch.cat([x, cond], dim=1)
        e1 = self.enc1(x)
        e1 = e1 + t_emb_16

        e2 = self.enc2(e1)
        e2 = e2 + t_emb_64

        e3 = self.enc3(e2)
        e3 = e3 + t_emb_128

        e4 = self.enc4(e3)
        m = self.middle(e4)

        d4 = self.dec4(m)
        d4 = d4 + t_emb_256
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4_conv(d4)

        d3 = self.dec3(d4)
        d3 = d3 + t_emb_128
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3_conv(d3)

        d2 = self.dec2(d3)
        d2 = d2 + t_emb_64
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_conv(d2)

        d1 = self.dec1(d2)
        d1 = d1 + t_emb_16
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_conv(d1)

        out = self.out_conv(d1)
        return out


def add_noise(images, gamma_t):
    noise = torch.randn_like(images)
    noisy_images = torch.sqrt(gamma_t) * images + torch.sqrt(1-gamma_t) * noise
    return noisy_images, noise

def train_conditional_ddpm(model, dataloader, num_timesteps=1000, num_epochs=100, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
   
    beta = np.linspace(1e-4, 0.02, num_timesteps)
    beta = torch.tensor(beta, dtype=torch.float32).to(device)
    alpha = 1-beta
    gamma = torch.cumprod(alpha, dim=0)
   
    for epoch in range(num_epochs):
        for lr_images, hr_images in dataloader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            t = torch.randint(0, num_timesteps, (hr_images.shape[0],)).long().to(device)
            gamma_t = gamma[t].view(-1, 1, 1, 1)
            noisy_images, noise = add_noise(hr_images, gamma_t)
            optimizer.zero_grad()
            predicted_noise = model(noisy_images, lr_images, t)
            loss = mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
       
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

def denoise_image_with_condition(model, lr_image, hr_size=(512, 512), num_timesteps=1000):
    beta = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
    alpha = 1 - beta
    gamma = torch.cumprod(alpha, dim=0)

    lr_image_resized = nn.functional.interpolate(lr_image, size=hr_size, mode='bicubic', align_corners=False)
   
    image = torch.randn_like(lr_image_resized)
    for t in reversed(range(num_timesteps)):
        beta_t = beta[t]
        gamma_t = gamma[t]
        alpha_t = alpha[t]
        with torch.no_grad():
            predicted_noise = model(image, lr_image_resized)
            if predicted_noise.shape[-1] != image.shape[-1] or predicted_noise.shape[-2] != image.shape[-2]:
                predicted_noise = torch.nn.functional.interpolate(predicted_noise, size=image.shape[-2:], mode='bicubic', align_corners=False)
            image = (image - beta_t * predicted_noise / torch.sqrt(1 - gamma_t)) / torch.sqrt(alpha_t)
    return image



if __name__ == "__main__":
    # Initialize and train the conditional model
    model = ConditionalUNet().to(device)
    train_conditional_ddpm(model, dataloader, num_timesteps=1000, num_epochs=100, lr=2e-4)

    torch.save(model.state_dict(), "sr.pth")