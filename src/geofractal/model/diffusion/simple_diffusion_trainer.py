# ============================================================================
# Flow Matching V-Pred - CIFAR-10 Training
# Single cell for Colab
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LR = 2e-4
IMAGE_SIZE = 32
BASE_CHANNELS = 64
SAMPLE_STEPS = 50

print(f"Device: {DEVICE}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                          persistent_workers=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim * 4
        self.mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        return self.mlp(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, out_ch))
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.gelu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.gelu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x).view(b, c, -1).permute(0, 2, 1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + attn_out.permute(0, 2, 1).view(b, c, h, w)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, channel_mults=(1, 2, 4), dropout=0.1):
        super().__init__()

        time_dim = base_channels * 4
        self.time_emb = TimeEmbedding(base_channels, time_dim)

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        ch = base_channels
        self.down1 = nn.ModuleList([
            ResBlock(ch, ch * channel_mults[0], time_dim, dropout),
            ResBlock(ch * channel_mults[0], ch * channel_mults[0], time_dim, dropout),
        ])
        self.pool1 = nn.Conv2d(ch * channel_mults[0], ch * channel_mults[0], 3, stride=2, padding=1)

        self.down2 = nn.ModuleList([
            ResBlock(ch * channel_mults[0], ch * channel_mults[1], time_dim, dropout),
            ResBlock(ch * channel_mults[1], ch * channel_mults[1], time_dim, dropout),
        ])
        self.pool2 = nn.Conv2d(ch * channel_mults[1], ch * channel_mults[1], 3, stride=2, padding=1)

        self.down3 = nn.ModuleList([
            ResBlock(ch * channel_mults[1], ch * channel_mults[2], time_dim, dropout),
            ResBlock(ch * channel_mults[2], ch * channel_mults[2], time_dim, dropout),
        ])

        # Middle
        mid_ch = ch * channel_mults[2]
        self.mid = nn.ModuleList([
            ResBlock(mid_ch, mid_ch, time_dim, dropout),
            AttentionBlock(mid_ch),
            ResBlock(mid_ch, mid_ch, time_dim, dropout),
        ])

        # Decoder
        self.up3 = nn.ModuleList([
            ResBlock(mid_ch * 2, ch * channel_mults[1], time_dim, dropout),
            ResBlock(ch * channel_mults[1], ch * channel_mults[1], time_dim, dropout),
        ])
        self.upsample3 = nn.ConvTranspose2d(ch * channel_mults[1], ch * channel_mults[1], 4, stride=2, padding=1)

        self.up2 = nn.ModuleList([
            ResBlock(ch * channel_mults[1] * 2, ch * channel_mults[0], time_dim, dropout),
            ResBlock(ch * channel_mults[0], ch * channel_mults[0], time_dim, dropout),
        ])
        self.upsample2 = nn.ConvTranspose2d(ch * channel_mults[0], ch * channel_mults[0], 4, stride=2, padding=1)

        self.up1 = nn.ModuleList([
            ResBlock(ch * channel_mults[0] * 2, ch, time_dim, dropout),
            ResBlock(ch, ch, time_dim, dropout),
        ])

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        h = self.conv_in(x)

        # Down
        h1 = h
        for block in self.down1:
            h1 = block(h1, t_emb)

        h2 = self.pool1(h1)
        for block in self.down2:
            h2 = block(h2, t_emb)

        h3 = self.pool2(h2)
        for block in self.down3:
            h3 = block(h3, t_emb)

        # Mid
        h = h3
        for block in self.mid:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Up
        h = torch.cat([h, h3], dim=1)
        for block in self.up3:
            h = block(h, t_emb)
        h = self.upsample3(h)

        h = torch.cat([h, h2], dim=1)
        for block in self.up2:
            h = block(h, t_emb)
        h = self.upsample2(h)

        h = torch.cat([h, h1], dim=1)
        for block in self.up1:
            h = block(h, t_emb)

        return self.conv_out(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Flow Matching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FlowMatching(nn.Module):
    def __init__(self, model, image_size=32, channels=3):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels

    def forward(self, x_0):
        """Training: returns loss."""
        batch_size = x_0.shape[0]
        device = x_0.device

        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(x_0)

        t_exp = t[:, None, None, None]
        x_t = (1 - t_exp) * x_0 + t_exp * noise
        v_target = noise - x_0
        v_pred = self.model(x_t, t)

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, batch_size=16, steps=50, device="cuda"):
        """Euler sampling from t=1 (noise) to t=0 (image)."""
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        timesteps = torch.linspace(1, 0, steps + 1, device=device)

        for i in range(steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - t  # Negative (going 1→0)
            t_batch = torch.full((batch_size,), t, device=device)
            v = self.model(x, t_batch)
            x = x + dt * v  # dt negative moves toward x_0

        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model = SimpleUNet(
    in_channels=3,
    base_channels=BASE_CHANNELS,
    channel_mults=(1, 2, 4),
    dropout=0.1,
).to(DEVICE)

flow = FlowMatching(model, image_size=IMAGE_SIZE, channels=3)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
history = {'loss': []}

print(f"\nTraining for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
    for images, _ in pbar:
        images = images.to(DEVICE)

        optimizer.zero_grad()
        loss = flow(images)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    scheduler.step()

    avg_loss = epoch_loss / len(train_loader)
    history['loss'].append(avg_loss)

    # Sample every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        samples = flow.sample(batch_size=16, steps=SAMPLE_STEPS, device=DEVICE)
        samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        samples = torch.clamp(samples, 0, 1)

        grid = make_grid(samples, nrow=4, padding=2)
        save_image(grid, f'samples_epoch_{epoch + 1}.png')

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f} | Saved samples")
    else:
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final Sampling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nGenerating final samples...")
model.eval()

samples = flow.sample(batch_size=64, steps=100, device=DEVICE)
samples = (samples + 1) / 2
samples = torch.clamp(samples, 0, 1)

grid = make_grid(samples, nrow=8, padding=2)
save_image(grid, 'final_samples.png')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualize
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(history['loss'])
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (Flow Matching V-Pred)')
axes[0].grid(True, alpha=0.3)

# Final samples
axes[1].imshow(grid.permute(1, 2, 0).cpu().numpy())
axes[1].set_title(f'Generated Samples (100 steps)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('flow_matching_training.png', dpi=150)
plt.show()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Save Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': EPOCHS,
    'loss': history['loss'][-1],
}, 'flow_matching_cifar10.pt')

print(f"\n✓ Done! Final loss: {history['loss'][-1]:.4f}")
print(f"Model saved to flow_matching_cifar10.pt")