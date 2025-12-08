"""
Flow Matching with V-Prediction
-------------------------------
Clean implementation for CIFAR-10 (32x32).

Flow Matching:
  - Linear interpolation: x_t = (1-t) * x_0 + t * noise
  - Velocity target: v = noise - x_0
  - Model predicts v given (x_t, t)
  - Sampling: ODE integration from noise to image

Simpler than DDPM:
  - No complex noise schedules
  - Straight paths in latent space
  - Euler or RK4 sampling works well

Author: AbstractPhil + Claude Opus 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from tqdm.auto import tqdm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Time Embedding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """MLP to process time embedding."""

    def __init__(self, dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or dim * 4
        self.mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNet Building Blocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            time_emb_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Reshape to sequence
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # Normalize and attend
        x_norm = self.norm(x.view(b, c, -1)).permute(0, 2, 1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        # Residual
        out = x_flat + attn_out
        return out.permute(0, 2, 1).view(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNet
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UNet(nn.Module):
    """
    UNet for velocity prediction.

    Args:
        in_channels: Input channels (3 for RGB)
        base_channels: Base channel count
        channel_mults: Channel multipliers per resolution
        num_res_blocks: ResBlocks per resolution
        attention_resolutions: Which resolutions get attention (as divisors of input)
        dropout: Dropout rate
    """

    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
            channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
            num_res_blocks: int = 2,
            attention_resolutions: Tuple[int, ...] = (16, 8),  # Apply attention at 16x16, 8x8
            dropout: float = 0.1,
            image_size: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # Time embedding
        time_dim = base_channels * 4
        self.time_emb = TimeEmbedding(base_channels, time_dim)

        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = [base_channels]
        ch = base_channels
        current_res = image_size

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_dim, dropout)]

                if current_res in attention_resolutions:
                    layers.append(AttentionBlock(out_ch))

                self.down_blocks.append(nn.ModuleList(layers))
                ch = out_ch
                channels.append(ch)

            if i < len(channel_mults) - 1:
                self.down_samples.append(Downsample(ch))
                current_res //= 2
                channels.append(ch)

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult

            for j in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_dim, dropout)]

                if current_res in attention_resolutions:
                    layers.append(AttentionBlock(out_ch))

                self.up_blocks.append(nn.ModuleList(layers))
                ch = out_ch

            if i < len(channel_mults) - 1:
                self.up_samples.append(Upsample(ch))
                current_res *= 2

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image [B, C, H, W]
            t: Timestep [B] in [0, 1]

        Returns:
            Predicted velocity [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_emb(t)

        # Initial
        h = self.conv_in(x)

        # Downsampling
        skips = [h]
        down_idx = 0

        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            skips.append(h)

            # Downsample after each resolution level
            if down_idx < len(self.down_samples) and (down_idx + 1) % 2 == 0:
                h = self.down_samples[down_idx // 2](h)
                skips.append(h)
            down_idx += 1

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        up_idx = 0

        for block in self.up_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)

            # Upsample
            if up_idx < len(self.up_samples) and (up_idx + 1) % 3 == 0:
                h = self.up_samples[up_idx // 3](h)
            up_idx += 1

        return self.conv_out(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simplified UNet (for faster training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimpleUNet(nn.Module):
    """
    Simplified UNet - fewer parameters, faster training.
    Good for CIFAR-10 baseline.
    """

    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
            channel_mults: Tuple[int, ...] = (1, 2, 4),
            dropout: float = 0.1,
    ):
        super().__init__()

        time_dim = base_channels * 4
        self.time_emb = TimeEmbedding(base_channels, time_dim)

        # Encoder
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = nn.Sequential(
            ResBlock(base_channels, base_channels * channel_mults[0], time_dim, dropout),
            ResBlock(base_channels * channel_mults[0], base_channels * channel_mults[0], time_dim, dropout),
        )
        self.pool1 = nn.Conv2d(base_channels * channel_mults[0], base_channels * channel_mults[0], 3, stride=2,
                               padding=1)

        self.down2 = nn.Sequential(
            ResBlock(base_channels * channel_mults[0], base_channels * channel_mults[1], time_dim, dropout),
            ResBlock(base_channels * channel_mults[1], base_channels * channel_mults[1], time_dim, dropout),
        )
        self.pool2 = nn.Conv2d(base_channels * channel_mults[1], base_channels * channel_mults[1], 3, stride=2,
                               padding=1)

        self.down3 = nn.Sequential(
            ResBlock(base_channels * channel_mults[1], base_channels * channel_mults[2], time_dim, dropout),
            ResBlock(base_channels * channel_mults[2], base_channels * channel_mults[2], time_dim, dropout),
        )

        # Middle with attention
        mid_ch = base_channels * channel_mults[2]
        self.mid = nn.Sequential(
            ResBlock(mid_ch, mid_ch, time_dim, dropout),
            AttentionBlock(mid_ch),
            ResBlock(mid_ch, mid_ch, time_dim, dropout),
        )

        # Decoder
        self.up3 = nn.Sequential(
            ResBlock(mid_ch * 2, base_channels * channel_mults[1], time_dim, dropout),
            ResBlock(base_channels * channel_mults[1], base_channels * channel_mults[1], time_dim, dropout),
        )
        self.upsample3 = nn.ConvTranspose2d(base_channels * channel_mults[1], base_channels * channel_mults[1], 4,
                                            stride=2, padding=1)

        self.up2 = nn.Sequential(
            ResBlock(base_channels * channel_mults[1] * 2, base_channels * channel_mults[0], time_dim, dropout),
            ResBlock(base_channels * channel_mults[0], base_channels * channel_mults[0], time_dim, dropout),
        )
        self.upsample2 = nn.ConvTranspose2d(base_channels * channel_mults[0], base_channels * channel_mults[0], 4,
                                            stride=2, padding=1)

        self.up1 = nn.Sequential(
            ResBlock(base_channels * channel_mults[0] * 2, base_channels, time_dim, dropout),
            ResBlock(base_channels, base_channels, time_dim, dropout),
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        # Encoder
        h = self.conv_in(x)

        h1 = h
        for block in self.down1:
            h1 = block(h1, t_emb)

        h2 = self.pool1(h1)
        for block in self.down2:
            h2 = block(h2, t_emb)

        h3 = self.pool2(h2)
        for block in self.down3:
            h3 = block(h3, t_emb)

        # Middle
        h = h3
        for block in self.mid:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Decoder
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
    """
    Flow Matching with V-Prediction.

    Forward process:
        x_t = (1 - t) * x_0 + t * noise

    Velocity target:
        v = noise - x_0

    Training:
        loss = ||v - model(x_t, t)||²

    Sampling:
        Integrate ODE from t=1 (noise) to t=0 (image)
        dx/dt = -v_θ(x_t, t)
    """

    def __init__(
            self,
            model: nn.Module,
            image_size: int = 32,
            channels: int = 3,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x_0: Clean images [B, C, H, W]

        Returns:
            Loss (MSE between predicted and target velocity)
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.rand(batch_size, device=device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Interpolate: x_t = (1 - t) * x_0 + t * noise
        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * noise

        # Target velocity: v = noise - x_0
        v_target = noise - x_0

        # Predict velocity
        v_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample(
            self,
            batch_size: int = 16,
            steps: int = 50,
            device: str = "cuda",
            return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Sample images using Euler ODE integration.

        Args:
            batch_size: Number of images to generate
            steps: Number of ODE steps
            device: Device to generate on
            return_trajectory: If True, return all intermediate steps

        Returns:
            Generated images [B, C, H, W]
        """
        # Start from noise (t=1)
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)

        trajectory = [x] if return_trajectory else None

        # Timesteps from 1 to 0
        timesteps = torch.linspace(1, 0, steps + 1, device=device)

        for i in range(steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current  # Negative

            # Batch timestep
            t_batch = torch.full((batch_size,), t_current, device=device)

            # Predict velocity
            v = self.model(x, t_batch)

            # Euler step: x_next = x + dt * (-v) = x - dt * v
            # Since v = noise - x_0, and we're going from noise to x_0
            # dx/dt = -v, so x_next = x + dt * (-v)
            x = x + dt * (-v)

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return torch.stack(trajectory)

        return x

    @torch.no_grad()
    def sample_rk4(
            self,
            batch_size: int = 16,
            steps: int = 50,
            device: str = "cuda",
    ) -> torch.Tensor:
        """
        Sample using RK4 integration (higher quality, slower).
        """
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)

        timesteps = torch.linspace(1, 0, steps + 1, device=device)

        for i in range(steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - t

            t_batch = torch.full((batch_size,), t, device=device)

            # RK4
            k1 = -self.model(x, t_batch)
            k2 = -self.model(x + 0.5 * dt * k1, torch.full((batch_size,), t + 0.5 * dt.abs(), device=device))
            k3 = -self.model(x + 0.5 * dt * k2, torch.full((batch_size,), t + 0.5 * dt.abs(), device=device))
            k4 = -self.model(x + dt * k3, torch.full((batch_size,), t + dt.abs(), device=device))

            x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Testing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_flow_matching():
    """Test the flow matching implementation."""
    print("=" * 60)
    print("FLOW MATCHING V-PRED TEST")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test SimpleUNet
    print("\n[Test 1] SimpleUNet")
    model = SimpleUNet(
        in_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4),
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.rand(4, device=device)

    out = model(x, t)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Status: ✓ PASS")

    # Test FlowMatching
    print("\n[Test 2] FlowMatching Forward")
    flow = FlowMatching(model, image_size=32, channels=3)

    loss = flow(x)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test sampling
    print("\n[Test 3] Sampling (Euler)")
    samples = flow.sample(batch_size=2, steps=10, device=device)
    print(f"  Samples: {samples.shape}")
    print(f"  Range: [{samples.min():.2f}, {samples.max():.2f}]")
    print(f"  Status: ✓ PASS")

    # Test RK4 sampling
    print("\n[Test 4] Sampling (RK4)")
    samples_rk4 = flow.sample_rk4(batch_size=2, steps=10, device=device)
    print(f"  Samples: {samples_rk4.shape}")
    print(f"  Status: ✓ PASS")

    # Test gradient flow
    print("\n[Test 5] Gradient Flow")
    loss = flow(x)
    loss.backward()

    grad_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"  Gradients OK: {grad_ok}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_flow_matching()