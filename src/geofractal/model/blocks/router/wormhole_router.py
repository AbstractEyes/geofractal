"""
wormhole_router.py - Unified router with mode switching

Author: AbstractPhil + Claude + ChatGPT + Gemini
License: MIT
Date: 12/2/25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class WormholeRouterConfig:
    dim: int = 512
    num_positions: int = 64
    num_wormholes: int = 8
    temperature: float = 0.1
    mode: Literal['learned', 'cantor', 'hybrid', 'local'] = 'hybrid'
    cantor_weight: float = 0.3
    local_window: int = 3
    learnable_bias: bool = True


class WormholeRouter(nn.Module):
    """
    Unified wormhole router with multiple routing strategies.

    Modes:
        - 'learned': Pure content-based Q·K routing
        - 'cantor': Pure geometric routing via Cantor pairing
        - 'hybrid': Content + Cantor prior (default)
        - 'local': Content restricted to local spatial window
    """

    def __init__(self, config: WormholeRouterConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_positions = config.num_positions
        self.num_wormholes = min(config.num_wormholes, config.num_positions - 1)
        self.temperature = config.temperature
        self.mode = config.mode
        self.grid_size = int(math.sqrt(config.num_positions))

        # Content projections (all modes except pure cantor)
        if config.mode != 'cantor':
            self.query_proj = nn.Linear(config.dim, config.dim)
            self.key_proj = nn.Linear(config.dim, config.dim)
        else:
            self.query_proj = None
            self.key_proj = None

        # Position bias (hybrid and cantor modes)
        if config.mode in ('hybrid', 'cantor'):
            bias = self._build_cantor_bias()
            if config.learnable_bias and config.mode == 'hybrid':
                self.position_bias = nn.Parameter(bias * config.cantor_weight)
            else:
                self.register_buffer('position_bias', bias * config.cantor_weight)
        else:
            self.position_bias = None

        # Local window mask
        if config.mode == 'local':
            mask = self._build_local_mask(config.local_window)
            self.register_buffer('local_mask', mask)
        else:
            self.local_mask = None

    def _build_cantor_bias(self) -> torch.Tensor:
        P, G = self.num_positions, self.grid_size
        x = torch.arange(P) % G
        y = torch.arange(P) // G
        z = ((x + y) * (x + y + 1)) // 2 + y
        z = z.float() / z.max().clamp(min=1)

        dist = (z.unsqueeze(0) - z.unsqueeze(1)).abs()
        affinity = 1.0 - dist
        affinity.fill_diagonal_(-1e9)
        return affinity

    def _build_local_mask(self, window: int) -> torch.Tensor:
        P, G = self.num_positions, self.grid_size
        mask = torch.ones(P, P, dtype=torch.bool)

        for i in range(P):
            xi, yi = i % G, i // G
            for j in range(P):
                xj, yj = j % G, j // G
                if abs(xi - xj) <= window and abs(yi - yj) <= window:
                    mask[i, j] = False
        return mask

    def _compute_content_scores(self, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.query_proj(x), dim=-1)
        k = F.normalize(self.key_proj(x), dim=-1)
        return torch.bmm(q, k.transpose(1, 2))

    def forward(
            self,
            x: torch.Tensor,
            skip_first: bool = True,
            return_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, S, D] input features
            skip_first: Skip first token (CLS)
            return_scores: Return raw [B, P, P] scores
        Returns:
            routes: [B, P, K] destination indices
            weights: [B, P, K] normalized weights
            scores: Optional raw scores
        """
        if skip_first:
            x = x[:, 1:, :]

        B, P, D = x.shape

        # Compute scores based on mode
        if self.mode == 'cantor':
            scores = self.position_bias[:P, :P].unsqueeze(0).expand(B, -1, -1).clone()

        elif self.mode == 'learned':
            scores = self._compute_content_scores(x)

        elif self.mode == 'hybrid':
            scores = self._compute_content_scores(x)
            scores = scores + self.position_bias[:P, :P].unsqueeze(0)

        elif self.mode == 'local':
            scores = self._compute_content_scores(x)
            scores = scores.masked_fill(self.local_mask[:P, :P].unsqueeze(0), -1e9)

        # Mask self-connections
        eye_mask = torch.eye(P, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(eye_mask.unsqueeze(0), -1e9)

        # Top-K selection
        topk_scores, routes = torch.topk(
            scores / self.temperature,
            self.num_wormholes,
            dim=-1
        )
        weights = F.softmax(topk_scores, dim=-1)

        return routes, weights, scores if return_scores else None

    # =========================================================================
    # GATHER UTILITIES (as methods for convenience)
    # =========================================================================

    @staticmethod
    def gather(x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        """[B, P, D] + [B, P, K] → [B, P, K, D]"""
        B, P, D = x.shape
        K = routes.shape[-1]
        routes_flat = routes.reshape(B, P * K).unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(x, 1, routes_flat).view(B, P, K, D)

    @staticmethod
    def gather_multihead(x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        """[B, H, P, D] + [B, P, K] → [B, H, P, K, D]"""
        B, H, P, D = x.shape
        K = routes.shape[-1]
        x_flat = x.reshape(B * H, P, D)
        routes_exp = routes.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, P * K)
        routes_exp = routes_exp.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(x_flat, 1, routes_exp).view(B, H, P, K, D)


# =============================================================================
# QUICK TEST
# =============================================================================

def test_router():
    print("Testing WormholeRouter modes...")

    for mode in ['learned', 'cantor', 'hybrid', 'local']:
        config = WormholeRouterConfig(
            dim=256,
            num_positions=64,
            num_wormholes=8,
            mode=mode,
        )
        router = WormholeRouter(config)

        x = torch.randn(2, 65, 256)  # [B, 1+P, D]
        routes, weights, scores = router(x, return_scores=True)

        print(f"  {mode:8s}: routes={routes.shape}, weights={weights.shape}")
        assert routes.shape == (2, 64, 8)
        assert weights.shape == (2, 64, 8)

    print("✓ All modes passed!")


if __name__ == '__main__':
    test_router()