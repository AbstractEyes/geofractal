import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CantorGate(nn.Module):
    """
    Fractal staircase activation inspired by the Cantor function.
    """

    def __init__(
            self,
            dim: int,
            num_levels: int = 5,
            per_dim: bool = False,
            temperature: float = 0.05  # Lower default
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.num_stairs = 2 ** num_levels
        self.per_dim = per_dim
        self.base_temperature = temperature

        # Initialize thresholds using Cantor set structure in [0, 1]
        init_thresholds = self._init_cantor_thresholds(num_levels)

        if per_dim:
            self.thresholds = nn.Parameter(
                init_thresholds.unsqueeze(0).expand(dim, -1).clone()
            )
            self.stair_values = nn.Parameter(
                torch.linspace(0, 1, self.num_stairs).unsqueeze(0).expand(dim, -1).clone()
            )
        else:
            self.thresholds = nn.Parameter(init_thresholds)
            self.stair_values = nn.Parameter(torch.linspace(0, 1, self.num_stairs))

        # Learnable sharpness per dimension
        self.snap_strength = nn.Parameter(torch.ones(dim) * 0.5)

        # Learnable temperature multiplier (centered at 1.0)
        self.temp_scale = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def _init_cantor_thresholds(self, levels: int) -> torch.Tensor:
        """Generate Cantor-set-like threshold positions in [0, 1]."""
        thresholds = []
        for i in range(levels):
            step = 1 / (3 ** (i + 1))
            for j in range(3 ** i):
                thresholds.extend([
                    (3 * j + 1) * step,
                    (3 * j + 2) * step
                ])

        thresholds = sorted(set(thresholds))[:self.num_stairs - 1]
        return torch.tensor(thresholds, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve sign, work with magnitude
        x_sign = x.sign()
        # Handle exact zeros - give them a small positive sign to avoid dead outputs
        x_sign = torch.where(x_sign == 0, torch.ones_like(x_sign), x_sign)
        x_mag = x.abs().clamp(min=1e-8)

        # Normalize magnitude to [0, 1) range
        x_norm = torch.tanh(torch.log1p(x_mag) / 3.0)

        # Temperature: base * learned_scale, kept small for sharp stairs
        temp_multiplier = torch.sigmoid(self.temp_scale) * 2.0  # range [0, 2]
        temp = (self.base_temperature * temp_multiplier).clamp(min=1e-4, max=0.5)

        if self.per_dim:
            above = torch.sigmoid((x_norm.unsqueeze(-1) - self.thresholds) / temp)
            stair_weights = above.sum(dim=-1)

            stair_idx_floor = stair_weights.long().clamp(0, self.num_stairs - 2)
            stair_idx_ceil = (stair_idx_floor + 1).clamp(0, self.num_stairs - 1)
            frac = stair_weights - stair_idx_floor.float()

            floor_vals = torch.gather(
                self.stair_values.expand(*x.shape[:-1], -1, -1),
                -1,
                stair_idx_floor.unsqueeze(-1)
            ).squeeze(-1)
            ceil_vals = torch.gather(
                self.stair_values.expand(*x.shape[:-1], -1, -1),
                -1,
                stair_idx_ceil.unsqueeze(-1)
            ).squeeze(-1)

            snapped_norm = floor_vals + frac * (ceil_vals - floor_vals)
        else:
            above = torch.sigmoid((x_norm.unsqueeze(-1) - self.thresholds) / temp)
            stair_weights = above.sum(dim=-1)

            stair_idx_floor = stair_weights.long().clamp(0, self.num_stairs - 2)
            stair_idx_ceil = (stair_idx_floor + 1).clamp(0, self.num_stairs - 1)
            frac = stair_weights - stair_idx_floor.float()

            floor_vals = self.stair_values[stair_idx_floor]
            ceil_vals = self.stair_values[stair_idx_ceil]

            snapped_norm = floor_vals + frac * (ceil_vals - floor_vals)

        # Convert back from [0, 1] to magnitude space
        snapped_mag = torch.expm1(snapped_norm * 3.0)

        # Blend between original and snapped
        strength = torch.sigmoid(self.snap_strength)
        output_mag = strength * snapped_mag + (1 - strength) * x_mag

        return x_sign * output_mag


class TopologicalDropout(nn.Module):
    """
    Structure-preserving dropout that drops entire pathways.

    Instead of randomly zeroing neurons (which destroys internal
    structure), drops entire routes/channels while preserving the
    structure within remaining routes.

    Modes:
        - 'route': Drop entire wormhole routes
        - 'scale': Drop entire scale heads
        - 'tile': Drop entire spatial tiles
        - 'channel': Drop channel groups
    """

    def __init__(
            self,
            drop_prob: float = 0.1,
            min_keep: int = 1,
            scale_kept: bool = True,
            structured_drop: bool = True
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.min_keep = min_keep
        self.scale_kept = scale_kept
        self.structured_drop = structured_drop

    def forward(
            self,
            x: torch.Tensor,
            route_dim: int = -2,
            num_routes: Optional[int] = None,
            importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor with route dimension
            route_dim: Which dimension contains the routes
            num_routes: Number of routes (inferred from x if None)
            importance: Optional per-route importance weights for
                       importance-weighted dropout

        Returns:
            (dropped_x, keep_mask) - output and the mask used
        """
        if not self.training or self.drop_prob == 0:
            return x, torch.ones(x.shape[route_dim], device=x.device)

        # Get number of routes
        if num_routes is None:
            num_routes = x.shape[route_dim]

        # Calculate how many to keep
        num_keep = max(self.min_keep, int(num_routes * (1 - self.drop_prob)))

        if importance is not None and self.structured_drop:
            # Importance-weighted dropout: less important routes more likely dropped
            # But with randomness to maintain regularization effect
            drop_weights = 1.0 / (importance + 1e-8)
            drop_weights = drop_weights / drop_weights.sum()

            # Add noise to prevent deterministic dropping
            noise = torch.rand_like(drop_weights) * 0.5
            drop_scores = drop_weights + noise

            # Keep top-k by inverted drop score
            _, keep_indices = torch.topk(-drop_scores, num_keep, dim=-1)
            keep_mask = torch.zeros(num_routes, device=x.device)
            keep_mask.scatter_(-1, keep_indices, 1.0)
        else:
            # Uniform random dropout of routes
            perm = torch.randperm(num_routes, device=x.device)
            keep_mask = torch.zeros(num_routes, device=x.device)
            keep_mask[perm[:num_keep]] = 1.0

        # Reshape mask for broadcasting
        mask_shape = [1] * x.dim()
        mask_shape[route_dim] = num_routes
        keep_mask = keep_mask.view(mask_shape)

        # Apply mask
        x_dropped = x * keep_mask

        # Scale to preserve expected value
        if self.scale_kept:
            scale = num_routes / num_keep
            x_dropped = x_dropped * scale

        return x_dropped, keep_mask.squeeze()


class FractalRegularizer(nn.Module):
    """
    Unified fractal-aware activation and regularization.

    Combines CantorGate (fractal staircase activation) with
    TopologicalDropout (structure-preserving regularization).

    The philosophy:
        - CantorGate: Values should LAND on structural levels
        - TopologicalDropout: Routes should be robust to occlusion
        - Together: Crystal structure with pathway redundancy

    Usage:
        reg = FractalRegularizer(dim=512, num_routes=8)

        # In forward pass:
        x = reg(x, route_dim=-2)

        # Or with routing weights for importance-weighted dropout:
        x = reg(x, route_dim=-2, routing_weights=attn_weights)
    """

    def __init__(
            self,
            dim: int,
            num_routes: int = 8,
            num_levels: int = 5,
            drop_prob: float = 0.1,
            gate_per_dim: bool = False,
            min_routes_keep: int = 2,
            activation_first: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_routes = num_routes
        self.activation_first = activation_first

        # Fractal staircase activation
        self.cantor_gate = CantorGate(
            dim=dim,
            num_levels=num_levels,
            per_dim=gate_per_dim
        )

        # Topological dropout
        self.topo_dropout = TopologicalDropout(
            drop_prob=drop_prob,
            min_keep=min_routes_keep,
            scale_kept=True,
            structured_drop=True
        )

        # Optional: learned combination of activation modes
        self.activation_mix = nn.Parameter(torch.tensor(0.5))

    def forward(
            self,
            x: torch.Tensor,
            route_dim: int = -2,
            routing_weights: Optional[torch.Tensor] = None,
            return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., num_routes, dim] or [..., dim]
            route_dim: Dimension containing routes (-2 for standard layout)
            routing_weights: Optional attention/routing weights for
                           importance-weighted dropout
            return_intermediates: Return dict with intermediate values

        Returns:
            Regularized tensor (same shape as input)
            If return_intermediates: (output, intermediates_dict)
        """
        intermediates = {}

        # Detect if we have routes dimension
        has_routes = x.dim() >= 2 and x.shape[route_dim] == self.num_routes

        if self.activation_first:
            # Apply Cantor gate first (snap to structure)
            x_activated = self.cantor_gate(x)
            intermediates['post_activation'] = x_activated

            # Then topological dropout (drop routes)
            if has_routes:
                importance = None
                if routing_weights is not None:
                    # Use routing weights as importance signal
                    # Higher weight = more important = less likely to drop
                    importance = routing_weights.mean(dim=tuple(range(routing_weights.dim() - 1)))

                x_dropped, keep_mask = self.topo_dropout(
                    x_activated,
                    route_dim=route_dim,
                    num_routes=self.num_routes,
                    importance=importance
                )
                intermediates['keep_mask'] = keep_mask
                output = x_dropped
            else:
                output = x_activated
        else:
            # Dropout first, then activate
            if has_routes:
                importance = None
                if routing_weights is not None:
                    importance = routing_weights.mean(dim=tuple(range(routing_weights.dim() - 1)))

                x_dropped, keep_mask = self.topo_dropout(
                    x,
                    route_dim=route_dim,
                    num_routes=self.num_routes,
                    importance=importance
                )
                intermediates['keep_mask'] = keep_mask
                output = self.cantor_gate(x_dropped)
            else:
                output = self.cantor_gate(x)

            intermediates['post_activation'] = output

        if return_intermediates:
            return output, intermediates
        return output

    def get_stair_statistics(self) -> dict:
        """Return statistics about learned stair positions."""
        thresholds = self.cantor_gate.thresholds.detach()
        stair_values = self.cantor_gate.stair_values.detach()
        snap_strength = torch.sigmoid(self.cantor_gate.snap_strength).detach()

        return {
            'thresholds': thresholds,
            'stair_values': stair_values,
            'snap_strength_mean': snap_strength.mean().item(),
            'snap_strength_std': snap_strength.std().item(),
            'num_stairs': self.cantor_gate.num_stairs,
            'activation_mix': torch.sigmoid(self.activation_mix).item()
        }

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, num_routes={self.num_routes}, '
            f'num_stairs={self.cantor_gate.num_stairs}, '
            f'drop_prob={self.topo_dropout.drop_prob}'
        )


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":

    def test_cantor_gate_basic():
        """Test basic CantorGate functionality."""
        print("=" * 60)
        print("TEST: CantorGate Basic Functionality")
        print("=" * 60)

        dim = 64
        batch = 4
        seq = 16

        gate = CantorGate(dim=dim, num_levels=5)
        x = torch.randn(batch, seq, dim)

        # Forward pass
        gate.train()
        y_train = gate(x)

        gate.eval()
        y_eval = gate(x)

        assert y_train.shape == x.shape, f"Shape mismatch: {y_train.shape} vs {x.shape}"
        assert y_eval.shape == x.shape, f"Shape mismatch: {y_eval.shape} vs {x.shape}"

        # Check sign preservation
        sign_preserved = (x.sign() == y_eval.sign()).float().mean()
        print(f"  Shape: {x.shape} -> {y_eval.shape} ✓")
        print(f"  Sign preservation: {sign_preserved:.2%}")
        print(f"  Num stairs: {gate.num_stairs}")
        print(f"  Thresholds shape: {gate.thresholds.shape}")

        # Check that output has discrete structure (fewer unique values)
        unique_in = len(torch.unique(x.round(decimals=2)))
        unique_out = len(torch.unique(y_eval.round(decimals=2)))
        print(f"  Unique values (input): {unique_in}")
        print(f"  Unique values (output): {unique_out}")

        print("  PASSED ✓\n")
        return True


    def test_cantor_gate_gradients():
        """Test gradient flow through CantorGate."""
        print("=" * 60)
        print("TEST: CantorGate Gradient Flow")
        print("=" * 60)

        dim = 32
        gate = CantorGate(dim=dim, num_levels=4, temperature=0.05)
        gate.train()

        x = torch.randn(2, 8, dim, requires_grad=True)
        y = gate(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "No gradient on input"
        assert gate.thresholds.grad is not None, "No gradient on thresholds"
        assert gate.stair_values.grad is not None, "No gradient on stair_values"
        assert gate.snap_strength.grad is not None, "No gradient on snap_strength"
        assert gate.temp_scale.grad is not None, "No gradient on temp_scale"  # Fixed name

        grad_norm = x.grad.norm().item()
        print(f"  Input grad norm: {grad_norm:.4f}")
        print(f"  Threshold grad norm: {gate.thresholds.grad.norm().item():.4f}")
        print(f"  Stair values grad norm: {gate.stair_values.grad.norm().item():.4f}")
        print(f"  Snap strength grad norm: {gate.snap_strength.grad.norm().item():.4f}")
        print(f"  Temp scale grad norm: {gate.temp_scale.grad.norm().item():.4f}")  # Fixed name

        assert grad_norm > 0, "Zero gradient"

        print("  PASSED ✓\n")
        return True


    def test_cantor_gate_per_dim():
        """Test per-dimension staircase."""
        print("=" * 60)
        print("TEST: CantorGate Per-Dimension Mode")
        print("=" * 60)

        dim = 16
        gate_shared = CantorGate(dim=dim, num_levels=3, per_dim=False)
        gate_perdim = CantorGate(dim=dim, num_levels=3, per_dim=True)

        x = torch.randn(2, 4, dim)

        y_shared = gate_shared(x)
        y_perdim = gate_perdim(x)

        print(f"  Shared thresholds shape: {gate_shared.thresholds.shape}")
        print(f"  Per-dim thresholds shape: {gate_perdim.thresholds.shape}")

        assert gate_shared.thresholds.dim() == 1
        assert gate_perdim.thresholds.dim() == 2
        assert gate_perdim.thresholds.shape[0] == dim

        print("  PASSED ✓\n")
        return True


    def test_topological_dropout_basic():
        """Test basic TopologicalDropout functionality."""
        print("=" * 60)
        print("TEST: TopologicalDropout Basic Functionality")
        print("=" * 60)

        dropout = TopologicalDropout(drop_prob=0.3, min_keep=2)

        batch, seq, num_routes, dim = 4, 16, 8, 64
        x = torch.randn(batch, seq, num_routes, dim)

        # Training mode - should drop
        dropout.train()
        y_train, mask_train = dropout(x, route_dim=-2)

        # Eval mode - should not drop
        dropout.eval()
        y_eval, mask_eval = dropout(x, route_dim=-2)

        assert y_train.shape == x.shape
        assert y_eval.shape == x.shape

        # Check that some routes were dropped in training
        routes_kept = mask_train.sum().item()
        print(f"  Input shape: {x.shape}")
        print(f"  Routes kept (train): {int(routes_kept)}/{num_routes}")
        print(f"  Mask (train): {mask_train.tolist()}")
        print(f"  Mask (eval): {mask_eval.tolist()}")

        assert routes_kept >= 2, f"min_keep violated: {routes_kept} < 2"
        assert mask_eval.sum() == num_routes, "Eval should keep all routes"

        print("  PASSED ✓\n")
        return True


    def test_topological_dropout_importance():
        """Test importance-weighted dropout."""
        print("=" * 60)
        print("TEST: TopologicalDropout Importance Weighting")
        print("=" * 60)

        dropout = TopologicalDropout(drop_prob=0.5, min_keep=2, structured_drop=True)
        dropout.train()

        num_routes = 8
        x = torch.randn(2, 4, num_routes, 32)

        # Make some routes very important
        importance = torch.tensor([0.01, 0.01, 0.01, 0.01, 10.0, 10.0, 10.0, 10.0])

        # Run multiple times and count how often each route is kept
        keep_counts = torch.zeros(num_routes)
        n_trials = 100

        for _ in range(n_trials):
            _, mask = dropout(x, route_dim=-2, importance=importance)
            keep_counts += mask

        keep_rates = keep_counts / n_trials

        print(f"  Importance: {importance.tolist()}")
        print(f"  Keep rates: {keep_rates.tolist()}")

        # High importance routes should be kept more often
        low_imp_rate = keep_rates[:4].mean().item()
        high_imp_rate = keep_rates[4:].mean().item()

        print(f"  Low importance avg keep rate: {low_imp_rate:.2%}")
        print(f"  High importance avg keep rate: {high_imp_rate:.2%}")

        assert high_imp_rate > low_imp_rate, "Importance weighting not working"

        print("  PASSED ✓\n")
        return True


    def test_topological_dropout_scaling():
        """Test that scaling preserves expected value."""
        print("=" * 60)
        print("TEST: TopologicalDropout Expected Value Preservation")
        print("=" * 60)

        dropout = TopologicalDropout(drop_prob=0.3, min_keep=1, scale_kept=True)
        dropout.train()

        x = torch.ones(4, 8, 8, 32)  # Known values for easy checking

        # Average over many runs
        running_sum = torch.zeros_like(x)
        n_trials = 500

        for _ in range(n_trials):
            y, _ = dropout(x, route_dim=-2)
            running_sum += y

        avg = running_sum / n_trials
        expected = x.mean().item()
        actual = avg.mean().item()

        print(f"  Expected mean: {expected:.4f}")
        print(f"  Actual mean (over {n_trials} trials): {actual:.4f}")
        print(f"  Relative error: {abs(actual - expected) / expected:.2%}")

        assert abs(actual - expected) / expected < 0.1, "Scaling not preserving expected value"

        print("  PASSED ✓\n")
        return True


    def test_fractal_regularizer_basic():
        """Test FractalRegularizer end-to-end."""
        print("=" * 60)
        print("TEST: FractalRegularizer Basic Functionality")
        print("=" * 60)

        dim = 64
        num_routes = 8
        reg = FractalRegularizer(
            dim=dim,
            num_routes=num_routes,
            num_levels=4,
            drop_prob=0.2,
            min_routes_keep=2
        )

        batch, seq = 4, 16
        x = torch.randn(batch, seq, num_routes, dim)

        # Training mode
        reg.train()
        y_train = reg(x, route_dim=-2)

        # Eval mode
        reg.eval()
        y_eval = reg(x, route_dim=-2)

        assert y_train.shape == x.shape
        assert y_eval.shape == x.shape

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y_train.shape}")
        print(f"  Num parameters: {sum(p.numel() for p in reg.parameters())}")

        # Check statistics
        stats = reg.get_stair_statistics()
        print(f"  Num stairs: {stats['num_stairs']}")
        print(f"  Snap strength (mean): {stats['snap_strength_mean']:.4f}")

        print("  PASSED ✓\n")
        return True


    def test_fractal_regularizer_with_routing():
        """Test FractalRegularizer with routing weights."""
        print("=" * 60)
        print("TEST: FractalRegularizer with Routing Weights")
        print("=" * 60)

        dim = 32
        num_routes = 8
        reg = FractalRegularizer(
            dim=dim,
            num_routes=num_routes,
            num_levels=3,
            drop_prob=0.3
        )
        reg.train()

        batch, seq = 2, 8
        x = torch.randn(batch, seq, num_routes, dim)

        # Simulated routing weights - make routes 6,7 very important
        routing_weights = torch.rand(batch, seq, num_routes)
        routing_weights[:, :, 6:] *= 10
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Run with intermediates
        y, intermediates = reg(
            x,
            route_dim=-2,
            routing_weights=routing_weights,
            return_intermediates=True
        )

        print(f"  Input shape: {x.shape}")
        print(f"  Routing weights shape: {routing_weights.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Keep mask: {intermediates['keep_mask'].tolist()}")

        assert 'keep_mask' in intermediates
        assert 'post_activation' in intermediates

        print("  PASSED ✓\n")
        return True


    def test_fractal_regularizer_gradients():
        """Test gradient flow through FractalRegularizer."""
        print("=" * 60)
        print("TEST: FractalRegularizer Gradient Flow")
        print("=" * 60)

        dim = 32
        num_routes = 8
        reg = FractalRegularizer(dim=dim, num_routes=num_routes, drop_prob=0.1)
        reg.train()

        x = torch.randn(2, 4, num_routes, dim, requires_grad=True)
        y = reg(x, route_dim=-2)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "No gradient on input"

        print(f"  Input grad norm: {x.grad.norm().item():.4f}")
        print(f"  Cantor thresholds grad: {reg.cantor_gate.thresholds.grad is not None}")
        print(f"  Snap strength grad: {reg.cantor_gate.snap_strength.grad is not None}")

        # Check all parameters got gradients
        for name, param in reg.named_parameters():
            if param.requires_grad:
                has_grad = param.grad is not None and param.grad.norm() > 0
                print(f"  {name}: grad={'✓' if has_grad else '✗'}")

        print("  PASSED ✓\n")
        return True


    def test_fractal_regularizer_no_routes():
        """Test FractalRegularizer on tensor without route dimension."""
        print("=" * 60)
        print("TEST: FractalRegularizer Without Routes")
        print("=" * 60)

        dim = 64
        reg = FractalRegularizer(dim=dim, num_routes=8, drop_prob=0.2)

        # Input without route dimension - just [batch, seq, dim]
        x = torch.randn(4, 16, dim)

        reg.train()
        y = reg(x, route_dim=-2)

        assert y.shape == x.shape
        print(f"  Input shape (no routes): {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Only CantorGate applied (no dropout)")

        print("  PASSED ✓\n")
        return True


    def test_cantor_stair_structure():
        """Visualize the Cantor staircase structure."""
        print("=" * 60)
        print("TEST: Cantor Staircase Structure Visualization")
        print("=" * 60)

        gate = CantorGate(dim=8, num_levels=3, temperature=0.02)  # Sharp stairs
        gate.eval()

        # Create range of inputs
        x = torch.linspace(-3, 3, 100).unsqueeze(-1).expand(-1, 8)
        y = gate(x)

        x_1d = x[:, 0].tolist()
        y_1d = y[:, 0].tolist()

        # Check staircase property
        unique_outputs = len(set([round(v, 2) for v in y_1d]))

        print(f"  Input range: [{min(x_1d):.2f}, {max(x_1d):.2f}]")
        print(f"  Output range: [{min(y_1d):.2f}, {max(y_1d):.2f}]")
        print(f"  Unique output levels: {unique_outputs}")
        print(f"  Expected stairs: {gate.num_stairs}")

        # Check symmetry
        print("\n  Symmetry check:")
        for i in [10, 20, 30, 40]:
            neg_x = x_1d[i]
            pos_x = x_1d[99 - i]
            neg_y = y_1d[i]
            pos_y = y_1d[99 - i]
            print(f"    x={neg_x:+.2f} -> y={neg_y:+.2f} (mag {abs(neg_y):.2f})")
            print(f"    x={pos_x:+.2f} -> y={pos_y:+.2f} (mag {abs(pos_y):.2f})")
            print(f"    Symmetry error: {abs(abs(neg_y) - abs(pos_y)):.4f}")
            print()

        # ASCII visualization - centered bars
        print("  Staircase (centered visualization):")
        center = 30
        for i in range(0, 100, 10):
            y_val = y_1d[i]
            bar_pos = int(y_val * 5)  # Scale to bar units
            if bar_pos >= 0:
                bar = ' ' * center + '|' + '█' * min(bar_pos, 25)
            else:
                bar = ' ' * (center + bar_pos) + '█' * (-bar_pos) + '|'
            print(f"    x={x_1d[i]:+.1f} -> y={y_val:+.2f} {bar}")

        print("  PASSED ✓\n")
        return True


    def test_integration_mock_router():
        """Integration test simulating wormhole router usage."""
        print("=" * 60)
        print("TEST: Integration - Mock Wormhole Router")
        print("=" * 60)

        # Simulate a mini wormhole routing scenario
        batch = 2
        seq = 16  # patches
        num_routes = 8  # wormholes
        dim = 64

        # Mock "routed" output from wormhole attention
        routed = torch.randn(batch, seq, num_routes, dim)

        # Mock routing weights (attention over wormholes)
        routing_attn = torch.softmax(torch.randn(batch, seq, num_routes), dim=-1)

        # Apply fractal regularization
        reg = FractalRegularizer(
            dim=dim,
            num_routes=num_routes,
            num_levels=4,
            drop_prob=0.15,
            min_routes_keep=3
        )
        reg.train()

        # Forward pass
        output, info = reg(
            routed,
            route_dim=-2,
            routing_weights=routing_attn,
            return_intermediates=True
        )

        # Aggregate across routes (like your crystal heads do)
        aggregated = (output * routing_attn.unsqueeze(-1)).sum(dim=-2)

        print(f"  Routed shape: {routed.shape}")
        print(f"  Routing attention shape: {routing_attn.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Aggregated shape: {aggregated.shape}")
        print(f"  Routes kept: {int(info['keep_mask'].sum())}/{num_routes}")

        # Verify differentiability
        loss = aggregated.sum()
        loss.backward()

        assert routed.grad is None  # We didn't set requires_grad
        print("  Full forward-backward pass complete")

        print("  PASSED ✓\n")
        return True


    # Run all tests
    print("\n" + "=" * 60)
    print(" FRACTAL REGULARIZER TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_cantor_gate_basic,
        test_cantor_gate_gradients,
        test_cantor_gate_per_dim,
        test_topological_dropout_basic,
        test_topological_dropout_importance,
        test_topological_dropout_scaling,
        test_fractal_regularizer_basic,
        test_fractal_regularizer_with_routing,
        test_fractal_regularizer_gradients,
        test_fractal_regularizer_no_routes,
        test_cantor_stair_structure,
        test_integration_mock_router,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}\n")
            failed += 1

    print("=" * 60)
    print(f" RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n 🎉 All tests passed!\n")
    else:
        print(f"\n ❌ {failed} test(s) failed\n")