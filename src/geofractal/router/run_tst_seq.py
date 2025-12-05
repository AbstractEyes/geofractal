"""
Test script for GeoFractal Router refactored components.

Tests:
1. Head components (CantorAttention, TopKRouter, Anchors, Gates)
2. HeadBuilder and ComposedHead
3. Fusion strategies
4. Factory specs
5. RouterCollective
"""

import torch
import torch.nn as nn
import sys

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Testing on: {DEVICE}")


def test_section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def test_pass(name: str):
    print(f"  ✓ {name}")


def test_fail(name: str, error: str):
    print(f"  ✗ {name}: {error}")
    return False


# =============================================================================
# TEST 1: Imports
# =============================================================================

def test_imports():
    test_section("1. Imports")

    errors = []

    try:
        from geofractal.router.config import GlobalFractalRouterConfig, CollectiveConfig
        test_pass("config imports")
    except ImportError as e:
        errors.append(test_fail("config imports", str(e)))

    try:
        from geofractal.router.registry import RouterRegistry, RouterMailbox, get_registry
        test_pass("registry imports")
    except ImportError as e:
        errors.append(test_fail("registry imports", str(e)))

    try:
        from geofractal.router.head.head_components import (
            cantor_pair, cantor_unpair, build_cantor_bias,
            HeadConfig, CantorAttention, TopKRouter, FingerprintGate,
            ConstitutiveAnchorBank, LearnableWeightCombiner, FFNRefinement,
        )
        test_pass("head.components imports")
    except ImportError as e:
        errors.append(test_fail("head.components imports", str(e)))

    try:
        from geofractal.router.head.head_builder import HeadBuilder, ComposedHead, build_standard_head
        test_pass("head.builder imports")
    except ImportError as e:
        errors.append(test_fail("head.builder imports", str(e)))

    try:
        from geofractal.router.streams import InputShape, BaseStream, StreamBuilder
        test_pass("streams imports")
    except ImportError as e:
        errors.append(test_fail("streams imports", str(e)))

    try:
        from geofractal.router.fusion import FusionBuilder, FusionStrategy
        test_pass("fusion imports")
    except ImportError as e:
        errors.append(test_fail("fusion imports", str(e)))

    try:
        from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec, PrototypeBuilder
        test_pass("factory imports")
    except ImportError as e:
        errors.append(test_fail("factory imports", str(e)))

    try:
        from geofractal.router.collective import RouterCollective
        test_pass("collective imports")
    except ImportError as e:
        errors.append(test_fail("collective imports", str(e)))

    try:
        from geofractal.router import GlobalFractalRouterConfig, RouterCollective, HeadBuilder, FusionBuilder, StreamSpec
        test_pass("router __init__ imports")
    except ImportError as e:
        errors.append(test_fail("router __init__ imports", str(e)))

    return len(errors) == 0


# =============================================================================
# TEST 2: Cantor Functions
# =============================================================================

def test_cantor_functions():
    test_section("2. Cantor Functions")

    from geofractal.router.head.head_components import cantor_pair, cantor_unpair, build_cantor_bias

    x = torch.tensor([0, 1, 2, 0, 1])
    y = torch.tensor([0, 0, 0, 1, 1])
    z = cantor_pair(x, y)

    assert len(z.unique()) == len(z), f"cantor_pair not bijective: {z}"
    test_pass(f"cantor_pair: {list(zip(x.tolist(), y.tolist()))} -> {z.tolist()}")

    x2, y2 = cantor_unpair(z)
    assert torch.equal(x, x2) and torch.equal(y, y2), f"cantor_unpair failed"
    test_pass("cantor_unpair (roundtrip verified)")

    height, width = 4, 4
    bias = build_cantor_bias(height, width, torch.device('cpu'))
    seq_len = height * width
    assert bias.shape == (seq_len, seq_len), f"Wrong shape: {bias.shape}"
    assert torch.allclose(bias.diagonal(), torch.ones(seq_len)), "Diagonal should be 1.0"
    test_pass(f"build_cantor_bias: {height}x{width} -> [{seq_len}, {seq_len}]")

    return True


# =============================================================================
# TEST 3: Head Components
# =============================================================================

def test_head_components():
    test_section("3. Head Components")

    from geofractal.router.head.head_components import (
        HeadConfig, CantorAttention, TopKRouter, FingerprintGate,
        ConstitutiveAnchorBank, LearnableWeightCombiner, FFNRefinement,
    )

    B, S, D, F = 4, 16, 256, 64
    config = HeadConfig(feature_dim=D, fingerprint_dim=F, num_heads=8, num_anchors=8, num_routes=4)

    x = torch.randn(B, S, D).to(DEVICE)
    fingerprint = torch.randn(F).to(DEVICE)

    # CantorAttention
    try:
        attn = CantorAttention(config).to(DEVICE)
        out, weights = attn(x)
        assert out.shape == (B, S, D), f"Wrong shape: {out.shape}"
        test_pass(f"CantorAttention: {x.shape} -> {out.shape}")
    except Exception as e:
        test_fail("CantorAttention", str(e))
        return False

    # TopKRouter
    try:
        router = TopKRouter(config).to(DEVICE)
        routes, weights, routed = router(x, x, x, fingerprint)
        assert routed.shape == (B, S, D), f"Wrong shape: {routed.shape}"
        test_pass(f"TopKRouter: {x.shape} -> {routed.shape}")
    except Exception as e:
        test_fail("TopKRouter", str(e))
        return False

    # FingerprintGate
    try:
        gate = FingerprintGate(config).to(DEVICE)
        gated = gate.gate_values(x, fingerprint)
        assert gated.shape == (B, S, D), f"Wrong shape: {gated.shape}"
        test_pass(f"FingerprintGate: {x.shape} -> {gated.shape}")
    except Exception as e:
        test_fail("FingerprintGate", str(e))
        return False

    # ConstitutiveAnchorBank
    try:
        anchors = ConstitutiveAnchorBank(config).to(DEVICE)
        anchor_out, affinities = anchors(x, fingerprint)
        assert anchor_out.shape == (B, S, D), f"Wrong shape: {anchor_out.shape}"
        test_pass(f"ConstitutiveAnchorBank: {x.shape} -> {anchor_out.shape}")
    except Exception as e:
        test_fail("ConstitutiveAnchorBank", str(e))
        return False

    # LearnableWeightCombiner
    try:
        combiner = LearnableWeightCombiner(config).to(DEVICE)
        signals = {'attention': x, 'routing': x * 0.5, 'anchors': x * 0.1}
        combined = combiner(signals)
        assert combined.shape == (B, S, D), f"Wrong shape: {combined.shape}"
        test_pass(f"LearnableWeightCombiner: dict -> {combined.shape}")
    except Exception as e:
        test_fail("LearnableWeightCombiner", str(e))
        return False

    # FFNRefinement
    try:
        ffn = FFNRefinement(config).to(DEVICE)
        refined = ffn(x)
        assert refined.shape == (B, S, D), f"Wrong shape: {refined.shape}"
        test_pass(f"FFNRefinement: {x.shape} -> {refined.shape}")
    except Exception as e:
        test_fail("FFNRefinement", str(e))
        return False

    return True


# =============================================================================
# TEST 4: HeadBuilder and ComposedHead
# =============================================================================

def test_head_builder():
    test_section("4. HeadBuilder and ComposedHead")

    from geofractal.router.head.head_builder import HeadBuilder, build_standard_head
    from geofractal.router.head.head_components import HeadConfig

    B, S, D = 4, 16, 512
    x = torch.randn(B, S, D).to(DEVICE)

    try:
        config = HeadConfig(feature_dim=D, fingerprint_dim=64, num_heads=8, num_anchors=16, num_routes=4)
        head = HeadBuilder(config).build().to(DEVICE)

        assert hasattr(head, 'fingerprint'), "Missing fingerprint"
        assert head.fingerprint.shape == (64,), f"Wrong fp shape: {head.fingerprint.shape}"

        # Default: returns just output
        out = head(x)
        assert out.shape == (B, S, D), f"Wrong output shape: {out.shape}"
        test_pass(f"HeadBuilder: {x.shape} -> {out.shape}")
    except Exception as e:
        test_fail("HeadBuilder", str(e))
        return False

    try:
        head2 = build_standard_head(HeadConfig(feature_dim=D)).to(DEVICE)
        out2 = head2(x)
        assert out2.shape == (B, S, D)
        test_pass("build_standard_head")
    except Exception as e:
        test_fail("build_standard_head", str(e))
        return False

    try:
        target_fp = torch.randn(64).to(DEVICE)
        out3 = head(x, target_fingerprint=target_fp)
        assert out3.shape == (B, S, D)
        test_pass("ComposedHead with target_fingerprint")
    except Exception as e:
        test_fail("ComposedHead with target_fingerprint", str(e))
        return False

    try:
        out4, info = head(x, return_info=True)
        assert out4.shape == (B, S, D)
        assert 'routes' in info
        test_pass("ComposedHead with return_info=True")
    except Exception as e:
        test_fail("ComposedHead with return_info", str(e))
        return False

    num_params = sum(p.numel() for p in head.parameters())
    test_pass(f"ComposedHead params: {num_params:,}")

    return True


# =============================================================================
# TEST 5: Fusion Strategies
# =============================================================================

def test_fusion():
    test_section("5. Fusion Strategies")

    from geofractal.router.fusion import FusionBuilder, FusionStrategy

    B, D = 4, 512
    stream_dims = {"clip": D, "dino": D, "t5": D}

    stream_outputs = {
        name: torch.randn(B, dim).to(DEVICE)
        for name, dim in stream_dims.items()
    }

    strategies = [
        FusionStrategy.CONCAT,
        FusionStrategy.WEIGHTED,
        FusionStrategy.GATED,
        FusionStrategy.ATTENTION,
        FusionStrategy.RESIDUAL,
    ]

    for strategy in strategies:
        try:
            fusion = (FusionBuilder()
                .with_streams(stream_dims)
                .with_output_dim(D)
                .with_strategy(strategy)
                .build()
                .to(DEVICE))

            # Fusion returns tuple (output, info)
            out, info = fusion(stream_outputs)
            assert out.shape == (B, D), f"Wrong shape: {out.shape}"
            test_pass(f"{strategy.name}: 3x[{B}, {D}] -> [{B}, {D}]")
        except Exception as e:
            test_fail(f"{strategy.name}", str(e))
            return False

    return True


# =============================================================================
# TEST 6: Factory Specs
# =============================================================================

def test_factory_specs():
    test_section("6. Factory Specs")

    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec

    try:
        spec1 = StreamSpec.feature_vector("clip", input_dim=512, feature_dim=256)
        assert spec1.name == "clip"
        assert spec1.input_dim == 512
        assert spec1.feature_dim == 256
        assert spec1.input_shape == "vector"
        test_pass("StreamSpec.feature_vector")

        spec2 = StreamSpec.sequence("t5", input_dim=768)
        assert spec2.input_shape == "sequence"
        test_pass("StreamSpec.sequence")

        spec3 = StreamSpec.transformer_sequence("bert", input_dim=768, num_layers=2)
        assert spec3.stream_type == "transformer_sequence"
        test_pass("StreamSpec.transformer_sequence")
    except Exception as e:
        test_fail("StreamSpec", str(e))
        return False

    try:
        h1 = HeadSpec.lightweight(feature_dim=256)
        h2 = HeadSpec.standard(feature_dim=512)
        h3 = HeadSpec.heavy(feature_dim=512)
        assert h1.num_anchors < h2.num_anchors <= h3.num_anchors
        test_pass("HeadSpec presets (lightweight, standard, heavy)")
    except Exception as e:
        test_fail("HeadSpec", str(e))
        return False

    try:
        f1 = FusionSpec.concat(output_dim=512)
        f2 = FusionSpec.gated(output_dim=512)
        f3 = FusionSpec.attention(output_dim=512)
        assert f1.strategy == "concat"
        assert f2.strategy == "gated"
        assert f3.strategy == "attention"
        test_pass("FusionSpec presets (concat, gated, attention)")
    except Exception as e:
        test_fail("FusionSpec", str(e))
        return False

    return True


# =============================================================================
# TEST 7: RouterCollective
# =============================================================================

def test_collective():
    test_section("7. RouterCollective")

    from geofractal.router.collective import RouterCollective
    from geofractal.router.config import CollectiveConfig
    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec

    B = 4

    config = CollectiveConfig(
        feature_dim=256,
        num_classes=10,
        device=DEVICE,
    )

    try:
        collective = RouterCollective.from_specs(
            stream_specs=[
                StreamSpec.feature_vector("stream_a", input_dim=512, feature_dim=256),
                StreamSpec.feature_vector("stream_b", input_dim=768, feature_dim=256),
            ],
            config=config,
            head_spec=HeadSpec.lightweight(feature_dim=256),
            fusion_spec=FusionSpec.gated(output_dim=256),
        )
        collective.to(DEVICE)
        test_pass("RouterCollective.from_specs")
    except Exception as e:
        test_fail("RouterCollective.from_specs", str(e))
        return False

    try:
        inputs = {
            "stream_a": torch.randn(B, 512).to(DEVICE),
            "stream_b": torch.randn(B, 768).to(DEVICE),
        }

        logits, info = collective(inputs, return_individual=True)

        assert logits.shape == (B, 10), f"Wrong logits shape: {logits.shape}"
        assert 'individual_logits' in info
        assert 'stream_a' in info['individual_logits']
        test_pass(f"Forward pass: {logits.shape}")
    except Exception as e:
        test_fail("Forward pass", str(e))
        return False

    try:
        labels = torch.randint(0, 10, (B,)).to(DEVICE)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        grad_count = sum(1 for p in collective.parameters() if p.grad is not None)
        test_pass(f"Backward pass: {grad_count} params with gradients")
    except Exception as e:
        test_fail("Backward pass", str(e))
        return False

    try:
        emergence = collective.compute_emergence(
            collective_acc=0.85,
            individual_accs={"stream_a": 0.10, "stream_b": 0.12},
        )
        assert emergence['rho'] > 1.0
        test_pass(f"Emergence: ρ = {emergence['rho']:.2f}")
    except Exception as e:
        test_fail("Emergence computation", str(e))
        return False

    try:
        summary = collective.summary()
        assert "stream_a" in summary
        test_pass("Summary generation")
        print(f"\n{summary}")
    except Exception as e:
        test_fail("Summary", str(e))
        return False

    return True


# =============================================================================
# TEST 8: End-to-End Training Step
# =============================================================================

def test_training_step():
    test_section("8. End-to-End Training Step")

    from geofractal.router.collective import RouterCollective
    from geofractal.router.config import CollectiveConfig
    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec

    B = 8
    num_classes = 10

    config = CollectiveConfig(
        feature_dim=128,
        num_classes=num_classes,
        device=DEVICE,
    )

    collective = RouterCollective.from_specs(
        stream_specs=[
            StreamSpec.feature_vector("a", input_dim=256, feature_dim=128),
            StreamSpec.feature_vector("b", input_dim=256, feature_dim=128),
            StreamSpec.feature_vector("c", input_dim=256, feature_dim=128),
        ],
        config=config,
        head_spec=HeadSpec.lightweight(feature_dim=128),
        fusion_spec=FusionSpec.gated(output_dim=128),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in collective.parameters() if p.requires_grad],
        lr=1e-3,
    )

    losses = []
    collective.train()

    for step in range(5):
        inputs = {
            "a": torch.randn(B, 256).to(DEVICE),
            "b": torch.randn(B, 256).to(DEVICE),
            "c": torch.randn(B, 256).to(DEVICE),
        }
        labels = torch.randint(0, num_classes, (B,)).to(DEVICE)

        optimizer.zero_grad()
        logits, info = collective(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    test_pass(f"5 training steps: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    collective.eval()
    with torch.no_grad():
        logits, info = collective(inputs, return_individual=True)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

    test_pass(f"Eval mode: acc = {acc*100:.1f}%")

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  GeoFractal Router Component Tests")
    print("="*60)

    results = {}

    results['imports'] = test_imports()

    if results['imports']:
        results['cantor'] = test_cantor_functions()
        results['components'] = test_head_components()
        results['head_builder'] = test_head_builder()
        results['fusion'] = test_fusion()
        results['factory'] = test_factory_specs()
        results['collective'] = test_collective()
        results['training'] = test_training_step()

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  {passed}/{total} tests passed")
    print("="*60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)