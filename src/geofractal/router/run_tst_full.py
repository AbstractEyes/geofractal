"""
Comprehensive GeoFractal Router Component Tests

Tests ALL components in the refactored architecture:
1. Imports - all modules load without circular dependencies
2. Cantor functions - pairing, unpairing, bias matrix
3. Head components - attention, router, anchors, gates, combiners, refinement
4. HeadBuilder - config, presets, composed head
5. Streams - all stream types with slot expansion
6. Fusion - all 8 strategies
7. Factory specs - StreamSpec, HeadSpec, FusionSpec
8. Registry - RouterRegistry, RouterMailbox
9. RouterCollective - construction, forward, backward, emergence
10. Gradient flow - all critical components receive gradients
11. End-to-end training - multi-step optimization
12. Serialization - save/load state dict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import tempfile
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Testing on: {DEVICE}\n")

PASSED = 0
FAILED = 0
ERRORS = []


def test_section(name: str):
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")


def test_pass(name: str):
    global PASSED
    PASSED += 1
    print(f"  ✓ {name}")


def test_fail(name: str, error: str):
    global FAILED, ERRORS
    FAILED += 1
    ERRORS.append(f"{name}: {error}")
    print(f"  ✗ {name}: {error}")


def run_test(name: str, fn):
    """Run a test function and catch exceptions."""
    try:
        result = fn()
        if result is True or result is None:
            test_pass(name)
            return True
        else:
            test_fail(name, str(result))
            return False
    except Exception as e:
        test_fail(name, str(e))
        return False


# =============================================================================
# TEST 1: IMPORTS
# =============================================================================

def test_imports():
    test_section("1. Imports")

    # Config
    def import_config():
        from geofractal.router.config import GlobalFractalRouterConfig, CollectiveConfig
        return True

    run_test("config", import_config)

    # Registry
    def import_registry():
        from geofractal.router.registry import RouterRegistry, RouterMailbox, get_registry
        return True

    run_test("registry", import_registry)

    # Head components
    def import_head_components():
        from geofractal.router.head.components import (
            HeadConfig,
            cantor_pair, cantor_unpair, build_cantor_bias,
            StandardAttention, CantorAttention,
            TopKRouter, SoftRouter,
            ConstitutiveAnchorBank, AttentiveAnchorBank,
            FingerprintGate, ChannelGate,
            LearnableWeightCombiner, GatedCombiner,
            FFNRefinement, MixtureOfExpertsRefinement,
        )
        return True

    run_test("head.components", import_head_components)

    # Head builder
    def import_head_builder():
        from geofractal.router.head.builder import (
            HeadBuilder, HeadPreset, ComposedHead,
            STANDARD_HEAD, LIGHTWEIGHT_HEAD, HEAVY_HEAD,
            build_standard_head, build_lightweight_head, build_custom_head,
        )
        return True

    run_test("head.builder", import_head_builder)

    # Head protocols
    def import_head_protocols():
        from geofractal.router.head.protocols import (
            BaseAttention, BaseRouter, BaseAnchorBank,
            BaseGate, BaseCombiner, BaseRefinement,
        )
        return True

    run_test("head.protocols", import_head_protocols)

    # Streams
    def import_streams():
        from geofractal.router.streams.vector import (
            InputShape,
            FeatureVectorStream, TrainableVectorStream,
            SequenceStream, TransformerSequenceStream, ConvSequenceStream,
            StreamBuilder,
        )
        return True

    run_test("streams.vector", import_streams)

    # Fusion methods
    def import_fusion_methods():
        from geofractal.router.fusion.methods import (
            FusionConfig,
            ConcatFusion, WeightedFusion, GatedFusion,
            AttentionFusion, FingerprintGuidedFusion,
            ResidualFusion, MoEFusion, HierarchicalTreeFusion,
        )
        return True

    run_test("fusion.methods", import_fusion_methods)

    # Fusion builder
    def import_fusion_builder():
        from geofractal.router.fusion import FusionBuilder, FusionStrategy
        return True

    run_test("fusion.builder", import_fusion_builder)

    # Factory
    def import_factory():
        from geofractal.router.factory import (
            StreamSpec, HeadSpec, FusionSpec, PrototypeBuilder,
        )
        return True

    run_test("factory", import_factory)

    # Collective
    def import_collective():
        from geofractal.router.collective import RouterCollective
        return True

    run_test("collective", import_collective)

    # Top-level router __init__
    def import_router_init():
        from geofractal.router import (
            GlobalFractalRouterConfig,
            CollectiveConfig,
            RouterCollective,
            HeadBuilder,
            FusionBuilder,
            StreamSpec,
            HeadSpec,
            FusionSpec,
        )
        return True

    run_test("router __init__", import_router_init)


# =============================================================================
# TEST 2: CANTOR FUNCTIONS
# =============================================================================

def test_cantor_functions():
    test_section("2. Cantor Functions")

    from geofractal.router.head.components import cantor_pair, cantor_unpair, build_cantor_bias

    # Pairing bijection
    def test_pair_bijection():
        x = torch.tensor([0, 1, 2, 3, 0, 1, 2])
        y = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        z = cantor_pair(x, y)
        assert len(z.unique()) == len(z), f"Not bijective: {z}"
        return True

    run_test("cantor_pair bijection", test_pair_bijection)

    # Unpairing roundtrip
    def test_unpair_roundtrip():
        x = torch.tensor([0, 1, 2, 5, 10])
        y = torch.tensor([3, 2, 1, 0, 5])
        z = cantor_pair(x, y)
        x2, y2 = cantor_unpair(z)
        assert torch.equal(x, x2) and torch.equal(y, y2)
        return True

    run_test("cantor_unpair roundtrip", test_unpair_roundtrip)

    # Large values
    def test_large_values():
        x = torch.tensor([100, 200, 500])
        y = torch.tensor([50, 100, 250])
        z = cantor_pair(x, y)
        x2, y2 = cantor_unpair(z)
        assert torch.equal(x, x2) and torch.equal(y, y2)
        return True

    run_test("cantor large values", test_large_values)

    # Bias matrix shape
    def test_bias_shape():
        for h, w in [(4, 4), (8, 8), (7, 9), (16, 16)]:
            bias = build_cantor_bias(h, w, torch.device('cpu'))
            assert bias.shape == (h * w, h * w), f"Wrong shape for {h}x{w}: {bias.shape}"
        return True

    run_test("build_cantor_bias shapes", test_bias_shape)

    # Bias matrix properties
    def test_bias_properties():
        bias = build_cantor_bias(8, 8, torch.device('cpu'))
        # Diagonal should be 1.0 (max similarity to self)
        assert torch.allclose(bias.diagonal(), torch.ones(64)), "Diagonal not 1.0"
        # Should be symmetric
        assert torch.allclose(bias, bias.T), "Not symmetric"
        # Values in [0, 1]
        assert bias.min() >= 0 and bias.max() <= 1, f"Values out of range: [{bias.min()}, {bias.max()}]"
        return True

    run_test("build_cantor_bias properties", test_bias_properties)


# =============================================================================
# TEST 3: HEAD COMPONENTS
# =============================================================================

def test_head_components():
    test_section("3. Head Components")

    from geofractal.router.head.components import (
        HeadConfig,
        StandardAttention, CantorAttention,
        TopKRouter, SoftRouter,
        ConstitutiveAnchorBank, AttentiveAnchorBank,
        FingerprintGate, ChannelGate,
        LearnableWeightCombiner, GatedCombiner,
        FFNRefinement, MixtureOfExpertsRefinement,
    )

    B, S, D, F = 4, 16, 256, 64
    config = HeadConfig(feature_dim=D, fingerprint_dim=F, num_heads=8, num_anchors=8, num_routes=4)
    x = torch.randn(B, S, D).to(DEVICE)
    fingerprint = torch.randn(F).to(DEVICE)

    # StandardAttention
    def test_standard_attention():
        attn = StandardAttention(config).to(DEVICE)
        out, weights = attn(x, return_weights=True)
        assert out.shape == (B, S, D)
        assert weights.shape == (B, config.num_heads, S, S)
        return True

    run_test("StandardAttention", test_standard_attention)

    # CantorAttention
    def test_cantor_attention():
        attn = CantorAttention(config).to(DEVICE)
        out, weights = attn(x, return_weights=True)
        assert out.shape == (B, S, D)
        assert weights.shape == (B, config.num_heads, S, S)
        return True

    run_test("CantorAttention", test_cantor_attention)

    # TopKRouter
    def test_topk_router():
        router = TopKRouter(config).to(DEVICE)
        routes, weights, routed = router(x, x, x, fingerprint)
        assert routes.shape == (B, S, config.num_routes)
        assert weights.shape == (B, S, config.num_routes)
        assert routed.shape == (B, S, D)
        return True

    run_test("TopKRouter", test_topk_router)

    # SoftRouter
    def test_soft_router():
        router = SoftRouter(config).to(DEVICE)
        routes, weights, routed = router(x, x, x, fingerprint)
        assert routed.shape == (B, S, D)
        return True

    run_test("SoftRouter", test_soft_router)

    # ConstitutiveAnchorBank
    def test_constitutive_anchors():
        anchors = ConstitutiveAnchorBank(config).to(DEVICE)
        out, affinities = anchors(x, fingerprint)
        assert out.shape == (B, S, D)
        assert affinities.shape == (config.num_anchors,)
        return True

    run_test("ConstitutiveAnchorBank", test_constitutive_anchors)

    # AttentiveAnchorBank
    def test_attentive_anchors():
        anchors = AttentiveAnchorBank(config).to(DEVICE)
        out, affinities = anchors(x, fingerprint)
        assert out.shape == (B, S, D)
        return True

    run_test("AttentiveAnchorBank", test_attentive_anchors)

    # FingerprintGate
    def test_fingerprint_gate():
        gate = FingerprintGate(config).to(DEVICE)
        gated = gate.gate_values(x, fingerprint)
        assert gated.shape == (B, S, D)
        # Test compute_similarity
        fp2 = torch.randn(F).to(DEVICE)
        sim = gate.compute_similarity(fingerprint, fp2)
        assert sim.shape == () or sim.numel() == 1
        return True

    run_test("FingerprintGate", test_fingerprint_gate)

    # ChannelGate
    def test_channel_gate():
        gate = ChannelGate(config).to(DEVICE)
        gated = gate.gate_values(x, fingerprint)
        assert gated.shape == (B, S, D)
        return True

    run_test("ChannelGate", test_channel_gate)

    # LearnableWeightCombiner
    def test_learnable_combiner():
        combiner = LearnableWeightCombiner(config).to(DEVICE)
        signals = {'attention': x, 'routing': x * 0.5, 'anchors': x * 0.1}
        combined = combiner(signals)
        assert combined.shape == (B, S, D)
        return True

    run_test("LearnableWeightCombiner", test_learnable_combiner)

    # GatedCombiner
    def test_gated_combiner():
        combiner = GatedCombiner(config).to(DEVICE)
        signals = {'attention': x, 'routing': x * 0.5, 'anchors': x * 0.1}
        combined = combiner(signals)
        assert combined.shape == (B, S, D)
        return True

    run_test("GatedCombiner", test_gated_combiner)

    # FFNRefinement
    def test_ffn_refinement():
        ffn = FFNRefinement(config).to(DEVICE)
        refined = ffn(x)
        assert refined.shape == (B, S, D)
        return True

    run_test("FFNRefinement", test_ffn_refinement)

    # MixtureOfExpertsRefinement
    def test_moe_refinement():
        moe = MixtureOfExpertsRefinement(config, num_experts=4, top_k=2).to(DEVICE)
        refined = moe(x)
        assert refined.shape == (B, S, D)
        return True

    run_test("MixtureOfExpertsRefinement", test_moe_refinement)


# =============================================================================
# TEST 4: HEAD BUILDER
# =============================================================================

def test_head_builder():
    test_section("4. HeadBuilder and ComposedHead")

    from geofractal.router.head.builder import (
        HeadBuilder, ComposedHead,
        build_standard_head, build_lightweight_head, build_custom_head,
    )
    from geofractal.router.head.components import (
        HeadConfig, CantorAttention, TopKRouter,
        ConstitutiveAnchorBank, FingerprintGate,
    )

    B, S, D = 4, 16, 512
    x = torch.randn(B, S, D).to(DEVICE)
    config = HeadConfig(feature_dim=D)

    # Basic build
    def test_basic_build():
        head = HeadBuilder(config).build().to(DEVICE)
        assert isinstance(head, ComposedHead)
        assert hasattr(head, 'fingerprint')
        assert head.fingerprint.shape == (config.fingerprint_dim,)
        return True

    run_test("HeadBuilder basic build", test_basic_build)

    # Standard preset
    def test_standard_preset():
        head = HeadBuilder.standard(config).build().to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True

    run_test("HeadBuilder.standard preset", test_standard_preset)

    # Lightweight preset
    def test_lightweight_preset():
        head = HeadBuilder.lightweight(config).build().to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True

    run_test("HeadBuilder.lightweight preset", test_lightweight_preset)

    # Heavy preset
    def test_heavy_preset():
        head = HeadBuilder.heavy(config).build().to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True

    run_test("HeadBuilder.heavy preset", test_heavy_preset)

    # Factory functions
    def test_factory_functions():
        h1 = build_standard_head(config).to(DEVICE)
        h2 = build_lightweight_head(config).to(DEVICE)
        assert h1(x).shape == (B, S, D)
        assert h2(x).shape == (B, S, D)
        return True

    run_test("build_standard_head / build_lightweight_head", test_factory_functions)

    # Forward with target fingerprint
    def test_target_fingerprint():
        head = HeadBuilder(config).build().to(DEVICE)
        target_fp = torch.randn(config.fingerprint_dim).to(DEVICE)
        out = head(x, target_fingerprint=target_fp)
        assert out.shape == (B, S, D)
        return True

    run_test("ComposedHead with target_fingerprint", test_target_fingerprint)

    # Forward with return_info
    def test_return_info():
        head = HeadBuilder(config).build().to(DEVICE)
        out, info = head(x, return_info=True)
        assert out.shape == (B, S, D)
        assert 'routes' in info
        assert 'route_weights' in info
        assert 'anchor_affinities' in info
        return True

    run_test("ComposedHead with return_info", test_return_info)

    # Fluent API
    def test_fluent_api():
        head = (HeadBuilder(config)
                .with_attention(CantorAttention)
                .with_router(TopKRouter)
                .with_anchors(ConstitutiveAnchorBank)
                .with_gate(FingerprintGate)
                .build()
                .to(DEVICE))
        out = head(x)
        assert out.shape == (B, S, D)
        return True

    run_test("HeadBuilder fluent API", test_fluent_api)

    # Component access
    def test_component_access():
        head = HeadBuilder(config).build().to(DEVICE)
        assert head.get_component('attention') is head.attention
        assert head.get_component('router') is head.router
        assert head.get_component('anchors') is head.anchors
        return True

    run_test("ComposedHead component access", test_component_access)


# =============================================================================
# TEST 5: STREAMS
# =============================================================================

def test_streams():
    test_section("5. Stream Types")

    from geofractal.router.streams.vector import (
        InputShape,
        FeatureVectorStream, TrainableVectorStream,
        SequenceStream, TransformerSequenceStream, ConvSequenceStream,
        StreamBuilder,
    )

    B = 4

    # FeatureVectorStream
    def test_feature_vector_stream():
        stream = FeatureVectorStream(
            input_dim=512,
            feature_dim=256,
            num_slots=16,
        ).to(DEVICE)
        x = torch.randn(B, 512).to(DEVICE)
        out = stream(x)
        assert out.shape == (B, 16, 256), f"Got {out.shape}"
        assert stream.input_shape == InputShape.VECTOR
        # Test pool
        pooled = stream.pool(out)
        assert pooled.shape == (B, 256)
        return True

    run_test("FeatureVectorStream", test_feature_vector_stream)

    # TrainableVectorStream
    def test_trainable_vector_stream():
        stream = TrainableVectorStream(
            input_dim=768,
            feature_dim=256,
            num_slots=8,
        ).to(DEVICE)
        x = torch.randn(B, 768).to(DEVICE)
        out = stream(x)
        assert out.shape == (B, 8, 256)
        return True

    run_test("TrainableVectorStream", test_trainable_vector_stream)

    # SequenceStream
    def test_sequence_stream():
        stream = SequenceStream(
            input_dim=512,
            feature_dim=256,
        ).to(DEVICE)
        x = torch.randn(B, 32, 512).to(DEVICE)
        out = stream(x)
        assert out.shape == (B, 32, 256)
        assert stream.input_shape == InputShape.SEQUENCE
        return True

    run_test("SequenceStream", test_sequence_stream)

    # TransformerSequenceStream
    def test_transformer_sequence_stream():
        stream = TransformerSequenceStream(
            input_dim=512,
            feature_dim=256,
            num_layers=2,
            num_heads=4,
        ).to(DEVICE)
        x = torch.randn(B, 32, 512).to(DEVICE)
        out = stream(x)
        assert out.shape == (B, 32, 256)
        return True

    run_test("TransformerSequenceStream", test_transformer_sequence_stream)

    # ConvSequenceStream
    def test_conv_sequence_stream():
        stream = ConvSequenceStream(
            input_dim=512,
            feature_dim=256,
            kernel_sizes=(3, 5, 7),
        ).to(DEVICE)
        x = torch.randn(B, 32, 512).to(DEVICE)
        out = stream(x)
        assert out.shape == (B, 32, 256)
        return True

    run_test("ConvSequenceStream", test_conv_sequence_stream)

    # StreamBuilder - all types
    def test_stream_builder():
        types = [
            ('feature_vector', 512, 256, True),
            ('trainable_vector', 768, 256, True),
            ('sequence', 512, 256, False),
            ('transformer_sequence', 512, 256, False),
            ('conv_sequence', 512, 256, False),
        ]
        for stream_type, in_dim, out_dim, is_vector in types:
            stream = StreamBuilder.build(
                stream_type=stream_type,
                input_dim=in_dim,
                feature_dim=out_dim,
                num_slots=8,
            ).to(DEVICE)
            if is_vector:
                x = torch.randn(B, in_dim).to(DEVICE)
                out = stream(x)
                assert out.shape == (B, 8, out_dim), f"{stream_type}: {out.shape}"
            else:
                x = torch.randn(B, 16, in_dim).to(DEVICE)
                out = stream(x)
                assert out.shape == (B, 16, out_dim), f"{stream_type}: {out.shape}"
        return True

    run_test("StreamBuilder all types", test_stream_builder)

    # Slot embeddings are learnable
    def test_slot_embeddings_learnable():
        stream = FeatureVectorStream(512, 256, num_slots=16).to(DEVICE)
        x = torch.randn(B, 512).to(DEVICE)
        out = stream(x)
        loss = out.sum()
        loss.backward()
        assert stream.slot_embed.grad is not None
        assert stream.slot_embed.grad.abs().sum() > 0
        return True

    run_test("Slot embeddings receive gradients", test_slot_embeddings_learnable)


# =============================================================================
# TEST 6: FUSION STRATEGIES
# =============================================================================

def test_fusion():
    test_section("6. Fusion Strategies")

    from geofractal.router.fusion import FusionBuilder, FusionStrategy
    from geofractal.router.fusion.methods import (
        ConcatFusion, WeightedFusion, GatedFusion,
        AttentionFusion, ResidualFusion, MoEFusion,
        HierarchicalTreeFusion, FingerprintGuidedFusion,
    )

    B, D = 4, 512
    stream_dims = {"clip": D, "dino": D, "t5": D}
    stream_outputs = {
        name: torch.randn(B, dim).to(DEVICE)
        for name, dim in stream_dims.items()
    }

    strategies = [
        (FusionStrategy.CONCAT, ConcatFusion),
        (FusionStrategy.WEIGHTED, WeightedFusion),
        (FusionStrategy.GATED, GatedFusion),
        (FusionStrategy.ATTENTION, AttentionFusion),
        (FusionStrategy.RESIDUAL, ResidualFusion),
        (FusionStrategy.MOE, MoEFusion),
        (FusionStrategy.HIERARCHICAL, HierarchicalTreeFusion),
    ]

    for strategy, cls in strategies:
        def test_strategy(s=strategy):
            fusion = (FusionBuilder()
                      .with_streams(stream_dims)
                      .with_output_dim(D)
                      .with_strategy(s)
                      .build()
                      .to(DEVICE))
            out, info = fusion(stream_outputs)
            assert out.shape == (B, D), f"Wrong shape: {out.shape}"
            return True

        run_test(f"FusionStrategy.{strategy.name}", test_strategy)

    # FingerprintGuidedFusion
    def test_fingerprint_fusion():
        fingerprints = {
            name: torch.randn(64).to(DEVICE)
            for name in stream_dims.keys()
        }
        fusion = (FusionBuilder()
                  .with_streams(stream_dims)
                  .with_output_dim(D)
                  .with_strategy(FusionStrategy.FINGERPRINT)
                  .with_extra_kwargs(fingerprint_dim=64)
                  .build()
                  .to(DEVICE))
        out, info = fusion(stream_outputs, stream_fingerprints=fingerprints)
        assert out.shape == (B, D)
        return True

    run_test("FusionStrategy.FINGERPRINT with fingerprints", test_fingerprint_fusion)

    # Fusion gradients
    def test_fusion_gradients():
        fusion = (FusionBuilder()
                  .with_streams(stream_dims)
                  .with_output_dim(D)
                  .with_strategy(FusionStrategy.GATED)
                  .build()
                  .to(DEVICE))

        inputs = {k: v.clone().requires_grad_(True) for k, v in stream_outputs.items()}
        out, _ = fusion(inputs)
        loss = out.sum()
        loss.backward()

        for name, inp in inputs.items():
            assert inp.grad is not None, f"No gradient for {name}"
            assert inp.grad.abs().sum() > 0, f"Zero gradient for {name}"
        return True

    run_test("Fusion gradient flow", test_fusion_gradients)


# =============================================================================
# TEST 7: FACTORY SPECS
# =============================================================================

def test_factory_specs():
    test_section("7. Factory Specs")

    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec

    # StreamSpec
    def test_stream_spec_feature():
        spec = StreamSpec.feature_vector("clip", input_dim=512, feature_dim=256)
        assert spec.name == "clip"
        assert spec.input_dim == 512
        assert spec.feature_dim == 256
        assert spec.stream_type == "feature_vector"
        assert spec.input_shape == "vector"
        return True

    run_test("StreamSpec.feature_vector", test_stream_spec_feature)

    def test_stream_spec_trainable():
        spec = StreamSpec.trainable_vector("encoder", input_dim=768, feature_dim=512)
        assert spec.stream_type == "trainable_vector"
        return True

    run_test("StreamSpec.trainable_vector", test_stream_spec_trainable)

    def test_stream_spec_sequence():
        spec = StreamSpec.sequence("t5", input_dim=768)
        assert spec.input_shape == "sequence"
        return True

    run_test("StreamSpec.sequence", test_stream_spec_sequence)

    def test_stream_spec_transformer():
        spec = StreamSpec.transformer_sequence("bert", input_dim=768, num_layers=4)
        assert spec.stream_type == "transformer_sequence"
        assert spec.num_layers == 4
        return True

    run_test("StreamSpec.transformer_sequence", test_stream_spec_transformer)

    # HeadSpec
    def test_head_spec_presets():
        h1 = HeadSpec.lightweight(feature_dim=256)
        h2 = HeadSpec.standard(feature_dim=512)
        h3 = HeadSpec.heavy(feature_dim=512)

        assert h1.num_anchors < h2.num_anchors
        assert h2.num_anchors <= h3.num_anchors
        assert h1.feature_dim == 256
        return True

    run_test("HeadSpec presets", test_head_spec_presets)

    def test_head_spec_custom():
        spec = HeadSpec(
            feature_dim=384,
            fingerprint_dim=32,
            num_heads=4,
            num_anchors=8,
            num_routes=2,
        )
        assert spec.feature_dim == 384
        assert spec.fingerprint_dim == 32
        return True

    run_test("HeadSpec custom", test_head_spec_custom)

    # FusionSpec
    def test_fusion_spec_presets():
        f1 = FusionSpec.concat(output_dim=512)
        f2 = FusionSpec.gated(output_dim=512)
        f3 = FusionSpec.attention(output_dim=512)
        f4 = FusionSpec.moe(output_dim=512, num_experts=8)

        assert f1.strategy == "concat"
        assert f2.strategy == "gated"
        assert f3.strategy == "attention"
        assert f4.strategy == "moe"
        assert f4.num_experts == 8
        return True

    run_test("FusionSpec presets", test_fusion_spec_presets)


# =============================================================================
# TEST 8: REGISTRY
# =============================================================================

def test_registry():
    test_section("8. Registry and Mailbox")

    from geofractal.router.registry import RouterRegistry, RouterMailbox, get_registry
    from geofractal.router.config import GlobalFractalRouterConfig

    # Singleton
    def test_singleton():
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
        return True

    run_test("RouterRegistry singleton", test_singleton)

    # Registration
    def test_registration():
        registry = get_registry()
        registry.reset()

        id1 = registry.register("stream_a", None, "collective", 64, 512)
        id2 = registry.register("stream_b", None, "collective", 64, 512)

        assert id1 != id2
        assert registry.get(id1) is not None
        assert registry.get_by_name("stream_a") is not None
        return True

    run_test("RouterRegistry registration", test_registration)

    # Groups
    def test_groups():
        registry = get_registry()
        registry.reset()

        registry.register("a", None, "group1", 64, 512)
        registry.register("b", None, "group1", 64, 512)
        registry.register("c", None, "group2", 64, 512)

        group1 = registry.get_group("group1")
        assert len(group1) == 2
        return True

    run_test("RouterRegistry groups", test_groups)

    # Mailbox
    def test_mailbox():
        config = GlobalFractalRouterConfig()
        mailbox = RouterMailbox(config)

        mailbox.post(
            sender_id="router_1",
            sender_name="stream_a",
            content=torch.randn(64),
        )
        mailbox.post(
            sender_id="router_2",
            sender_name="stream_b",
            content=torch.randn(64),
        )

        assert len(mailbox) == 2

        msg = mailbox.read("router_1")
        assert msg is not None

        all_msgs = mailbox.read_all(exclude="router_1")
        assert len(all_msgs) == 1

        mailbox.clear()
        assert len(mailbox) == 0
        return True

    run_test("RouterMailbox operations", test_mailbox)


# =============================================================================
# TEST 9: ROUTER COLLECTIVE
# =============================================================================

def test_collective():
    test_section("9. RouterCollective")

    from geofractal.router.collective import RouterCollective
    from geofractal.router.config import CollectiveConfig
    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec
    from geofractal.router.registry import get_registry

    B = 4

    config = CollectiveConfig(
        feature_dim=256,
        num_classes=10,
        num_slots=16,
        device=DEVICE,
    )

    # from_specs
    def test_from_specs():
        get_registry().reset()
        collective = RouterCollective.from_specs(
            stream_specs=[
                StreamSpec.feature_vector("a", input_dim=512, feature_dim=256),
                StreamSpec.feature_vector("b", input_dim=768, feature_dim=256),
            ],
            config=config,
            head_spec=HeadSpec.lightweight(feature_dim=256),
            fusion_spec=FusionSpec.gated(output_dim=256),
        )
        assert len(collective.stream_names) == 2
        return True

    run_test("RouterCollective.from_specs", test_from_specs)

    # from_feature_dims
    def test_from_feature_dims():
        get_registry().reset()
        collective = RouterCollective.from_feature_dims(
            feature_configs={"clip": 512, "dino": 768},
            config=config,
        )
        assert len(collective.stream_names) == 2
        return True

    run_test("RouterCollective.from_feature_dims", test_from_feature_dims)

    # Forward pass
    def test_forward():
        get_registry().reset()
        collective = RouterCollective.from_specs(
            stream_specs=[
                StreamSpec.feature_vector("a", input_dim=512, feature_dim=256),
                StreamSpec.feature_vector("b", input_dim=512, feature_dim=256),
            ],
            config=config,
        ).to(DEVICE)

        inputs = {
            "a": torch.randn(B, 512).to(DEVICE),
            "b": torch.randn(B, 512).to(DEVICE),
        }

        logits, info = collective(inputs)
        assert logits.shape == (B, 10)
        assert 'stream_infos' in info
        return True

    run_test("RouterCollective forward", test_forward)

    # Forward with return_individual
    def test_forward_individual():
        get_registry().reset()
        collective = RouterCollective.from_specs(
            stream_specs=[
                StreamSpec.feature_vector("a", input_dim=512, feature_dim=256),
                StreamSpec.feature_vector("b", input_dim=512, feature_dim=256),
            ],
            config=config,
        ).to(DEVICE)

        inputs = {
            "a": torch.randn(B, 512).to(DEVICE),
            "b": torch.randn(B, 512).to(DEVICE),
        }

        logits, info = collective(inputs, return_individual=True)
        assert 'individual_logits' in info
        assert 'a' in info['individual_logits']
        assert 'b' in info['individual_logits']
        return True

    run_test("RouterCollective return_individual", test_forward_individual)

    # Backward pass
    def test_backward():
        get_registry().reset()
        collective = RouterCollective.from_specs(
            stream_specs=[
                StreamSpec.feature_vector("a", input_dim=512, feature_dim=256),
                StreamSpec.feature_vector("b", input_dim=512, feature_dim=256),
            ],
            config=config,
        ).to(DEVICE)

        inputs = {
            "a": torch.randn(B, 512).to(DEVICE),
            "b": torch.randn(B, 512).to(DEVICE),
        }
        labels = torch.randint(0, 10, (B,)).to(DEVICE)

        logits, _ = collective(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        grad_count = sum(1 for p in collective.parameters() if p.grad is not None)
        assert grad_count > 0
        return True

    run_test("RouterCollective backward", test_backward)

    # Emergence computation
    def test_emergence():
        get_registry().reset()
        collective = RouterCollective.from_specs(
            stream_specs=[
                StreamSpec.feature_vector("a", input_dim=512, feature_dim=256),
                StreamSpec.feature_vector("b", input_dim=512, feature_dim=256),
            ],
            config=config,
        ).to(DEVICE)

        emergence = collective.compute_emergence(
            collective_acc=0.85,
            individual_accs={"a": 0.10, "b": 0.12},
        )

        assert 'rho' in emergence
        assert emergence['rho'] == 0.85 / 0.12
        assert emergence['emergence'] == True
        return True

    run_test("RouterCollective compute_emergence", test_emergence)

    # Summary
    def test_summary():
        get_registry().reset()
        collective = RouterCollective.from_specs(
            stream_specs=[
                StreamSpec.feature_vector("a", input_dim=512, feature_dim=256),
            ],
            config=config,
        ).to(DEVICE)

        summary = collective.summary()
        assert "RouterCollective" in summary
        assert "a" in summary
        return True

    run_test("RouterCollective summary", test_summary)


# =============================================================================
# TEST 10: GRADIENT FLOW
# =============================================================================

def test_gradient_flow():
    test_section("10. Gradient Flow (All Components)")

    from geofractal.router.collective import RouterCollective
    from geofractal.router.config import CollectiveConfig
    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec
    from geofractal.router.registry import get_registry

    B = 4
    config = CollectiveConfig(
        feature_dim=128,
        num_classes=10,
        num_slots=8,
        device=DEVICE,
    )

    get_registry().reset()
    collective = RouterCollective.from_specs(
        stream_specs=[
            StreamSpec.feature_vector("a", input_dim=256, feature_dim=128),
            StreamSpec.feature_vector("b", input_dim=256, feature_dim=128),
        ],
        config=config,
        head_spec=HeadSpec.lightweight(feature_dim=128),
        fusion_spec=FusionSpec.gated(output_dim=128),
    ).to(DEVICE)

    collective.zero_grad()

    inputs = {
        "a": torch.randn(B, 256, device=DEVICE),
        "b": torch.randn(B, 256, device=DEVICE),
    }
    labels = torch.randint(0, 10, (B,), device=DEVICE)

    logits, info = collective(inputs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    # Check critical components
    components = {}

    # Fingerprints
    for name in collective.stream_names:
        fp = collective.heads[name].fingerprint
        has_grad = fp.grad is not None and fp.grad.abs().sum() > 0
        components[f"head[{name}].fingerprint"] = has_grad

    # Anchors
    for name in collective.stream_names:
        anchors = collective.heads[name].anchors
        if hasattr(anchors, 'anchors'):
            p = anchors.anchors
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            components[f"head[{name}].anchors"] = has_grad

    # Attention
    for name in collective.stream_names:
        attn = collective.heads[name].attention
        if hasattr(attn, 'q_proj'):
            p = attn.q_proj.weight
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            components[f"head[{name}].attention.q_proj"] = has_grad

    # Router fp_to_bias
    for name in collective.stream_names:
        router = collective.heads[name].router
        if hasattr(router, 'fp_to_bias'):
            p = router.fp_to_bias.weight
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            components[f"head[{name}].router.fp_to_bias"] = has_grad

    # Gate fp_compare
    for name in collective.stream_names:
        gate = collective.heads[name].gate
        if hasattr(gate, 'fp_compare'):
            for pname, p in gate.fp_compare.named_parameters():
                has_grad = p.grad is not None and p.grad.abs().sum() > 0
                components[f"head[{name}].gate.fp_compare"] = has_grad
                break

    # Combiner
    for name in collective.stream_names:
        combiner = collective.heads[name].combiner
        if hasattr(combiner, 'weights'):
            p = combiner.weights
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            components[f"head[{name}].combiner.weights"] = has_grad

    # Streams
    for name in collective.stream_names:
        stream = collective.streams[name]
        for pname, p in stream.named_parameters():
            if p.requires_grad:
                has_grad = p.grad is not None and p.grad.abs().sum() > 0
                components[f"stream[{name}]"] = has_grad
                break

    # Fusion
    if hasattr(collective.fusion, 'gate_net'):
        for pname, p in collective.fusion.gate_net.named_parameters():
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            components["fusion.gate_net"] = has_grad
            break

    # Classifier
    for pname, p in collective.classifier.named_parameters():
        if p.requires_grad:
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            components["classifier"] = has_grad
            break

    # Report
    for comp, has_grad in sorted(components.items()):
        if has_grad:
            test_pass(comp)
        else:
            test_fail(comp, "No gradient")


# =============================================================================
# TEST 11: END-TO-END TRAINING
# =============================================================================

def test_training():
    test_section("11. End-to-End Training")

    from geofractal.router.collective import RouterCollective
    from geofractal.router.config import CollectiveConfig
    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec
    from geofractal.router.registry import get_registry

    B = 8
    num_classes = 10

    config = CollectiveConfig(
        feature_dim=128,
        num_classes=num_classes,
        num_slots=8,
        device=DEVICE,
    )

    get_registry().reset()
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

    # Training loop
    def test_training_loop():
        losses = []
        collective.train()

        for step in range(10):
            inputs = {
                "a": torch.randn(B, 256).to(DEVICE),
                "b": torch.randn(B, 256).to(DEVICE),
                "c": torch.randn(B, 256).to(DEVICE),
            }
            labels = torch.randint(0, num_classes, (B,)).to(DEVICE)

            optimizer.zero_grad()
            logits, info = collective(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return True

    run_test("10-step training loop", test_training_loop)

    # Evaluation
    def test_evaluation():
        collective.eval()
        with torch.no_grad():
            inputs = {
                "a": torch.randn(B, 256).to(DEVICE),
                "b": torch.randn(B, 256).to(DEVICE),
                "c": torch.randn(B, 256).to(DEVICE),
            }
            labels = torch.randint(0, num_classes, (B,)).to(DEVICE)

            logits, info = collective(inputs, return_individual=True)
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        return True

    run_test("Evaluation mode", test_evaluation)

    # Freeze/unfreeze
    def test_freeze_unfreeze():
        if hasattr(collective, 'freeze_streams'):
            collective.freeze_streams()
            for name in collective.stream_names:
                for p in collective.streams[name].parameters():
                    assert not p.requires_grad

            collective.unfreeze_streams()
            for name in collective.stream_names:
                for p in collective.streams[name].parameters():
                    assert p.requires_grad
        return True

    run_test("Freeze/unfreeze streams", test_freeze_unfreeze)


# =============================================================================
# TEST 12: SERIALIZATION
# =============================================================================

def test_serialization():
    test_section("12. Serialization")

    from geofractal.router.collective import RouterCollective
    from geofractal.router.config import CollectiveConfig
    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec
    from geofractal.router.registry import get_registry

    B = 4
    config = CollectiveConfig(
        feature_dim=128,
        num_classes=10,
        num_slots=8,
        device=DEVICE,
    )

    get_registry().reset()
    collective = RouterCollective.from_specs(
        stream_specs=[
            StreamSpec.feature_vector("a", input_dim=256, feature_dim=128),
            StreamSpec.feature_vector("b", input_dim=256, feature_dim=128),
        ],
        config=config,
    ).to(DEVICE)

    inputs = {
        "a": torch.randn(B, 256).to(DEVICE),
        "b": torch.randn(B, 256).to(DEVICE),
    }

    # Save/load state dict
    def test_state_dict():
        out1, _ = collective(inputs)

        state = collective.state_dict()

        get_registry().reset()
        collective2 = RouterCollective.from_specs(
            stream_specs=[
                StreamSpec.feature_vector("a", input_dim=256, feature_dim=128),
                StreamSpec.feature_vector("b", input_dim=256, feature_dim=128),
            ],
            config=config,
        ).to(DEVICE)

        collective2.load_state_dict(state)

        out2, _ = collective2(inputs)

        assert torch.allclose(out1, out2, atol=1e-5)
        return True

    run_test("state_dict save/load", test_state_dict)

    # Save/load to file
    def test_save_load_file():
        out1, _ = collective(inputs)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(collective.state_dict(), f.name)
            path = f.name

        try:
            get_registry().reset()
            collective2 = RouterCollective.from_specs(
                stream_specs=[
                    StreamSpec.feature_vector("a", input_dim=256, feature_dim=128),
                    StreamSpec.feature_vector("b", input_dim=256, feature_dim=128),
                ],
                config=config,
            ).to(DEVICE)

            collective2.load_state_dict(torch.load(path, weights_only=True))

            out2, _ = collective2(inputs)

            assert torch.allclose(out1, out2, atol=1e-5)
        finally:
            os.unlink(path)

        return True

    run_test("File save/load", test_save_load_file)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  GeoFractal Router - Comprehensive Component Tests")
    print("=" * 70)

    test_imports()
    test_cantor_functions()
    test_head_components()
    test_head_builder()
    test_streams()
    test_fusion()
    test_factory_specs()
    test_registry()
    test_collective()
    test_gradient_flow()
    test_training()
    test_serialization()

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  ✓ Passed: {PASSED}")
    print(f"  ✗ Failed: {FAILED}")
    print(f"  Total:   {PASSED + FAILED}")

    if ERRORS:
        print(f"\n  Errors:")
        for e in ERRORS:
            print(f"    - {e}")

    print("=" * 70)

    return FAILED == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)