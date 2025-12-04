"""
GeoFractal Router - Factory & Prototype Builder Tests

Tests:
1. HeadBuilder - fluent API, presets, static injection, component access
2. AssembledPrototype - full prototype construction from PrototypeConfig
3. LightweightPrototype - minimal overhead prototype
4. Gradient flow verification for all configurations
5. Component swapping and customization
6. Serialization

Copyright 2025 AbstractPhil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import tempfile
import os
from typing import Dict, List, Tuple, Any

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Testing on: {DEVICE}\n")

PASSED = 0
FAILED = 0
ERRORS = []


def test_section(name: str):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")


def test_pass(name: str):
    global PASSED
    PASSED += 1
    print(f"  ✓ {name}")


def test_fail(name: str, error: str):
    global FAILED, ERRORS
    FAILED += 1
    ERRORS.append(f"{name}: {error}")
    print(f"  ✗ {name}: {error[:100]}...")


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
        import traceback
        test_fail(name, f"{e}\n{traceback.format_exc()}")
        return False


def verify_head_gradients(head: nn.Module, x: torch.Tensor) -> Tuple[bool, Dict[str, bool]]:
    """Verify gradients flow through head components."""
    head.zero_grad()
    head.train()

    out = head(x)
    loss = out.sum()
    loss.backward()

    results = {}

    # Fingerprint
    if hasattr(head, 'fingerprint'):
        has_grad = head.fingerprint.grad is not None and head.fingerprint.grad.abs().sum() > 0
        results['fingerprint'] = has_grad

    # Attention
    if hasattr(head, 'attention') and hasattr(head.attention, 'q_proj'):
        p = head.attention.q_proj.weight
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        results['attention'] = has_grad

    # Router
    if hasattr(head, 'router') and hasattr(head.router, 'fp_to_bias'):
        p = head.router.fp_to_bias.weight
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        results['router'] = has_grad

    # Anchors
    if hasattr(head, 'anchors') and hasattr(head.anchors, 'anchors'):
        p = head.anchors.anchors
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        results['anchors'] = has_grad

    # Gate
    if hasattr(head, 'gate') and hasattr(head.gate, 'fp_compare'):
        for pname, p in head.gate.fp_compare.named_parameters():
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            results['gate'] = has_grad
            break

    # Combiner
    if hasattr(head, 'combiner') and hasattr(head.combiner, 'weights'):
        p = head.combiner.weights
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        results['combiner'] = has_grad

    # Refinement
    if hasattr(head, 'refinement'):
        for pname, p in head.refinement.named_parameters():
            if p.requires_grad:
                has_grad = p.grad is not None and p.grad.abs().sum() > 0
                results['refinement'] = has_grad
                break

    all_passed = all(results.values())
    return all_passed, results


# =============================================================================
# TEST 1: HEADBUILDER BASIC
# =============================================================================

def test_headbuilder_basic():
    test_section("1. HeadBuilder Basic")

    from geofractal.router.head.builder import (
        HeadBuilder, ComposedHead, HeadPreset,
        STANDARD_HEAD, LIGHTWEIGHT_HEAD, HEAVY_HEAD,
    )
    from geofractal.router.head.components import HeadConfig

    B, S, D = 4, 16, 256
    config = HeadConfig(feature_dim=D, fingerprint_dim=64, num_heads=8, num_anchors=16, num_routes=4)
    x = torch.randn(B, S, D).to(DEVICE)

    # Default build
    def test_default_build():
        head = HeadBuilder(config).build().to(DEVICE)
        assert isinstance(head, ComposedHead)
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("Default build", test_default_build)

    # Standard preset
    def test_standard_preset():
        head = HeadBuilder.standard(config).build().to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("Standard preset", test_standard_preset)

    # Lightweight preset
    def test_lightweight_preset():
        head = HeadBuilder.lightweight(config).build().to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("Lightweight preset", test_lightweight_preset)

    # Heavy preset
    def test_heavy_preset():
        head = HeadBuilder.heavy(config).build().to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("Heavy preset", test_heavy_preset)

    # Has fingerprint
    def test_has_fingerprint():
        head = HeadBuilder(config).build().to(DEVICE)
        assert hasattr(head, 'fingerprint')
        assert head.fingerprint.shape == (config.fingerprint_dim,)
        return True
    run_test("Head has fingerprint", test_has_fingerprint)

    # return_info works
    def test_return_info():
        head = HeadBuilder(config).build().to(DEVICE)
        out, info = head(x, return_info=True)
        assert out.shape == (B, S, D)
        assert 'routes' in info
        assert 'route_weights' in info
        assert 'anchor_affinities' in info
        assert 'attn_weights' in info
        return True
    run_test("return_info returns expected keys", test_return_info)


# =============================================================================
# TEST 2: HEADBUILDER FLUENT API
# =============================================================================

def test_headbuilder_fluent():
    test_section("2. HeadBuilder Fluent API")

    from geofractal.router.head.builder import HeadBuilder, ComposedHead
    from geofractal.router.head.components import (
        HeadConfig,
        CantorAttention, StandardAttention,
        TopKRouter, SoftRouter,
        ConstitutiveAnchorBank, AttentiveAnchorBank,
        FingerprintGate, ChannelGate,
        LearnableWeightCombiner, GatedCombiner,
        FFNRefinement, MixtureOfExpertsRefinement,
    )

    B, S, D = 4, 16, 256
    config = HeadConfig(feature_dim=D)
    x = torch.randn(B, S, D).to(DEVICE)

    # with_attention variations
    attention_types = [
        ('CantorAttention', CantorAttention),
        ('StandardAttention', StandardAttention),
    ]
    for name, cls in attention_types:
        def test_attn(c=cls, n=name):
            head = HeadBuilder(config).with_attention(c).build().to(DEVICE)
            out = head(x)
            assert out.shape == (B, S, D)
            return True
        run_test(f"with_attention({name})", test_attn)

    # with_router variations
    router_types = [
        ('TopKRouter', TopKRouter),
        ('SoftRouter', SoftRouter),
    ]
    for name, cls in router_types:
        def test_router(c=cls, n=name):
            head = HeadBuilder(config).with_router(c).build().to(DEVICE)
            out = head(x)
            assert out.shape == (B, S, D)
            return True
        run_test(f"with_router({name})", test_router)

    # with_anchors variations
    anchor_types = [
        ('ConstitutiveAnchorBank', ConstitutiveAnchorBank),
        ('AttentiveAnchorBank', AttentiveAnchorBank),
    ]
    for name, cls in anchor_types:
        def test_anchor(c=cls, n=name):
            head = HeadBuilder(config).with_anchors(c).build().to(DEVICE)
            out = head(x)
            assert out.shape == (B, S, D)
            return True
        run_test(f"with_anchors({name})", test_anchor)

    # with_gate variations
    gate_types = [
        ('FingerprintGate', FingerprintGate),
        ('ChannelGate', ChannelGate),
    ]
    for name, cls in gate_types:
        def test_gate(c=cls, n=name):
            head = HeadBuilder(config).with_gate(c).build().to(DEVICE)
            out = head(x)
            assert out.shape == (B, S, D)
            return True
        run_test(f"with_gate({name})", test_gate)

    # with_combiner variations
    combiner_types = [
        ('LearnableWeightCombiner', LearnableWeightCombiner),
        ('GatedCombiner', GatedCombiner),
    ]
    for name, cls in combiner_types:
        def test_combiner(c=cls, n=name):
            head = HeadBuilder(config).with_combiner(c).build().to(DEVICE)
            out = head(x)
            assert out.shape == (B, S, D)
            return True
        run_test(f"with_combiner({name})", test_combiner)

    # with_refinement variations
    refinement_types = [
        ('FFNRefinement', FFNRefinement),
        ('MixtureOfExpertsRefinement', MixtureOfExpertsRefinement),
    ]
    for name, cls in refinement_types:
        def test_refine(c=cls, n=name):
            head = HeadBuilder(config).with_refinement(c).build().to(DEVICE)
            out = head(x)
            assert out.shape == (B, S, D)
            return True
        run_test(f"with_refinement({name})", test_refine)

    # Full chain
    def test_full_chain():
        head = (HeadBuilder(config)
            .with_attention(CantorAttention)
            .with_router(TopKRouter)
            .with_anchors(AttentiveAnchorBank)
            .with_gate(ChannelGate)
            .with_combiner(GatedCombiner)
            .with_refinement(MixtureOfExpertsRefinement)
            .build()
            .to(DEVICE))
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("Full fluent chain", test_full_chain)


# =============================================================================
# TEST 3: HEADBUILDER STATIC INJECTION
# =============================================================================

def test_headbuilder_injection():
    test_section("3. HeadBuilder Static Injection")

    from geofractal.router.head.builder import HeadBuilder, ComposedHead
    from geofractal.router.head.components import (
        HeadConfig,
        CantorAttention, StandardAttention,
        TopKRouter,
        ConstitutiveAnchorBank,
        FingerprintGate,
        LearnableWeightCombiner,
        FFNRefinement,
    )

    B, S, D = 4, 16, 256
    config = HeadConfig(feature_dim=D)
    x = torch.randn(B, S, D).to(DEVICE)

    # Inject attention
    def test_inject_attention():
        custom_attn = StandardAttention(config).to(DEVICE)
        head = HeadBuilder(config).inject_attention(custom_attn).build().to(DEVICE)
        assert head.attention is custom_attn
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("inject_attention", test_inject_attention)

    # Inject router
    def test_inject_router():
        custom_router = TopKRouter(config).to(DEVICE)
        head = HeadBuilder(config).inject_router(custom_router).build().to(DEVICE)
        assert head.router is custom_router
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("inject_router", test_inject_router)

    # Inject anchors
    def test_inject_anchors():
        custom_anchors = ConstitutiveAnchorBank(config).to(DEVICE)
        head = HeadBuilder(config).inject_anchors(custom_anchors).build().to(DEVICE)
        assert head.anchors is custom_anchors
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("inject_anchors", test_inject_anchors)

    # Inject gate
    def test_inject_gate():
        custom_gate = FingerprintGate(config).to(DEVICE)
        head = HeadBuilder(config).inject_gate(custom_gate).build().to(DEVICE)
        assert head.gate is custom_gate
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("inject_gate", test_inject_gate)

    # Inject combiner
    def test_inject_combiner():
        custom_combiner = LearnableWeightCombiner(config).to(DEVICE)
        head = HeadBuilder(config).inject_combiner(custom_combiner).build().to(DEVICE)
        assert head.combiner is custom_combiner
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("inject_combiner", test_inject_combiner)

    # Inject refinement
    def test_inject_refinement():
        custom_refine = FFNRefinement(config).to(DEVICE)
        head = HeadBuilder(config).inject_refinement(custom_refine).build().to(DEVICE)
        assert head.refinement is custom_refine
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("inject_refinement", test_inject_refinement)

    # Multiple injections
    def test_multiple_injections():
        custom_attn = StandardAttention(config).to(DEVICE)
        custom_router = TopKRouter(config).to(DEVICE)

        head = (HeadBuilder(config)
            .inject_attention(custom_attn)
            .inject_router(custom_router)
            .build()
            .to(DEVICE))

        assert head.attention is custom_attn
        assert head.router is custom_router
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("Multiple injections", test_multiple_injections)


# =============================================================================
# TEST 4: HEADBUILDER GRADIENT FLOW
# =============================================================================

# =============================================================================
# TEST 4: HEADBUILDER GRADIENT FLOW
# =============================================================================

def test_headbuilder_gradients():
    test_section("4. HeadBuilder Gradient Flow")

    from geofractal.router.head.builder import HeadBuilder
    from geofractal.router.head.components import (
        HeadConfig,
        CantorAttention, StandardAttention,
        TopKRouter, SoftRouter,
        ConstitutiveAnchorBank, AttentiveAnchorBank,
    )

    B, S, D = 4, 16, 256
    config = HeadConfig(feature_dim=D)
    x = torch.randn(B, S, D).to(DEVICE)

    # Standard preset gradients
    def test_standard_gradients():
        head = HeadBuilder.standard(config).build().to(DEVICE)
        all_passed, results = verify_head_gradients(head, x)
        failed = [k for k, v in results.items() if not v]
        if failed:
            return f"No gradients: {failed}"
        return True

    run_test("Standard preset gradients", test_standard_gradients)

    # Lightweight preset gradients
    def test_lightweight_gradients():
        head = HeadBuilder.lightweight(config).build().to(DEVICE)
        all_passed, results = verify_head_gradients(head, x)
        failed = [k for k, v in results.items() if not v]
        if failed:
            return f"No gradients: {failed}"
        return True

    run_test("Lightweight preset gradients", test_lightweight_gradients)

    # Heavy preset gradients
    def test_heavy_gradients():
        head = HeadBuilder.heavy(config).build().to(DEVICE)
        all_passed, results = verify_head_gradients(head, x)
        failed = [k for k, v in results.items() if not v]
        if failed:
            return f"No gradients: {failed}"
        return True

    run_test("Heavy preset gradients", test_heavy_gradients)

    # CantorAttention gradients
    def test_cantor_gradients():
        head = HeadBuilder(config).with_attention(CantorAttention).build().to(DEVICE)
        all_passed, results = verify_head_gradients(head, x)
        failed = [k for k, v in results.items() if not v]
        if failed:
            return f"No gradients: {failed}"
        return True

    run_test("CantorAttention gradients", test_cantor_gradients)

    # AttentiveAnchorBank gradients
    def test_attentive_anchor_gradients():
        head = HeadBuilder(config).with_anchors(AttentiveAnchorBank).build().to(DEVICE)
        all_passed, results = verify_head_gradients(head, x)
        failed = [k for k, v in results.items() if not v]
        if failed:
            return f"No gradients: {failed}"
        return True

    run_test("AttentiveAnchorBank gradients", test_attentive_anchor_gradients)

    # With target_fingerprint
    def test_target_fingerprint_gradients():
        head = HeadBuilder(config).build().to(DEVICE)
        target_fp = torch.randn(config.fingerprint_dim, device=DEVICE)

        head.zero_grad()
        out = head(x, target_fingerprint=target_fp)
        loss = out.sum()
        loss.backward()

        # Gate should have gradients due to target_fingerprint
        has_grad = head.fingerprint.grad is not None and head.fingerprint.grad.abs().sum() > 0
        if not has_grad:
            return "No fingerprint gradient with target_fingerprint"
        return True

    run_test("target_fingerprint gradient flow", test_target_fingerprint_gradients)


# =============================================================================
# TEST 5: COMPOSED HEAD COMPONENT ACCESS
# =============================================================================

def test_composedhead_access():
    test_section("5. ComposedHead Component Access")

    from geofractal.router.head.builder import HeadBuilder, ComposedHead
    from geofractal.router.head.components import (
        HeadConfig,
        CantorAttention,
        TopKRouter,
        ConstitutiveAnchorBank,
        FingerprintGate,
        LearnableWeightCombiner,
        FFNRefinement,
    )

    config = HeadConfig(feature_dim=256)
    head = HeadBuilder.standard(config).build().to(DEVICE)

    # get_component
    def test_get_component():
        assert head.get_component('attention') is head.attention
        assert head.get_component('router') is head.router
        assert head.get_component('anchors') is head.anchors
        assert head.get_component('gate') is head.gate
        assert head.get_component('combiner') is head.combiner
        assert head.get_component('refinement') is head.refinement
        return True
    run_test("get_component access", test_get_component)

    # Direct attribute access
    def test_direct_access():
        assert isinstance(head.attention, nn.Module)
        assert isinstance(head.router, nn.Module)
        assert isinstance(head.anchors, nn.Module)
        assert isinstance(head.gate, nn.Module)
        assert isinstance(head.combiner, nn.Module)
        assert isinstance(head.refinement, nn.Module)
        return True
    run_test("Direct attribute access", test_direct_access)

    # replace_component
    def test_replace_component():
        B, S, D = 4, 16, 256
        x = torch.randn(B, S, D).to(DEVICE)

        # Replace attention with standard
        from geofractal.router.head.components import StandardAttention
        new_attn = StandardAttention(config).to(DEVICE)
        head.replace_component('attention', new_attn)

        assert head.attention is new_attn
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("replace_component", test_replace_component)

    # num_parameters property
    def test_num_parameters():
        num_params = head.num_parameters
        assert num_params > 0

        # Should match manual count
        manual_count = sum(p.numel() for p in head.parameters() if p.requires_grad)
        assert num_params == manual_count
        return True
    run_test("num_parameters property", test_num_parameters)


# =============================================================================
# TEST 6: PROTOTYPE CONFIG
# =============================================================================

def test_prototype_config():
    test_section("6. PrototypeConfig")

    from geofractal.router.factory.prototype import PrototypeConfig
    from geofractal.router.factory.protocols import StreamSpec, HeadSpec, FusionSpec

    # Basic construction
    def test_basic_config():
        config = PrototypeConfig(
            num_classes=1000,
            prototype_name="test_prototype",
        )
        assert config.num_classes == 1000
        assert config.prototype_name == "test_prototype"
        return True
    run_test("Basic PrototypeConfig", test_basic_config)

    # With stream specs
    def test_with_streams():
        config = PrototypeConfig(
            num_classes=100,
            stream_specs=[
                StreamSpec.feature_stream("clip", input_dim=512, feature_dim=256),
                StreamSpec.feature_stream("dino", input_dim=768, feature_dim=256),
            ],
        )
        assert len(config.stream_specs) == 2
        assert config.stream_specs[0].name == "clip"
        assert config.stream_specs[1].name == "dino"
        return True
    run_test("PrototypeConfig with stream_specs", test_with_streams)

    # With head spec
    def test_with_head_spec():
        config = PrototypeConfig(
            num_classes=100,
            head_spec=HeadSpec.lightweight(feature_dim=256),
        )
        assert config.head_spec.feature_dim == 256
        return True
    run_test("PrototypeConfig with head_spec", test_with_head_spec)

    # With fusion spec
    def test_with_fusion_spec():
        config = PrototypeConfig(
            num_classes=100,
            fusion_spec=FusionSpec.adaptive(output_dim=512),
        )
        assert config.fusion_spec.strategy == "gated"
        assert config.fusion_spec.output_dim == 512
        return True
    run_test("PrototypeConfig with fusion_spec", test_with_fusion_spec)

    # to_dict / from_dict roundtrip
    def test_config_roundtrip():
        config = PrototypeConfig(
            num_classes=100,
            stream_specs=[
                StreamSpec.feature_stream("a", input_dim=512, feature_dim=256),
            ],
            head_spec=HeadSpec.standard(feature_dim=256),
            fusion_spec=FusionSpec.standard(output_dim=256),
        )

        d = config.to_dict()
        config2 = PrototypeConfig.from_dict(d)

        assert config2.num_classes == config.num_classes
        assert len(config2.stream_specs) == len(config.stream_specs)
        return True
    run_test("Config to_dict/from_dict roundtrip", test_config_roundtrip)


# =============================================================================
# TEST 7: ASSEMBLED PROTOTYPE
# =============================================================================

def test_assembled_prototype():
    test_section("7. AssembledPrototype")

    from geofractal.router.factory.prototype import AssembledPrototype, PrototypeConfig
    from geofractal.router.factory.protocols import StreamSpec, HeadSpec, FusionSpec

    B = 4

    # Basic construction with feature streams
    def test_basic_prototype():
        config = PrototypeConfig(
            num_classes=10,
            stream_specs=[
                StreamSpec.feature_stream("a", input_dim=512, feature_dim=256),
                StreamSpec.feature_stream("b", input_dim=768, feature_dim=256),
            ],
            head_spec=HeadSpec.lightweight(feature_dim=256),
            fusion_spec=FusionSpec.standard(output_dim=256),
            freeze_streams=False,
        )

        prototype = AssembledPrototype(config).to(DEVICE)

        assert len(prototype.stream_names) == 2
        assert 'a' in prototype.stream_names
        assert 'b' in prototype.stream_names
        return True
    run_test("Basic AssembledPrototype construction", test_basic_prototype)

    # Forward pass
    def test_forward():
        config = PrototypeConfig(
            num_classes=10,
            stream_specs=[
                StreamSpec.feature_stream("a", input_dim=256, feature_dim=128),
                StreamSpec.feature_stream("b", input_dim=256, feature_dim=128),
            ],
            head_spec=HeadSpec.lightweight(feature_dim=128),
            fusion_spec=FusionSpec.standard(output_dim=128),
            freeze_streams=False,
        )

        prototype = AssembledPrototype(config).to(DEVICE)

        # Feature inputs
        inputs = {
            "a": torch.randn(B, 256).to(DEVICE),
            "b": torch.randn(B, 256).to(DEVICE),
        }

        logits, info = prototype(inputs)
        assert logits.shape == (B, 10)
        return True
    run_test("AssembledPrototype forward", test_forward)

    # Forward with return_info
    def test_forward_info():
        config = PrototypeConfig(
            num_classes=10,
            stream_specs=[
                StreamSpec.feature_stream("a", input_dim=256, feature_dim=128),
                StreamSpec.feature_stream("b", input_dim=256, feature_dim=128),
            ],
            head_spec=HeadSpec.lightweight(feature_dim=128),
            fusion_spec=FusionSpec.standard(output_dim=128),
            freeze_streams=False,
        )

        prototype = AssembledPrototype(config).to(DEVICE)

        inputs = {
            "a": torch.randn(B, 256).to(DEVICE),
            "b": torch.randn(B, 256).to(DEVICE),
        }

        logits, info = prototype(inputs, return_info=True)
        assert logits.shape == (B, 10)
        assert info is not None
        assert hasattr(info, 'routing_info')
        assert hasattr(info, 'fingerprints')
        return True
    run_test("AssembledPrototype return_info", test_forward_info)

    # freeze_streams / unfreeze_streams
    def test_freeze_unfreeze():
        config = PrototypeConfig(
            num_classes=10,
            stream_specs=[
                StreamSpec.feature_stream("a", input_dim=256, feature_dim=128),
            ],
            freeze_streams=False,
        )

        prototype = AssembledPrototype(config).to(DEVICE)

        # Check initially unfrozen
        for p in prototype.streams["a"].parameters():
            assert p.requires_grad

        # Freeze
        prototype.freeze_streams()
        for p in prototype.streams["a"].parameters():
            assert not p.requires_grad

        # Unfreeze
        prototype.unfreeze_streams()
        for p in prototype.streams["a"].parameters():
            assert p.requires_grad

        return True
    run_test("freeze_streams / unfreeze_streams", test_freeze_unfreeze)


# =============================================================================
# TEST 8: LIGHTWEIGHT PROTOTYPE
# =============================================================================

def test_lightweight_prototype():
    test_section("8. LightweightPrototype")

    from geofractal.router.factory.prototype import LightweightPrototype

    B = 4

    # Basic construction
    def test_basic_lightweight():
        prototype = LightweightPrototype(
            stream_dims={"a": 512, "b": 768},
            num_classes=10,
            hidden_dim=256,
        ).to(DEVICE)

        assert len(prototype.stream_names) == 2
        return True
    run_test("Basic LightweightPrototype", test_basic_lightweight)

    # Forward pass
    def test_lightweight_forward():
        prototype = LightweightPrototype(
            stream_dims={"a": 512, "b": 768},
            num_classes=10,
            hidden_dim=256,
        ).to(DEVICE)

        inputs = {
            "a": torch.randn(B, 512).to(DEVICE),
            "b": torch.randn(B, 768).to(DEVICE),
        }

        logits, info = prototype(inputs)
        assert logits.shape == (B, 10)
        return True
    run_test("LightweightPrototype forward", test_lightweight_forward)

    # return_info
    def test_lightweight_info():
        prototype = LightweightPrototype(
            stream_dims={"a": 512, "b": 768},
            num_classes=10,
            hidden_dim=256,
        ).to(DEVICE)

        inputs = {
            "a": torch.randn(B, 512).to(DEVICE),
            "b": torch.randn(B, 768).to(DEVICE),
        }

        logits, info = prototype(inputs, return_info=True)
        assert logits.shape == (B, 10)
        assert info is not None
        assert info.fusion_weights is not None
        return True
    run_test("LightweightPrototype return_info", test_lightweight_info)

    # Gradients flow
    def test_lightweight_gradients():
        prototype = LightweightPrototype(
            stream_dims={"a": 512, "b": 512},
            num_classes=10,
            hidden_dim=256,
        ).to(DEVICE)

        inputs = {
            "a": torch.randn(B, 512).to(DEVICE),
            "b": torch.randn(B, 512).to(DEVICE),
        }
        labels = torch.randint(0, 10, (B,)).to(DEVICE)

        prototype.zero_grad()
        logits, _ = prototype(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        grad_count = sum(1 for p in prototype.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0
        return True
    run_test("LightweightPrototype gradients", test_lightweight_gradients)

    # Many streams
    def test_lightweight_many_streams():
        prototype = LightweightPrototype(
            stream_dims={"a": 512, "b": 768, "c": 1024, "d": 384, "e": 256},
            num_classes=100,
            hidden_dim=512,
        ).to(DEVICE)

        inputs = {
            "a": torch.randn(B, 512).to(DEVICE),
            "b": torch.randn(B, 768).to(DEVICE),
            "c": torch.randn(B, 1024).to(DEVICE),
            "d": torch.randn(B, 384).to(DEVICE),
            "e": torch.randn(B, 256).to(DEVICE),
        }

        logits, _ = prototype(inputs)
        assert logits.shape == (B, 100)
        return True
    run_test("LightweightPrototype 5 streams", test_lightweight_many_streams)


# =============================================================================
# TEST 9: FACTORY FUNCTIONS
# =============================================================================

def test_factory_functions():
    test_section("9. Factory Functions")

    from geofractal.router.head.builder import (
        build_standard_head, build_lightweight_head, build_custom_head,
    )
    from geofractal.router.head.components import (
        HeadConfig,
        StandardAttention,
        SoftRouter,
        AttentiveAnchorBank,
    )

    B, S, D = 4, 16, 256
    config = HeadConfig(feature_dim=D)
    x = torch.randn(B, S, D).to(DEVICE)

    # build_standard_head
    def test_build_standard():
        head = build_standard_head(config).to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("build_standard_head", test_build_standard)

    # build_lightweight_head
    def test_build_lightweight():
        head = build_lightweight_head(config).to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("build_lightweight_head", test_build_lightweight)

    # build_custom_head
    def test_build_custom():
        head = build_custom_head(
            config,
            attention_cls=StandardAttention,
            router_cls=SoftRouter,
            anchor_cls=AttentiveAnchorBank,
        ).to(DEVICE)
        out = head(x)
        assert out.shape == (B, S, D)
        return True
    run_test("build_custom_head", test_build_custom)


# =============================================================================
# TEST 10: SERIALIZATION
# =============================================================================

def test_serialization():
    test_section("10. Serialization")

    from geofractal.router.head.builder import HeadBuilder
    from geofractal.router.head.components import HeadConfig
    from geofractal.router.factory.prototype import LightweightPrototype

    B, S, D = 4, 16, 256

    # Head state dict
    def test_head_state_dict():
        config = HeadConfig(feature_dim=D)
        head1 = HeadBuilder.standard(config).build().to(DEVICE)

        torch.manual_seed(42)
        x = torch.randn(B, S, D).to(DEVICE)

        head1.eval()
        with torch.no_grad():
            out1 = head1(x)

        state = head1.state_dict()

        head2 = HeadBuilder.standard(config).build().to(DEVICE)
        head2.load_state_dict(state)
        head2.eval()

        with torch.no_grad():
            out2 = head2(x)

        assert torch.allclose(out1, out2, atol=1e-5)
        return True
    run_test("Head state_dict save/load", test_head_state_dict)

    # LightweightPrototype state dict
    def test_prototype_state_dict():
        prototype1 = LightweightPrototype(
            stream_dims={"a": 256, "b": 256},
            num_classes=10,
            hidden_dim=128,
        ).to(DEVICE)

        torch.manual_seed(42)
        inputs = {
            "a": torch.randn(B, 256).to(DEVICE),
            "b": torch.randn(B, 256).to(DEVICE),
        }

        prototype1.eval()
        with torch.no_grad():
            out1, _ = prototype1(inputs)

        state = prototype1.state_dict()

        prototype2 = LightweightPrototype(
            stream_dims={"a": 256, "b": 256},
            num_classes=10,
            hidden_dim=128,
        ).to(DEVICE)
        prototype2.load_state_dict(state)
        prototype2.eval()

        with torch.no_grad():
            out2, _ = prototype2(inputs)

        assert torch.allclose(out1, out2, atol=1e-5)
        return True
    run_test("LightweightPrototype state_dict", test_prototype_state_dict)

    # File save/load
    def test_file_save_load():
        config = HeadConfig(feature_dim=D)
        head1 = HeadBuilder.standard(config).build().to(DEVICE)

        torch.manual_seed(123)
        x = torch.randn(B, S, D).to(DEVICE)

        head1.eval()
        with torch.no_grad():
            out1 = head1(x)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(head1.state_dict(), f.name)
            path = f.name

        try:
            head2 = HeadBuilder.standard(config).build().to(DEVICE)
            head2.load_state_dict(torch.load(path, weights_only=True))
            head2.eval()

            with torch.no_grad():
                out2 = head2(x)

            assert torch.allclose(out1, out2, atol=1e-5)
        finally:
            os.unlink(path)

        return True
    run_test("File save/load", test_file_save_load)


# =============================================================================
# TEST 11: TRAINING LOOP
# =============================================================================

def test_training():
    test_section("11. Training Loop")

    from geofractal.router.head.builder import HeadBuilder
    from geofractal.router.head.components import HeadConfig
    from geofractal.router.factory.prototype import LightweightPrototype

    B = 8

    # Head training
    def test_head_training():
        config = HeadConfig(feature_dim=128)
        head = HeadBuilder.lightweight(config).build().to(DEVICE)

        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

        head.train()
        for step in range(20):
            x = torch.randn(B, 16, 128).to(DEVICE)

            optimizer.zero_grad()
            out = head(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

            if torch.isnan(loss):
                return f"NaN at step {step}"

        return True
    run_test("Head training 20 steps", test_head_training)

    # Prototype training
    def test_prototype_training():
        prototype = LightweightPrototype(
            stream_dims={"a": 256, "b": 256},
            num_classes=10,
            hidden_dim=128,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(prototype.parameters(), lr=1e-3)

        prototype.train()
        for step in range(20):
            inputs = {
                "a": torch.randn(B, 256).to(DEVICE),
                "b": torch.randn(B, 256).to(DEVICE),
            }
            labels = torch.randint(0, 10, (B,)).to(DEVICE)

            optimizer.zero_grad()
            logits, _ = prototype(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            if torch.isnan(loss):
                return f"NaN at step {step}"

        return True
    run_test("LightweightPrototype training 20 steps", test_prototype_training)

    # Loss decreases
    def test_loss_decreases():
        prototype = LightweightPrototype(
            stream_dims={"a": 128, "b": 128},
            num_classes=5,
            hidden_dim=64,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(prototype.parameters(), lr=1e-2)

        # Fixed dataset
        torch.manual_seed(42)
        fixed_inputs = {
            "a": torch.randn(32, 128).to(DEVICE),
            "b": torch.randn(32, 128).to(DEVICE),
        }
        fixed_labels = torch.randint(0, 5, (32,)).to(DEVICE)

        losses = []
        prototype.train()
        for step in range(50):
            optimizer.zero_grad()
            logits, _ = prototype(fixed_inputs)
            loss = F.cross_entropy(logits, fixed_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        first_5 = sum(losses[:5]) / 5
        last_5 = sum(losses[-5:]) / 5

        if last_5 >= first_5:
            return f"Loss did not decrease: {first_5:.4f} -> {last_5:.4f}"

        return True
    run_test("Loss decreases over training", test_loss_decreases)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("  GeoFractal Router - Factory & Prototype Builder Tests")
    print("="*70)

    test_headbuilder_basic()
    test_headbuilder_fluent()
    test_headbuilder_injection()
    test_headbuilder_gradients()
    test_composedhead_access()
    test_prototype_config()
    test_assembled_prototype()
    test_lightweight_prototype()
    test_factory_functions()
    test_serialization()
    test_training()

    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"\n  ✓ Passed: {PASSED}")
    print(f"  ✗ Failed: {FAILED}")
    print(f"  Total:   {PASSED + FAILED}")

    if ERRORS:
        print(f"\n  Errors:")
        for e in ERRORS:
            print(f"    - {e}")
        if len(ERRORS) > 10:
            print(f"    ... and {len(ERRORS) - 10} more")

    print("="*70)

    return FAILED == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)