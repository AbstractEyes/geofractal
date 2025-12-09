"""
Gradient Flow Diagnostic for GeoFractal Router
Tests that gradients actually reach critical components.
"""

import torch
import torch.nn as nn


def test_gradient_flow():
    """Test that gradients flow to all critical components."""
    print("\n" + "=" * 60)
    print("  Gradient Flow Diagnostic")
    print("=" * 60)

    from geofractal.router.collective import RouterCollective
    from geofractal.router.config import CollectiveConfig
    from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 4

    config = CollectiveConfig(
        feature_dim=128,
        num_classes=10,
        device=DEVICE,
    )

    collective = RouterCollective.from_specs(
        stream_specs=[
            StreamSpec.feature_vector("a", input_dim=256, feature_dim=128),
            StreamSpec.feature_vector("b", input_dim=256, feature_dim=128),
        ],
        config=config,
        head_spec=HeadSpec.lightweight(feature_dim=128),
        fusion_spec=FusionSpec.gated(output_dim=128),
    ).to(DEVICE)

    # Zero all gradients
    collective.zero_grad()

    # Forward pass
    inputs = {
        "a": torch.randn(B, 256, device=DEVICE),
        "b": torch.randn(B, 256, device=DEVICE),
    }
    labels = torch.randint(0, 10, (B,), device=DEVICE)

    logits, info = collective(inputs)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # Check gradient flow to critical components
    results = {}

    # 1. Fingerprints (CRITICAL - identity of each head)
    for name in collective.stream_names:
        fp = collective.heads[name].fingerprint
        has_grad = fp.grad is not None and fp.grad.abs().sum() > 0
        grad_norm = fp.grad.norm().item() if has_grad else 0
        results[f"head[{name}].fingerprint"] = (has_grad, grad_norm)

    # 2. Anchor banks (CRITICAL - geometric structure)
    for name in collective.stream_names:
        anchors = collective.heads[name].anchors
        if hasattr(anchors, 'anchors'):
            anchor_param = anchors.anchors
            has_grad = anchor_param.grad is not None and anchor_param.grad.abs().sum() > 0
            grad_norm = anchor_param.grad.norm().item() if has_grad else 0
            results[f"head[{name}].anchors.anchors"] = (has_grad, grad_norm)
        if hasattr(anchors, 'fp_to_affinity'):
            for pname, p in anchors.fp_to_affinity.named_parameters():
                has_grad = p.grad is not None and p.grad.abs().sum() > 0
                grad_norm = p.grad.norm().item() if has_grad else 0
                results[f"head[{name}].anchors.fp_to_affinity.{pname}"] = (has_grad, grad_norm)
                break  # Just check first

    # 3. Router fp_to_bias (CRITICAL - fingerprint affects routing)
    for name in collective.stream_names:
        router = collective.heads[name].router
        if hasattr(router, 'fp_to_bias'):
            for pname, p in router.fp_to_bias.named_parameters():
                has_grad = p.grad is not None and p.grad.abs().sum() > 0
                grad_norm = p.grad.norm().item() if has_grad else 0
                results[f"head[{name}].router.fp_to_bias.{pname}"] = (has_grad, grad_norm)
                break

    # 4. Gate (fingerprint gating)
    for name in collective.stream_names:
        gate = collective.heads[name].gate
        for pname, p in gate.named_parameters():
            if p.requires_grad:
                has_grad = p.grad is not None and p.grad.abs().sum() > 0
                grad_norm = p.grad.norm().item() if has_grad else 0
                results[f"head[{name}].gate.{pname}"] = (has_grad, grad_norm)
                break

    # 5. Attention projections
    for name in collective.stream_names:
        attn = collective.heads[name].attention
        if hasattr(attn, 'q_proj'):
            p = attn.q_proj.weight
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            grad_norm = p.grad.norm().item() if has_grad else 0
            results[f"head[{name}].attention.q_proj"] = (has_grad, grad_norm)

    # 6. Combiner weights
    for name in collective.stream_names:
        combiner = collective.heads[name].combiner
        if hasattr(combiner, 'weights'):
            p = combiner.weights
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            grad_norm = p.grad.norm().item() if has_grad else 0
            results[f"head[{name}].combiner.weights"] = (has_grad, grad_norm)

    # 7. Fusion gate network
    if hasattr(collective.fusion, 'gate_net'):
        for pname, p in collective.fusion.gate_net.named_parameters():
            if p.requires_grad:
                has_grad = p.grad is not None and p.grad.abs().sum() > 0
                grad_norm = p.grad.norm().item() if has_grad else 0
                results[f"fusion.gate_net.{pname}"] = (has_grad, grad_norm)
                break

    # 8. Stream projections
    for name in collective.stream_names:
        stream = collective.streams[name]
        for pname, p in stream.named_parameters():
            if p.requires_grad:
                has_grad = p.grad is not None and p.grad.abs().sum() > 0
                grad_norm = p.grad.norm().item() if has_grad else 0
                results[f"stream[{name}].{pname}"] = (has_grad, grad_norm)
                break

    # 9. Classifier
    for pname, p in collective.classifier.named_parameters():
        if p.requires_grad:
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            grad_norm = p.grad.norm().item() if has_grad else 0
            results[f"classifier.{pname}"] = (has_grad, grad_norm)
            break

    # Report
    print("\nComponent                                    | Grad? | Norm")
    print("-" * 60)

    failed = []
    for component, (has_grad, grad_norm) in sorted(results.items()):
        status = "✓" if has_grad else "✗"
        print(f"{component:44} | {status:5} | {grad_norm:.2e}")
        if not has_grad:
            failed.append(component)

    print("-" * 60)

    if failed:
        print(f"\n⚠️  {len(failed)} components have NO gradients:")
        for f in failed:
            print(f"   - {f}")
        return False
    else:
        print(f"\n✓ All {len(results)} checked components receive gradients")
        return True


if __name__ == "__main__":
    success = test_gradient_flow()
    exit(0 if success else 1)