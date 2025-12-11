# AgathaHeadRouter Real Model Integration Test
# Run in Colab with geofractal repo installed
# pip install git+https://github.com/huggingface/diffusers -U
# geofractal.router.prefab.agatha.head_router_tester.py

import torch
from transformers import (
    Qwen2ForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
)
from diffusers import AutoencoderKLFlux2

from geofractal.router.prefab.agatha.head_router import (
    AgathaHeadRouter,
    create_agatha_head,
    StreamType,
)


# =============================================================================
# EXTRACT FUNCTIONS
# =============================================================================

def qwen_extract(encoder, input_ids, attention_mask=None, **kwargs):
    """Extract hidden states from QWEN."""
    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1]


def dino_extract(encoder, pixel_values, **kwargs):
    """
    Extract features from DINOv3 ConvNeXt.

    ConvNeXt outputs:
        - last_hidden_state: [B, C, H, W]
        - pooler_output: [B, C] (global pooled)
    """
    outputs = encoder(pixel_values, output_hidden_states=True, return_dict=True)

    # Use pooler_output [B, C] -> [B, 1, C] for sequence format
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        return outputs.pooler_output.unsqueeze(1)  # [B, 1, D]

    # Fallback: flatten spatial dims [B, C, H, W] -> [B, H*W, C]
    x = outputs.last_hidden_state
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, H * W, C)


def flux_ae_extract(encoder, images, **kwargs):
    """
    Extract latents from FLUX.2 AE.

    Args:
        encoder: AutoencoderKLFlux2
        images: [B, 3, H, W] normalized to [-1, 1]

    Returns:
        latents: [B, L, D] where L = (H/8 * W/8), D = 32
    """
    latent_dist = encoder.encode(images).latent_dist
    latents = latent_dist.sample()  # [B, 32, H/8, W/8]

    # Flatten spatial dims: [B, C, H, W] -> [B, H*W, C]
    B, C, H, W = latents.shape
    return latents.permute(0, 2, 3, 1).reshape(B, H * W, C)


# =============================================================================
# LOAD MODELS
# =============================================================================

device = "cuda"
dtype = torch.bfloat16

print("Loading QWEN 2.5...")
qwen_name = "Qwen/Qwen2.5-1.5B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True)
qwen_model = Qwen2ForCausalLM.from_pretrained(
    qwen_name,
    torch_dtype=dtype,
    device_map=device,
    trust_remote_code=True,
)
qwen_model.eval()
qwen_dim = qwen_model.config.hidden_size
print(f"  QWEN: {qwen_dim}")

print("Loading DINOv3 ConvNeXt Large...")
dino_name = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
dino_processor = AutoImageProcessor.from_pretrained(dino_name)
dino_model = AutoModel.from_pretrained(dino_name, torch_dtype=dtype)
dino_model.to(device).eval()

# ConvNeXt uses hidden_sizes (list per stage), not hidden_size
dino_dim = dino_model.config.hidden_sizes[-1]  # Last stage dim
print(f"  DINO: {dino_dim}")

print("Loading FLUX.2 AE...")
flux_ae = AutoencoderKLFlux2.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    subfolder="vae",
    torch_dtype=dtype,
)
flux_ae.to(device).eval()
flux_latent_dim = flux_ae.config.latent_channels  # 32
print(f"  FLUX AE: latent_channels={flux_latent_dim}")

# =============================================================================
# CREATE HEAD ROUTER
# =============================================================================

print("\nCreating AgathaHeadRouter...")

head = create_agatha_head(
    embed_dim=1024,
    fingerprint_dim=64,
    fusion_type='adaptive',
).to(device)

head.attach_encoder(
    'qwen',
    qwen_model,
    embed_dim=qwen_dim,
    stream_type=StreamType.TEXT,
    extract_fn=qwen_extract,
    frozen=True,
)

head.attach_encoder(
    'dino',
    dino_model,
    embed_dim=dino_dim,
    stream_type=StreamType.GUIDANCE,
    extract_fn=dino_extract,
    frozen=True,
)

head.attach_encoder(
    'flux_ae',
    flux_ae,
    embed_dim=flux_latent_dim,
    stream_type=StreamType.IMAGE,
    extract_fn=flux_ae_extract,
    frozen=True,
)

print(f"Head: {head}")
for name, status in head.stream_status().items():
    print(f"  {name}: {status}")

# =============================================================================
# TEST FORWARD
# =============================================================================

print("\nPreparing inputs...")

texts = [
    "A beautiful sunset over the ocean with warm colors",
    "A cyberpunk city at night with neon lights",
]
text_encoded = qwen_tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=77,
)

fake_images_dino = torch.randn(2, 3, 224, 224, device=device, dtype=dtype)
fake_images_flux = torch.randn(2, 3, 256, 256, device=device, dtype=dtype)

inputs = {
    'qwen': text_encoded.input_ids.to(device),
    'dino': fake_images_dino,
    'flux_ae': fake_images_flux,
}

print(f"QWEN input: {inputs['qwen'].shape}")
print(f"DINO input: {inputs['dino'].shape}")
print(f"FLUX AE input: {inputs['flux_ae'].shape}")

print("\nRunning forward...")
head.debug_on()

with torch.no_grad():
    mail = head(inputs)

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

for name in mail.sources:
    m = mail[name]
    print(f"\n{name}:")
    print(f"  content: {list(m.content.shape)}")
    print(f"  fingerprint: {list(m.fingerprint.shape)}")
    print(f"  content norm: {m.content.norm(dim=-1).mean():.4f}")

print(f"\nFused: {list(mail.fused.shape)}")
print(f"Fused FP: {list(mail.fused_fingerprint.shape)}")

print("\nFingerprint similarities:")
for pair, sim in head.fingerprint_similarity().items():
    print(f"  {pair}: {sim:.4f}")

print(f"\nMemory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
print("\n✓ Done - Head router working with QWEN + DINOv3 ConvNeXt + FLUX.2 AE")