# lyra_loader.py - ComfyUI Compatible VAE Lyra Loader

"""
VAE Lyra Model Loader for ComfyUI
=================================
Supports v1 (standard fusion) and v2 (adaptive cantor) variants.
Includes Illustrious CLIP trained models.

Place adjacent to lyra.py and lyra_v2.py in your ComfyUI custom_nodes folder.
"""

import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from huggingface_hub import hf_hub_download

# ============================================================================
# MODEL REGISTRY
# ============================================================================

KNOWN_MODELS = {
    # V1 Models
    "AbstractPhil/vae-lyra": {
        "version": "v1",
        "description": "Original VAE Lyra with CLIP-L + T5-base",
        "modalities": ["clip", "t5"],
        "fusion": "cantor",
        "latent_dim": 768,
    },
    "AbstractPhil/vae-lyra-sdxl-t5xl": {
        "version": "v1",
        "description": "SDXL-compatible with CLIP-L + CLIP-G + T5-XL",
        "modalities": ["clip_l", "clip_g", "t5_xl"],
        "fusion": "cantor",
        "latent_dim": 2048,
    },
    # V2 Models
    "AbstractPhil/vae-lyra-xl-adaptive-cantor": {
        "version": "v2",
        "description": "Adaptive Cantor with decoupled T5 scales",
        "modalities": ["clip_l", "clip_g", "t5_xl_l", "t5_xl_g"],
        "fusion": "adaptive_cantor",
        "latent_dim": 2048,
    },
    # V2 Illustrious Models
    "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious": {
        "version": "v2",
        "description": "Adaptive Cantor with Illustrious CLIP weights",
        "modalities": ["clip_l", "clip_g", "t5_xl_l", "t5_xl_g"],
        "fusion": "adaptive_cantor",
        "latent_dim": 2048,
        "clip_weights": "illustrious",
        "clip_skip": 2,
    },
}

# ============================================================================
# VERSION DETECTION
# ============================================================================

V2_INDICATORS = [
    'modality_seq_lens',
    'binding_config',
    'alpha_init',
    'beta_init',
    'alpha_lr_scale',
    'beta_lr_scale',
    'beta_alpha_regularization',
    'kl_clamp_max',
    'logvar_clamp_min',
    'logvar_clamp_max',
]


def detect_lyra_version(config: Dict[str, Any], repo_id: Optional[str] = None) -> str:
    """Detect VAE Lyra version from configuration."""
    if repo_id and repo_id in KNOWN_MODELS:
        return KNOWN_MODELS[repo_id]['version']

    v2_score = sum(1 for key in V2_INDICATORS if key in config)

    if config.get('fusion_strategy') == 'adaptive_cantor':
        v2_score += 3

    return "v2" if v2_score >= 2 else "v1"


def get_known_model_info(repo_id: str) -> Optional[Dict[str, Any]]:
    """Get registry info for a known model."""
    return KNOWN_MODELS.get(repo_id)


def list_known_models():
    """Print all known VAE Lyra models."""
    print("\n" + "=" * 70)
    print("KNOWN VAE LYRA MODELS")
    print("=" * 70)

    for repo_id, info in KNOWN_MODELS.items():
        print(f"\n{repo_id}")
        print(f"  Version: {info['version']}")
        print(f"  Fusion: {info['fusion']}")
        print(f"  Modalities: {', '.join(info['modalities'])}")
        print(f"  Latent dim: {info['latent_dim']}")
        if info.get('clip_weights'):
            print(f"  CLIP weights: {info['clip_weights']}")
        if info.get('clip_skip'):
            print(f"  CLIP skip: {info['clip_skip']}")

    print("=" * 70 + "\n")


# ============================================================================
# CONFIG BUILDERS
# ============================================================================

def build_v1_config(config_dict: Dict[str, Any]):
    """Build v1 config object."""
    from .lyra import MultiModalVAEConfig as V1Config

    return V1Config(
        modality_dims=config_dict.get('modality_dims', {"clip": 768, "t5": 768}),
        latent_dim=config_dict.get('latent_dim', 768),
        seq_len=config_dict.get('seq_len', 77),
        encoder_layers=config_dict.get('encoder_layers', 3),
        decoder_layers=config_dict.get('decoder_layers', 3),
        hidden_dim=config_dict.get('hidden_dim', 1024),
        dropout=config_dict.get('dropout', 0.1),
        fusion_strategy=config_dict.get('fusion_strategy', 'cantor'),
        fusion_heads=config_dict.get('fusion_heads', 8),
        fusion_dropout=config_dict.get('fusion_dropout', 0.1),
        beta_kl=config_dict.get('beta_kl', 0.1),
        beta_reconstruction=config_dict.get('beta_reconstruction', 1.0),
        beta_cross_modal=config_dict.get('beta_cross_modal', 0.05),
        seed=config_dict.get('seed')
    )


def build_v2_config(config_dict: Dict[str, Any]):
    """Build v2 config object with all parameters including KL clamping."""
    from geofractal.model.vae.vae_lyra_v2 import MultiModalVAEConfig as V2Config

    # Default SDXL configuration with decoupled T5
    default_dims = {
        "clip_l": 768,
        "clip_g": 1280,
        "t5_xl_l": 2048,
        "t5_xl_g": 2048
    }

    default_seq_lens = {
        "clip_l": 77,
        "clip_g": 77,
        "t5_xl_l": 512,
        "t5_xl_g": 512
    }

    default_binding = {
        "clip_l": {"t5_xl_l": 0.3},
        "clip_g": {"t5_xl_g": 0.3},
        "t5_xl_l": {},
        "t5_xl_g": {}
    }

    return V2Config(
        # Modality configuration
        modality_dims=config_dict.get('modality_dims', default_dims),
        modality_seq_lens=config_dict.get('modality_seq_lens', default_seq_lens),
        binding_config=config_dict.get('binding_config', default_binding),

        # Latent space
        latent_dim=config_dict.get('latent_dim', 2048),
        seq_len=config_dict.get('seq_len', 77),

        # Architecture
        encoder_layers=config_dict.get('encoder_layers', 3),
        decoder_layers=config_dict.get('decoder_layers', 3),
        hidden_dim=config_dict.get('hidden_dim', 1024),
        dropout=config_dict.get('dropout', 0.1),

        # Fusion
        fusion_strategy=config_dict.get('fusion_strategy', 'adaptive_cantor'),
        fusion_heads=config_dict.get('fusion_heads', 8),
        fusion_dropout=config_dict.get('fusion_dropout', 0.1),
        cantor_depth=config_dict.get('cantor_depth', 8),
        cantor_local_window=config_dict.get('cantor_local_window', 3),

        # Adaptive parameters
        alpha_init=config_dict.get('alpha_init', 1.0),
        beta_init=config_dict.get('beta_init', 0.3),
        alpha_lr_scale=config_dict.get('alpha_lr_scale', 0.1),
        beta_lr_scale=config_dict.get('beta_lr_scale', 1.0),

        # Loss weights
        beta_kl=config_dict.get('beta_kl', 0.1),
        beta_reconstruction=config_dict.get('beta_reconstruction', 1.0),
        beta_cross_modal=config_dict.get('beta_cross_modal', 0.0),  # Disabled by default in v2
        beta_alpha_regularization=config_dict.get('beta_alpha_regularization', 0.01),

        # KL clamping (v2 feature)
        kl_clamp_max=config_dict.get('kl_clamp_max', 1.0),
        logvar_clamp_min=config_dict.get('logvar_clamp_min', -10.0),
        logvar_clamp_max=config_dict.get('logvar_clamp_max', 10.0),

        # Training
        use_amp=config_dict.get('use_amp', True),
        seed=config_dict.get('seed')
    )


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_vae_lyra(
        repo_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        force_version: Optional[str] = None
) -> torch.nn.Module:
    """
    Load VAE Lyra from HuggingFace Hub with automatic version detection.

    Args:
        repo_id: HuggingFace repository ID
        device: Device to load model on
        force_version: Force specific version ("v1" or "v2")

    Returns:
        Loaded VAE Lyra model
    """
    print(f"[VAE Lyra] Loading from: {repo_id}")

    registry_info = get_known_model_info(repo_id)
    if registry_info:
        print(f"[VAE Lyra] Known model: {registry_info['description']}")
        if registry_info.get('clip_weights'):
            print(f"[VAE Lyra] CLIP weights: {registry_info['clip_weights']}")

    # Download config
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model"
        )
    except Exception as e:
        raise ValueError(f"Could not download config from {repo_id}: {e}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Detect version
    version = force_version or detect_lyra_version(config_dict, repo_id)
    print(f"[VAE Lyra] Version: {version}")

    # Load appropriate version
    if version == "v1":
        return _load_v1(repo_id, device, config_dict)
    elif version == "v2":
        return _load_v2(repo_id, device, config_dict)
    else:
        raise ValueError(f"Unknown version: {version}")


def _load_v1(
        repo_id: str,
        device: str,
        config_dict: Dict[str, Any]
) -> torch.nn.Module:
    """Load VAE Lyra v1."""
    from .lyra import MultiModalVAE as V1Model

    print("[VAE Lyra] Loading v1 (standard fusion)...")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt",
        repo_type="model"
    )

    config = build_v1_config(config_dict)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = V1Model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[VAE Lyra] Loaded from step {checkpoint.get('global_step', '?')}")
    print(f"[VAE Lyra] Fusion: {config.fusion_strategy}")
    print(f"[VAE Lyra] Modalities: {list(config.modality_dims.keys())}")

    return model


def _load_v2(
        repo_id: str,
        device: str,
        config_dict: Dict[str, Any]
) -> torch.nn.Module:
    """Load VAE Lyra v2 (adaptive cantor)."""
    from geofractal.model.vae.vae_lyra_v2 import MultiModalVAE as V2Model

    print("[VAE Lyra] Loading v2 (adaptive cantor)...")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt",
        repo_type="model"
    )

    config = build_v2_config(config_dict)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = V2Model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[VAE Lyra] Loaded from step {checkpoint.get('global_step', '?')}")
    print(f"[VAE Lyra] Best loss: {checkpoint.get('best_loss', '?')}")
    print(f"[VAE Lyra] Fusion: {config.fusion_strategy}")
    print(f"[VAE Lyra] Modalities: {list(config.modality_dims.keys())}")
    print(f"[VAE Lyra] Seq lens: {config.modality_seq_lens}")

    # Show learned parameters
    fusion_params = model.get_fusion_params()
    if fusion_params:
        print("[VAE Lyra] Learned parameters:")
        if 'alphas' in fusion_params:
            for name, alpha in fusion_params['alphas'].items():
                print(f"  α_{name}: {torch.sigmoid(alpha).item():.4f}")
        if 'betas' in fusion_params:
            for name, beta in fusion_params['betas'].items():
                print(f"  β_{name}: {torch.sigmoid(beta).item():.4f}")

    return model


# ============================================================================
# LOCAL FILE LOADING (for ComfyUI models folder)
# ============================================================================

def load_vae_lyra_local(
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        force_version: Optional[str] = None
) -> torch.nn.Module:
    """
    Load VAE Lyra from local files.

    Args:
        model_path: Path to model.pt checkpoint
        config_path: Path to config.json (if None, looks for config.json next to model.pt)
        device: Device to load model on
        force_version: Force specific version ("v1" or "v2")

    Returns:
        Loaded VAE Lyra model
    """
    model_path = Path(model_path)

    if config_path is None:
        config_path = model_path.parent / "config.json"
    else:
        config_path = Path(config_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"[VAE Lyra] Loading from local: {model_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    version = force_version or detect_lyra_version(config_dict)
    print(f"[VAE Lyra] Version: {version}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if version == "v1":
        from geofractal.model.vae.vae_lyra import MultiModalVAE as V1Model
        config = build_v1_config(config_dict)
        model = V1Model(config)
    else:
        from geofractal.model.vae.vae_lyra_v2 import MultiModalVAE as V2Model
        config = build_v2_config(config_dict)
        model = V2Model(config)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[VAE Lyra] Loaded successfully")
    return model


# ============================================================================
# MODEL INFO (without loading)
# ============================================================================

def get_model_info(repo_id: str) -> Dict[str, Any]:
    """Get information about a VAE Lyra model without loading it."""
    registry_info = get_known_model_info(repo_id)

    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model"
        )
        with open(config_path) as f:
            config_dict = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not download config from {repo_id}: {e}")

    version = detect_lyra_version(config_dict, repo_id)

    info = {
        'repo_id': repo_id,
        'version': version,
        'fusion_strategy': config_dict.get('fusion_strategy', 'unknown'),
        'modality_dims': config_dict.get('modality_dims', {}),
        'latent_dim': config_dict.get('latent_dim', 'unknown'),
        'is_adaptive': version == 'v2',
    }

    if version == 'v2':
        info['modality_seq_lens'] = config_dict.get('modality_seq_lens', {})
        info['binding_config'] = config_dict.get('binding_config', {})
        info['kl_clamp_max'] = config_dict.get('kl_clamp_max', 1.0)

    if registry_info:
        info['description'] = registry_info['description']
        if registry_info.get('clip_weights'):
            info['clip_weights'] = registry_info['clip_weights']
        if registry_info.get('clip_skip'):
            info['clip_skip'] = registry_info['clip_skip']

    return info


def print_model_info(repo_id: str):
    """Print formatted information about a VAE Lyra model."""
    info = get_model_info(repo_id)

    print(f"\n{'=' * 60}")
    print(f"VAE LYRA MODEL INFO")
    print(f"{'=' * 60}")
    print(f"Repository: {info['repo_id']}")
    print(f"Version: {info['version']}")

    if 'description' in info:
        print(f"Description: {info['description']}")

    print(f"\nArchitecture:")
    print(f"  Fusion: {info['fusion_strategy']}")
    print(f"  Latent dim: {info['latent_dim']}")

    if info.get('clip_weights'):
        print(f"  CLIP weights: {info['clip_weights']}")
    if info.get('clip_skip'):
        print(f"  CLIP skip: {info['clip_skip']}")

    print(f"\nModalities:")
    for name, dim in info['modality_dims'].items():
        seq_info = ""
        if info.get('modality_seq_lens'):
            seq_len = info['modality_seq_lens'].get(name, '?')
            seq_info = f" @ {seq_len} tokens"
        print(f"  {name}: {dim}d{seq_info}")

    if info.get('binding_config'):
        print(f"\nBinding (hard masking):")
        for target, sources in info['binding_config'].items():
            if sources:
                print(f"  {target} ← {list(sources.keys())}")
            else:
                print(f"  {target} (independent)")

    print(f"{'=' * 60}\n")


# ============================================================================
# COMFYUI INTEGRATION HELPERS
# ============================================================================

def get_available_models() -> List[str]:
    """Get list of known model repo IDs for ComfyUI dropdown."""
    return list(KNOWN_MODELS.keys())


def get_model_choices() -> List[Tuple[str, str]]:
    """Get (display_name, repo_id) tuples for ComfyUI dropdown."""
    choices = []
    for repo_id, info in KNOWN_MODELS.items():
        short_name = repo_id.split('/')[-1]
        display = f"{short_name} ({info['version']}, {info['fusion']})"
        choices.append((display, repo_id))
    return choices