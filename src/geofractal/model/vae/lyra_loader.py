# geofractal/model/vae/loader.py

"""
VAE Lyra Model Loader - Intelligent Version Detection and Loading
=================================================================

Automatically detects and loads the correct VAE Lyra variant from HuggingFace Hub
based on configuration parameters.

Supported Variants:
- v1: Standard multi-modal VAE with fusion strategies (concatenate, attention,
      gated, cantor, geometric, hierarchical)
      Examples: vae-lyra, vae-lyra-sdxl-t5xl

- v2: Adaptive Cantor VAE with learned alpha/beta parameters, variable sequence
      lengths, decoupled T5 scales, and binding configuration
      Examples: vae-lyra-xl-adaptive-cantor, vae-lyra-xl-adaptive-cantor-illustrious

Known Models:
- AbstractPhil/vae-lyra: Original CLIP-L + T5-base (v1)
- AbstractPhil/vae-lyra-sdxl-t5xl: SDXL with CLIP-L + CLIP-G + T5-XL (v1)
- AbstractPhil/vae-lyra-xl-adaptive-cantor: Adaptive Cantor with decoupled T5 (v2)
- AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious: Illustrious variant (v2)

Usage:
    from geofractal.model.vae.loader import load_vae_lyra

    # Auto-detect version and best checkpoint
    model = load_vae_lyra("AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious")

    # Load specific checkpoint from weights/ folder
    model = load_vae_lyra(
        "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious",
        checkpoint="lyra_illustrious_step_9000.safetensors"
    )

    # List all known models
    from geofractal.model.vae.loader import list_known_models
    list_known_models()

Author: AbstractPhil
"""

import torch
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import asdict

# Try importing safetensors
try:
    from safetensors.torch import load_file as load_safetensors

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("⚠️ safetensors not installed - .safetensors files will not be supported")

# Try importing huggingface_hub
try:
    from huggingface_hub import hf_hub_download, list_repo_files

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("⚠️ huggingface_hub not installed - remote loading disabled")

# ============================================================================
# MODEL REGISTRY
# ============================================================================

KNOWN_MODELS = {
    # ─────────────────────────────────────────────────────────────────────────
    # V1 Models - Standard fusion strategies
    # ─────────────────────────────────────────────────────────────────────────
    "AbstractPhil/vae-lyra": {
        "version": "v1",
        "description": "Original VAE Lyra with CLIP-L + T5-base",
        "modalities": ["clip", "t5"],
        "fusion": "cantor",
        "latent_dim": 768,
        "checkpoint_format": "model.pt",
        "recommended_for": "General text embedding transformation"
    },
    "AbstractPhil/vae-lyra-sdxl-t5xl": {
        "version": "v1",
        "description": "SDXL-compatible with CLIP-L + CLIP-G + T5-XL",
        "modalities": ["clip_l", "clip_g", "t5_xl"],
        "fusion": "cantor",
        "latent_dim": 2048,
        "checkpoint_format": "model.pt",
        "recommended_for": "SDXL text encoder replacement"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # V2 Models - Adaptive Cantor with learned parameters
    # ─────────────────────────────────────────────────────────────────────────
    "AbstractPhil/vae-lyra-xl-adaptive-cantor": {
        "version": "v2",
        "description": "Adaptive Cantor with decoupled T5 scales and learned parameters",
        "modalities": ["clip_l", "clip_g", "t5_xl_l", "t5_xl_g"],
        "fusion": "adaptive_cantor",
        "latent_dim": 2048,
        "checkpoint_format": "safetensors",
        "weights_folder": "weights",
        "has_adaptive_params": True,
        "has_variable_seq_lens": True,
        "recommended_for": "Advanced SDXL with geometric consciousness"
    },
    "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious": {
        "version": "v2",
        "description": "Illustrious variant - trained on anime/illustration data with enhanced generalization",
        "modalities": ["clip_l", "clip_g", "t5_xl_l", "t5_xl_g"],
        "fusion": "adaptive_cantor",
        "latent_dim": 2048,
        "checkpoint_format": "safetensors",
        "weights_folder": "weights",
        "has_adaptive_params": True,
        "has_variable_seq_lens": True,
        "has_binding_groups": True,
        "recommended_for": "Illustrious/anime generation with SDXL architecture"
    },
}

# Aliases for convenience
MODEL_ALIASES = {
    "lyra": "AbstractPhil/vae-lyra",
    "lyra-sdxl": "AbstractPhil/vae-lyra-sdxl-t5xl",
    "lyra-xl": "AbstractPhil/vae-lyra-xl-adaptive-cantor",
    "lyra-illustrious": "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious",
    "lyra-cantor": "AbstractPhil/vae-lyra-xl-adaptive-cantor",
}


def resolve_repo_id(repo_id: str) -> str:
    """Resolve aliases to full repo IDs."""
    return MODEL_ALIASES.get(repo_id, repo_id)


def list_known_models():
    """List all known VAE Lyra models with descriptions."""
    print("\n" + "=" * 80)
    print("KNOWN VAE LYRA MODELS")
    print("=" * 80)

    for repo_id, info in KNOWN_MODELS.items():
        print(f"\n📦 {repo_id}")
        print(f"   Version: {info['version']}")
        print(f"   Description: {info['description']}")
        print(f"   Modalities: {', '.join(info['modalities'])}")
        print(f"   Fusion: {info['fusion']}")
        print(f"   Latent dim: {info['latent_dim']}")
        print(f"   Checkpoint format: {info['checkpoint_format']}")
        if info.get('has_adaptive_params'):
            print(f"   🎯 Learned alpha/beta parameters")
        if info.get('has_variable_seq_lens'):
            print(f"   📏 Variable sequence lengths")
        if info.get('has_binding_groups'):
            print(f"   🔗 Binding group configuration")
        print(f"   Use case: {info['recommended_for']}")

    print("\n📝 Aliases:")
    for alias, full_id in MODEL_ALIASES.items():
        print(f"   {alias} → {full_id}")

    print("\n" + "=" * 80 + "\n")


def get_known_model_info(repo_id: str) -> Optional[Dict[str, Any]]:
    """Get registry info for a known model."""
    repo_id = resolve_repo_id(repo_id)
    return KNOWN_MODELS.get(repo_id)


# ============================================================================
# VERSION DETECTION
# ============================================================================

def detect_lyra_version(config: Dict[str, Any], repo_id: Optional[str] = None) -> str:
    """
    Detect VAE Lyra version from configuration.

    Args:
        config: Configuration dictionary
        repo_id: Optional repository ID for registry lookup

    Returns:
        Version string: "v1" or "v2"
    """
    # Check registry first if repo_id provided
    if repo_id:
        repo_id = resolve_repo_id(repo_id)
        if repo_id in KNOWN_MODELS:
            return KNOWN_MODELS[repo_id]['version']

    # v2 signature features
    v2_indicators = [
        'modality_seq_lens',  # Variable sequence lengths (v2)
        'binding_config',  # Binding configuration (v2)
        'alpha_init',  # Adaptive alpha parameters (v2)
        'beta_init',  # Adaptive beta parameters (v2)
        'alpha_lr_scale',  # Alpha learning rate scaling (v2)
        'beta_lr_scale',  # Beta learning rate scaling (v2)
        'beta_alpha_regularization'  # Alpha regularization (v2)
    ]

    # Check for v2 indicators
    v2_score = sum(1 for key in v2_indicators if key in config)

    # Also check for adaptive_cantor fusion strategy
    if config.get('fusion_strategy') == 'adaptive_cantor':
        v2_score += 3  # Strong indicator

    # Check for decoupled T5 (t5_xl_l, t5_xl_g)
    modality_dims = config.get('modality_dims', {})
    if 't5_xl_l' in modality_dims or 't5_xl_g' in modality_dims:
        v2_score += 2

    # Decision threshold
    if v2_score >= 2:
        return "v2"
    else:
        return "v1"


def detect_model_variant(config: Dict[str, Any], repo_id: Optional[str] = None) -> str:
    """
    Detect specific model variant for more precise loading.

    Returns: "base", "sdxl", "xl-adaptive", "illustrious"
    """
    if repo_id:
        repo_id = resolve_repo_id(repo_id)
        if "illustrious" in repo_id.lower():
            return "illustrious"
        if "adaptive-cantor" in repo_id.lower():
            return "xl-adaptive"
        if "sdxl" in repo_id.lower():
            return "sdxl"

    # Detect from config
    modality_dims = config.get('modality_dims', {})

    if 't5_xl_l' in modality_dims and 't5_xl_g' in modality_dims:
        # Decoupled T5 = xl-adaptive or illustrious
        model_name = config.get('model_name', '').lower()
        if 'illustrious' in model_name:
            return "illustrious"
        return "xl-adaptive"

    if 'clip_g' in modality_dims:
        return "sdxl"

    return "base"


# ============================================================================
# CHECKPOINT DISCOVERY
# ============================================================================

def discover_checkpoints(repo_id: str) -> Dict[str, List[str]]:
    """
    Discover available checkpoints in a repository.

    Returns:
        Dict with 'weights' (safetensors in weights/), 'root' (files in root)
    """
    if not HAS_HF_HUB:
        return {'weights': [], 'root': []}

    try:
        files = list_repo_files(repo_id, repo_type="model")
    except Exception as e:
        print(f"⚠️ Could not list files in {repo_id}: {e}")
        return {'weights': [], 'root': []}

    result = {
        'weights': [],  # Files in weights/ folder
        'root': [],  # Files in root
    }

    for f in files:
        if f.endswith('.safetensors') or f.endswith('.pt'):
            if f.startswith('weights/'):
                result['weights'].append(f.replace('weights/', ''))
            else:
                result['root'].append(f)

    return result


def find_best_checkpoint(repo_id: str, config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Find the best checkpoint file to load.

    Returns:
        (checkpoint_path, format) where format is 'safetensors' or 'pt'
    """
    repo_id = resolve_repo_id(repo_id)
    model_name = config.get('model_name', 'lyra')

    # Registry hint
    registry_info = get_known_model_info(repo_id)
    preferred_format = registry_info.get('checkpoint_format', 'safetensors') if registry_info else 'safetensors'
    weights_folder = registry_info.get('weights_folder', 'weights') if registry_info else 'weights'

    # Priority order for checkpoint discovery
    candidates = []

    if preferred_format == 'safetensors':
        # Prefer safetensors in weights/ folder
        candidates.extend([
            (f"{weights_folder}/{model_name}_best.safetensors", 'safetensors'),
            (f"{weights_folder}/{model_name}_illustrious_best.safetensors", 'safetensors'),
            (f"{weights_folder}/lyra_best.safetensors", 'safetensors'),
            (f"{weights_folder}/lyra_illustrious_best.safetensors", 'safetensors'),
            ("model.safetensors", 'safetensors'),
            ("model.pt", 'pt'),
        ])
    else:
        # Prefer .pt files
        candidates.extend([
            ("model.pt", 'pt'),
            (f"{weights_folder}/{model_name}_best.safetensors", 'safetensors'),
            ("model.safetensors", 'safetensors'),
        ])

    # Try each candidate
    for ckpt_path, fmt in candidates:
        try:
            _ = hf_hub_download(repo_id=repo_id, filename=ckpt_path, repo_type="model")
            return ckpt_path, fmt
        except:
            continue

    raise FileNotFoundError(f"No checkpoint found in {repo_id}. Tried: {[c[0] for c in candidates]}")


def extract_step_from_filename(filename: str) -> Optional[int]:
    """Extract training step from checkpoint filename."""
    patterns = [
        r'step[_-](\d+)',
        r'_(\d+)\.(?:safetensors|pt)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return None


# ============================================================================
# WEIGHT LOADING
# ============================================================================

def load_weights(
        path: str,
        device: str = "cpu",
        is_safetensors: Optional[bool] = None
) -> Dict[str, torch.Tensor]:
    """
    Load weights from file (safetensors or pt).

    Args:
        path: Path to weight file
        device: Device to load to
        is_safetensors: Force format detection (None = auto-detect from extension)

    Returns:
        State dict
    """
    if is_safetensors is None:
        is_safetensors = path.endswith('.safetensors')

    if is_safetensors:
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors not installed. Run: pip install safetensors")
        return load_safetensors(path, device=device)
    else:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                return checkpoint['state_dict']
        return checkpoint


def load_training_state(path: str, device: str = "cpu") -> Optional[Dict[str, Any]]:
    """
    Load training state (optimizer, step, loss) from .pt checkpoint.
    Returns None for safetensors (no training state).
    """
    if path.endswith('.safetensors'):
        return None

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        return {
            'global_step': checkpoint.get('global_step'),
            'epoch': checkpoint.get('epoch'),
            'best_loss': checkpoint.get('best_loss'),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        }
    return None


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_vae_lyra(
        repo_id: str,
        checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        force_version: Optional[str] = None,
        return_info: bool = False
) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Dict[str, Any]]]:
    """
    Load VAE Lyra from HuggingFace Hub with automatic version detection.

    Args:
        repo_id: HuggingFace repository ID or alias (e.g., "lyra-illustrious")
        checkpoint: Specific checkpoint file (e.g., "lyra_step_9000.safetensors")
                   If in weights/ folder, can omit the prefix.
        device: Device to load model on
        force_version: Force specific version ("v1" or "v2"), otherwise auto-detect
        return_info: If True, return (model, info_dict) tuple

    Returns:
        Loaded VAE Lyra model, or (model, info) if return_info=True

    Examples:
        # Auto-detect best checkpoint
        model = load_vae_lyra("AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious")

        # Use alias
        model = load_vae_lyra("lyra-illustrious")

        # Specific checkpoint
        model = load_vae_lyra("lyra-illustrious", checkpoint="lyra_step_9000.safetensors")

        # Get info alongside model
        model, info = load_vae_lyra("lyra-illustrious", return_info=True)
    """
    repo_id = resolve_repo_id(repo_id)
    print(f"🔍 Loading VAE Lyra from: {repo_id}")

    # Show registry info if available
    registry_info = get_known_model_info(repo_id)
    if registry_info:
        print(f"📋 Known model: {registry_info['description']}")

    # Download config
    if not HAS_HF_HUB:
        raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", repo_type="model")
    except Exception as e:
        raise ValueError(f"Could not download config from {repo_id}: {e}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Detect version
    if force_version:
        version = force_version
        print(f"⚙️  Forced version: {version}")
    else:
        version = detect_lyra_version(config_dict, repo_id)
        print(f"✓ Detected version: {version}")

    # Detect variant
    variant = detect_model_variant(config_dict, repo_id)
    print(f"✓ Variant: {variant}")

    # Find checkpoint
    if checkpoint:
        # User specified checkpoint
        # Auto-add weights/ prefix if needed
        if not checkpoint.startswith('weights/') and not checkpoint.startswith('/'):
            ckpt_path = f"weights/{checkpoint}"
        else:
            ckpt_path = checkpoint

        ckpt_format = 'safetensors' if checkpoint.endswith('.safetensors') else 'pt'

        try:
            weights_path = hf_hub_download(repo_id=repo_id, filename=ckpt_path, repo_type="model")
            print(f"✓ Checkpoint: {ckpt_path}")
        except Exception as e:
            # Try without weights/ prefix
            try:
                weights_path = hf_hub_download(repo_id=repo_id, filename=checkpoint, repo_type="model")
                ckpt_path = checkpoint
                print(f"✓ Checkpoint: {ckpt_path}")
            except:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint} (tried {ckpt_path} and {checkpoint})")
    else:
        # Auto-discover best checkpoint
        ckpt_path, ckpt_format = find_best_checkpoint(repo_id, config_dict)
        weights_path = hf_hub_download(repo_id=repo_id, filename=ckpt_path, repo_type="model")
        print(f"✓ Auto-selected checkpoint: {ckpt_path}")

    # Extract step from filename
    step = extract_step_from_filename(ckpt_path)
    if step:
        print(f"✓ Training step: {step:,}")

    # Load appropriate version
    if version == "v1":
        model = _load_vae_lyra_v1(config_dict, weights_path, device)
    elif version == "v2":
        model = _load_vae_lyra_v2(config_dict, weights_path, device)
    else:
        raise ValueError(f"Unknown version: {version}")

    # Compile info
    info = {
        'repo_id': repo_id,
        'version': version,
        'variant': variant,
        'checkpoint': ckpt_path,
        'step': step,
        'config': config_dict,
        'device': device,
        'registry_info': registry_info,
    }

    # Try to get training state
    training_state = load_training_state(weights_path, device)
    if training_state:
        info['training_state'] = training_state
        if training_state.get('best_loss'):
            print(f"✓ Best loss: {training_state['best_loss']:.6f}")

    if return_info:
        return model, info
    return model


def _load_vae_lyra_v1(
        config_dict: Dict[str, Any],
        weights_path: str,
        device: str
) -> torch.nn.Module:
    """Load VAE Lyra v1 (standard fusion strategies)."""
    from geofractal.model.vae.vae_lyra import MultiModalVAE, MultiModalVAEConfig

    print("📦 Loading VAE Lyra v1...")

    vae_config = MultiModalVAEConfig(
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

    model = MultiModalVAE(vae_config)
    state_dict = load_weights(weights_path, device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"✓ Fusion strategy: {vae_config.fusion_strategy}")
    print(f"✓ Modalities: {list(vae_config.modality_dims.keys())}")
    print(f"✓ Latent dimension: {vae_config.latent_dim}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def _load_vae_lyra_v2(
        config_dict: Dict[str, Any],
        weights_path: str,
        device: str
) -> torch.nn.Module:
    """Load VAE Lyra v2 (adaptive Cantor with learned parameters)."""
    from geofractal.model.vae.vae_lyra_v2 import MultiModalVAE, MultiModalVAEConfig

    print("📦 Loading VAE Lyra v2 (Adaptive Cantor)...")

    vae_config = MultiModalVAEConfig(
        modality_dims=config_dict.get('modality_dims'),
        modality_seq_lens=config_dict.get('modality_seq_lens'),
        binding_config=config_dict.get('binding_config'),
        latent_dim=config_dict.get('latent_dim', 2048),
        seq_len=config_dict.get('seq_len', 77),
        encoder_layers=config_dict.get('encoder_layers', 3),
        decoder_layers=config_dict.get('decoder_layers', 3),
        hidden_dim=config_dict.get('hidden_dim', 1024),
        dropout=config_dict.get('dropout', 0.1),
        fusion_strategy=config_dict.get('fusion_strategy', 'adaptive_cantor'),
        fusion_heads=config_dict.get('fusion_heads', 8),
        fusion_dropout=config_dict.get('fusion_dropout', 0.1),
        cantor_depth=config_dict.get('cantor_depth', 8),
        cantor_local_window=config_dict.get('cantor_local_window', 3),
        alpha_init=config_dict.get('alpha_init', 1.0),
        beta_init=config_dict.get('beta_init', 0.3),
        alpha_lr_scale=config_dict.get('alpha_lr_scale', 0.1),
        beta_lr_scale=config_dict.get('beta_lr_scale', 1.0),
        beta_kl=config_dict.get('beta_kl', 0.1),
        beta_reconstruction=config_dict.get('beta_reconstruction', 1.0),
        beta_cross_modal=config_dict.get('beta_cross_modal', 0.05),
        beta_alpha_regularization=config_dict.get('beta_alpha_regularization', 0.01),
        seed=config_dict.get('seed')
    )

    model = MultiModalVAE(vae_config)
    state_dict = load_weights(weights_path, device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"✓ Fusion strategy: {vae_config.fusion_strategy}")
    print(f"✓ Modalities: {list(vae_config.modality_dims.keys())}")
    print(f"✓ Latent dimension: {vae_config.latent_dim}")
    if vae_config.modality_seq_lens:
        print(f"✓ Sequence lengths: {vae_config.modality_seq_lens}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Show learned fusion parameters
    if hasattr(model, 'get_fusion_params'):
        fusion_params = model.get_fusion_params()
        if fusion_params:
            print(f"\n📊 Learned Fusion Parameters:")
            if 'alphas' in fusion_params and fusion_params['alphas']:
                print(f"   Alpha (visibility):")
                for name, alpha in fusion_params['alphas'].items():
                    val = torch.sigmoid(alpha).item() if isinstance(alpha, torch.Tensor) else alpha
                    print(f"     • {name}: {val:.4f}")
            if 'betas' in fusion_params and fusion_params['betas']:
                print(f"   Beta (cross-group):")
                for name, beta in fusion_params['betas'].items():
                    val = torch.sigmoid(beta).item() if isinstance(beta, torch.Tensor) else beta
                    print(f"     • {name}: {val:.4f}")

    return model


# ============================================================================
# LOCAL LOADING
# ============================================================================

def load_vae_lyra_local(
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        force_version: Optional[str] = None
) -> torch.nn.Module:
    """
    Load VAE Lyra from local files.

    Args:
        checkpoint_path: Path to .pt or .safetensors file
        config_path: Path to config.json (auto-discovers if None)
        device: Device to load model on
        force_version: Force specific version

    Returns:
        Loaded VAE Lyra model
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Find config
    if config_path is None:
        # Try common locations
        candidates = [
            checkpoint_path.parent / "config.json",
            checkpoint_path.parent.parent / "config.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found. Searched: {candidates}")

    print(f"🔍 Loading VAE Lyra from local files...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Config: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Detect version
    if force_version:
        version = force_version
    else:
        version = detect_lyra_version(config_dict)
    print(f"✓ Detected version: {version}")

    # Load model
    if version == "v1":
        model = _load_vae_lyra_v1(config_dict, str(checkpoint_path), device)
    else:
        model = _load_vae_lyra_v2(config_dict, str(checkpoint_path), device)

    return model


# ============================================================================
# MODEL INFORMATION
# ============================================================================

def get_model_info(repo_id: str) -> Dict[str, Any]:
    """
    Get information about a VAE Lyra model without loading weights.

    Args:
        repo_id: HuggingFace repository ID or alias

    Returns:
        Dictionary with model information
    """
    repo_id = resolve_repo_id(repo_id)
    print(f"🔍 Inspecting model: {repo_id}")

    registry_info = get_known_model_info(repo_id)

    # Download config
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", repo_type="model")
    with open(config_path) as f:
        config_dict = json.load(f)

    version = detect_lyra_version(config_dict, repo_id)
    variant = detect_model_variant(config_dict, repo_id)

    # Discover checkpoints
    checkpoints = discover_checkpoints(repo_id)

    info = {
        'repo_id': repo_id,
        'version': version,
        'variant': variant,
        'fusion_strategy': config_dict.get('fusion_strategy', 'unknown'),
        'modality_dims': config_dict.get('modality_dims', {}),
        'modality_seq_lens': config_dict.get('modality_seq_lens', {}),
        'latent_dim': config_dict.get('latent_dim', 'unknown'),
        'has_adaptive_params': version == 'v2' and config_dict.get('fusion_strategy') == 'adaptive_cantor',
        'has_variable_seq_lens': 'modality_seq_lens' in config_dict,
        'has_binding_config': 'binding_config' in config_dict,
        'available_checkpoints': checkpoints,
        'config': config_dict,
    }

    if registry_info:
        info['registry_description'] = registry_info['description']
        info['recommended_for'] = registry_info['recommended_for']

    return info


def print_model_info(repo_id: str):
    """Print formatted information about a VAE Lyra model."""
    info = get_model_info(repo_id)

    print(f"\n{'=' * 80}")
    print(f"VAE LYRA MODEL INFO")
    print(f"{'=' * 80}")
    print(f"Repository: {info['repo_id']}")
    print(f"Version: {info['version']}")
    print(f"Variant: {info['variant']}")

    if 'registry_description' in info:
        print(f"Description: {info['registry_description']}")
        print(f"Recommended for: {info['recommended_for']}")

    print(f"\nArchitecture:")
    print(f"  Fusion Strategy: {info['fusion_strategy']}")
    print(f"  Latent Dimension: {info['latent_dim']}")

    print(f"\nModalities:")
    for name, dim in info['modality_dims'].items():
        seq_len = info['modality_seq_lens'].get(name, 77)
        print(f"  • {name}: {dim}d @ {seq_len} tokens")

    if info['available_checkpoints']['weights']:
        print(f"\n📦 Available Checkpoints (weights/):")
        for ckpt in info['available_checkpoints']['weights'][:10]:
            step = extract_step_from_filename(ckpt)
            step_str = f" (step {step:,})" if step else ""
            print(f"  • {ckpt}{step_str}")
        if len(info['available_checkpoints']['weights']) > 10:
            print(f"  ... and {len(info['available_checkpoints']['weights']) - 10} more")

    print(f"{'=' * 80}\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_lyra_illustrious(
        checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.nn.Module:
    """Convenience function to load the Illustrious variant."""
    return load_vae_lyra(
        "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious",
        checkpoint=checkpoint,
        device=device
    )


def load_lyra_xl(
        checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.nn.Module:
    """Convenience function to load the XL adaptive cantor variant."""
    return load_vae_lyra(
        "AbstractPhil/vae-lyra-xl-adaptive-cantor",
        checkpoint=checkpoint,
        device=device
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("VAE Lyra Loader - Examples")
    print("=" * 80)

    # List known models
    list_known_models()

    # Example usage
    print("\n📝 Usage Examples:")
    print("-" * 80)
    print("""
    from geofractal.model.vae.loader import load_vae_lyra, load_lyra_illustrious

    # Auto-detect and load best checkpoint
    model = load_vae_lyra("AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious")

    # Use alias
    model = load_vae_lyra("lyra-illustrious")

    # Load specific checkpoint
    model = load_vae_lyra("lyra-illustrious", checkpoint="lyra_step_9000.safetensors")

    # Convenience function
    model = load_lyra_illustrious()

    # Get model info without loading weights
    from geofractal.model.vae.loader import print_model_info
    print_model_info("lyra-illustrious")

    # Load from local files
    from geofractal.model.vae.loader import load_vae_lyra_local
    model = load_vae_lyra_local("./weights/lyra_best.safetensors", config_path="./config.json")
    """)
    print("=" * 80)