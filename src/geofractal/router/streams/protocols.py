# streams/protocols.py
"""Stream interface definitions."""

from typing import Protocol, Tuple, Dict, Any, Optional, runtime_checkable
import torch


class InputShape:
    """Stream input shape types."""
    VECTOR = "vector"      # [B, D]
    SEQUENCE = "sequence"  # [B, S, D]
    IMAGE = "image"        # [B, C, H, W]


@runtime_checkable
class StreamProtocol(Protocol):
    """What all streams must implement."""
    
    name: str
    
    @property
    def fingerprint(self) -> torch.Tensor:
        """Access head's fingerprint."""
        ...
    
    def forward(
        self, 
        x: Any,
        target_fingerprint: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        ...
    
    def pool(self, x: torch.Tensor) -> torch.Tensor:
        ...