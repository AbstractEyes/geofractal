"""
geofractal.router.registry
==========================
Global registry and mailbox for router coordination.

The registry tracks all routers in a collective and their relationships.
The mailbox enables inter-router communication during forward passes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import uuid
from threading import Lock


@dataclass
class RouterInfo:
    """Information about a registered router."""
    module_id: str
    name: str
    parent_id: Optional[str]
    cooperation_group: str
    fingerprint_dim: int
    feature_dim: int

    # Runtime state
    children: Set[str] = field(default_factory=set)


class RouterRegistry:
    """
    Global registry for router coordination.

    Tracks:
    - All routers in the system
    - Parent-child relationships
    - Cooperation groups
    - Fingerprint associations

    Usage:
        registry = get_registry()
        registry.register(router_info)
        children = registry.get_children(module_id)
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.routers: Dict[str, RouterInfo] = {}
        self.groups: Dict[str, Set[str]] = {}  # group_name -> set of module_ids
        self.name_to_id: Dict[str, str] = {}  # name -> module_id
        self._initialized = True

    def reset(self):
        """Clear all registrations. Call before building new collective."""
        self.routers.clear()
        self.groups.clear()
        self.name_to_id.clear()

    def register(
            self,
            name: str,
            parent_id: Optional[str],
            cooperation_group: str,
            fingerprint_dim: int,
            feature_dim: int,
    ) -> str:
        """
        Register a new router.

        Returns:
            module_id: Unique identifier for this router
        """
        module_id = str(uuid.uuid4())

        info = RouterInfo(
            module_id=module_id,
            name=name,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
            fingerprint_dim=fingerprint_dim,
            feature_dim=feature_dim,
        )

        self.routers[module_id] = info
        self.name_to_id[name] = module_id

        # Add to cooperation group
        if cooperation_group not in self.groups:
            self.groups[cooperation_group] = set()
        self.groups[cooperation_group].add(module_id)

        # Register as child of parent
        if parent_id and parent_id in self.routers:
            self.routers[parent_id].children.add(module_id)

        return module_id

    def get(self, module_id: str) -> Optional[RouterInfo]:
        """Get router info by ID."""
        return self.routers.get(module_id)

    def get_by_name(self, name: str) -> Optional[RouterInfo]:
        """Get router info by name."""
        module_id = self.name_to_id.get(name)
        return self.routers.get(module_id) if module_id else None

    def get_children(self, module_id: str) -> List[str]:
        """Get all child router IDs."""
        info = self.routers.get(module_id)
        return list(info.children) if info else []

    def get_siblings(self, module_id: str) -> List[str]:
        """Get routers in same cooperation group."""
        info = self.routers.get(module_id)
        if not info:
            return []

        group = self.groups.get(info.cooperation_group, set())
        return [mid for mid in group if mid != module_id]

    def get_group(self, group_name: str) -> List[str]:
        """Get all router IDs in a cooperation group."""
        return list(self.groups.get(group_name, set()))

    def get_hierarchy(self, module_id: str) -> Dict[str, Any]:
        """Get full hierarchy starting from a router."""
        info = self.routers.get(module_id)
        if not info:
            return {}

        return {
            'id': module_id,
            'name': info.name,
            'children': [
                self.get_hierarchy(child_id)
                for child_id in info.children
            ]
        }


def get_registry() -> RouterRegistry:
    """Get the global router registry singleton."""
    return RouterRegistry()


@dataclass
class RouterMessage:
    """Message passed between routers via mailbox."""
    sender_id: str
    sender_name: str
    content: torch.Tensor  # Routing state or attention pattern
    timestamp: int  # Step counter for ordering


class RouterMailbox:
    """
    Shared mailbox for inter-router communication.

    Routers post their routing states after each forward pass.
    Other routers can read these states to inform their decisions.

    This enables emergent coordination without explicit supervision.

    Usage:
        mailbox = RouterMailbox(config)

        # In router forward:
        mailbox.post(self.module_id, self.name, routing_state)
        peer_states = mailbox.read_all(exclude=self.module_id)
    """

    def __init__(self, config=None):
        self.messages: Dict[str, RouterMessage] = {}
        self.step_counter = 0
        self.config = config

    def clear(self):
        """Clear all messages. Call at start of collective forward."""
        self.messages.clear()
        self.step_counter = 0

    def post(
            self,
            sender_id: str,
            sender_name: str,
            content: torch.Tensor,
    ):
        """Post a message to the mailbox."""
        self.messages[sender_id] = RouterMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            content=content.detach(),  # Don't backprop through mailbox
            timestamp=self.step_counter,
        )
        self.step_counter += 1

    def read(self, sender_id: str) -> Optional[torch.Tensor]:
        """Read message from a specific sender."""
        msg = self.messages.get(sender_id)
        return msg.content if msg else None

    def read_all(self, exclude: Optional[str] = None) -> List[torch.Tensor]:
        """Read all messages, optionally excluding sender."""
        return [
            msg.content
            for sender_id, msg in self.messages.items()
            if sender_id != exclude
        ]

    def read_latest(self, n: int = 1, exclude: Optional[str] = None) -> List[torch.Tensor]:
        """Read n most recent messages."""
        sorted_msgs = sorted(
            self.messages.values(),
            key=lambda m: m.timestamp,
            reverse=True
        )

        result = []
        for msg in sorted_msgs:
            if msg.sender_id != exclude:
                result.append(msg.content)
                if len(result) >= n:
                    break

        return result

    def __len__(self):
        return len(self.messages)