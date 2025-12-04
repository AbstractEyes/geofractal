"""
geofractal.router.factory.registry
==================================
Registry for tracking and managing router prototypes.

Provides:
- Prototype registration and lookup
- Configuration persistence
- Hot-swapping of components
- Experiment tracking

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from datetime import datetime
import threading

from .protocols import StreamSpec, HeadSpec, FusionSpec
from .prototype import PrototypeConfig, AssembledPrototype


# =============================================================================
# PROTOTYPE RECORD
# =============================================================================

@dataclass
class PrototypeRecord:
    """Record of a registered prototype."""
    prototype_id: str
    name: str
    config: PrototypeConfig
    created_at: str
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prototype_id': self.prototype_id,
            'name': self.name,
            'config': self.config.to_dict(),
            'created_at': self.created_at,
            'tags': self.tags,
            'metrics': self.metrics,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PrototypeRecord':
        d = d.copy()
        d['config'] = PrototypeConfig.from_dict(d['config'])
        return cls(**d)


# =============================================================================
# PROTOTYPE REGISTRY
# =============================================================================

class PrototypeRegistry:
    """
    Registry for managing router prototypes.

    Features:
    - Register and track prototypes
    - Save/load configurations
    - Compare prototype configurations
    - Track metrics and experiments

    Usage:
        registry = PrototypeRegistry()

        # Register a prototype
        prototype_id = registry.register(prototype, tags=['experiment_1'])

        # Retrieve later
        record = registry.get(prototype_id)

        # List all prototypes
        for record in registry.list():
            print(record.name, record.metrics)
    """

    _instance: Optional['PrototypeRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._records: Dict[str, PrototypeRecord] = {}
        self._prototypes: Dict[str, nn.Module] = {}  # Weak refs to actual modules
        self._initialized = True

    def _generate_id(self, config: PrototypeConfig) -> str:
        """Generate unique ID from config."""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{config.prototype_name}_{timestamp}_{hash_obj.hexdigest()[:8]}"

    def register(
            self,
            prototype: AssembledPrototype,
            tags: Optional[List[str]] = None,
            notes: str = "",
    ) -> str:
        """
        Register a prototype.

        Args:
            prototype: The prototype to register
            tags: Optional tags for organization
            notes: Optional notes

        Returns:
            prototype_id: Unique identifier for retrieval
        """
        config = prototype.config
        prototype_id = self._generate_id(config)

        record = PrototypeRecord(
            prototype_id=prototype_id,
            name=config.prototype_name,
            config=config,
            created_at=datetime.now().isoformat(),
            tags=tags or [],
            notes=notes,
        )

        self._records[prototype_id] = record
        self._prototypes[prototype_id] = prototype

        return prototype_id

    def get(self, prototype_id: str) -> Optional[PrototypeRecord]:
        """Get record by ID."""
        return self._records.get(prototype_id)

    def get_prototype(self, prototype_id: str) -> Optional[nn.Module]:
        """Get actual prototype module by ID."""
        return self._prototypes.get(prototype_id)

    def list(
            self,
            tags: Optional[List[str]] = None,
            name_contains: Optional[str] = None,
    ) -> List[PrototypeRecord]:
        """
        List registered prototypes.

        Args:
            tags: Filter by tags (any match)
            name_contains: Filter by name substring

        Returns:
            Matching records
        """
        records = list(self._records.values())

        if tags:
            records = [r for r in records if any(t in r.tags for t in tags)]

        if name_contains:
            records = [r for r in records if name_contains in r.name]

        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def update_metrics(
            self,
            prototype_id: str,
            metrics: Dict[str, float],
    ):
        """Update metrics for a prototype."""
        if prototype_id in self._records:
            self._records[prototype_id].metrics.update(metrics)

    def add_tags(self, prototype_id: str, tags: List[str]):
        """Add tags to a prototype."""
        if prototype_id in self._records:
            self._records[prototype_id].tags.extend(tags)

    def remove(self, prototype_id: str):
        """Remove a prototype from registry."""
        self._records.pop(prototype_id, None)
        self._prototypes.pop(prototype_id, None)

    def clear(self):
        """Clear all records."""
        self._records.clear()
        self._prototypes.clear()

    def save(self, path: str):
        """Save registry to JSON file."""
        data = {
            pid: record.to_dict()
            for pid, record in self._records.items()
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str):
        """Load registry from JSON file."""
        data = json.loads(Path(path).read_text())
        for pid, record_dict in data.items():
            self._records[pid] = PrototypeRecord.from_dict(record_dict)

    def compare(
            self,
            id_a: str,
            id_b: str,
    ) -> Dict[str, Any]:
        """Compare two prototype configurations."""
        record_a = self.get(id_a)
        record_b = self.get(id_b)

        if not record_a or not record_b:
            return {'error': 'One or both prototypes not found'}

        config_a = record_a.config.to_dict()
        config_b = record_b.config.to_dict()

        differences = {}
        all_keys = set(config_a.keys()) | set(config_b.keys())

        for key in all_keys:
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            if val_a != val_b:
                differences[key] = {'a': val_a, 'b': val_b}

        return {
            'prototype_a': id_a,
            'prototype_b': id_b,
            'differences': differences,
            'metrics_comparison': {
                'a': record_a.metrics,
                'b': record_b.metrics,
            },
        }


# Global registry accessor
def get_prototype_registry() -> PrototypeRegistry:
    """Get the global prototype registry."""
    return PrototypeRegistry()


# =============================================================================
# COMPONENT SWAPPER
# =============================================================================

class ComponentSwapper:
    """
    Utility for hot-swapping components in prototypes.

    Enables runtime modification of:
    - Heads
    - Fusion layers
    - Streams
    - Classifiers

    Usage:
        swapper = ComponentSwapper(prototype)

        # Swap fusion strategy
        swapper.swap_fusion(new_fusion)

        # Swap a head
        swapper.swap_head("clip_b32", new_head)
    """

    def __init__(self, prototype: AssembledPrototype):
        self.prototype = prototype

    def swap_fusion(self, new_fusion: nn.Module):
        """Replace fusion layer."""
        self.prototype.fusion = new_fusion

    def swap_head(self, stream_name: str, new_head: nn.Module):
        """Replace head for a specific stream."""
        if stream_name not in self.prototype.heads:
            raise ValueError(f"Stream '{stream_name}' not found")
        self.prototype.heads[stream_name] = new_head

    def swap_stream(self, stream_name: str, new_stream: nn.Module):
        """Replace a stream."""
        if stream_name not in self.prototype.streams:
            raise ValueError(f"Stream '{stream_name}' not found")
        self.prototype.streams[stream_name] = new_stream

    def swap_classifier(self, new_classifier: nn.Module):
        """Replace classifier head."""
        self.prototype.classifier = new_classifier

    def swap_projection(self, stream_name: str, new_projection: nn.Module):
        """Replace projection layer for a stream."""
        if stream_name not in self.prototype.projections:
            raise ValueError(f"Stream '{stream_name}' not found")
        self.prototype.projections[stream_name] = new_projection

    def add_stream(
            self,
            name: str,
            stream: nn.Module,
            head: nn.Module,
            projection: Optional[nn.Module] = None,
            input_shape: str = "vector",  # NEW - just track this
    ):
        """Add a new stream to the prototype."""
        self.prototype.streams[name] = stream
        self.prototype.heads[name] = head

        if projection is None:
            projection = nn.Identity()
        self.prototype.projections[name] = projection

        self.prototype._stream_input_shapes[name] = input_shape  # NEW
        self.prototype.stream_names.append(name)


    def remove_stream(self, name: str):
        """Remove a stream from the prototype."""
        if name not in self.prototype.streams:
            raise ValueError(f"Stream '{name}' not found")

        del self.prototype.streams[name]
        del self.prototype.heads[name]
        del self.prototype.projections[name]
        self.prototype.stream_names.remove(name)


# =============================================================================
# EXPERIMENT TRACKER
# =============================================================================

@dataclass
class ExperimentRun:
    """Record of an experiment run."""
    run_id: str
    prototype_id: str
    started_at: str
    completed_at: Optional[str] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    status: str = "running"  # running, completed, failed


class ExperimentTracker:
    """
    Track experiments across prototypes.

    Usage:
        tracker = ExperimentTracker()

        run_id = tracker.start_run(prototype_id)

        # During training
        tracker.log_metric(run_id, "accuracy", 0.85)
        tracker.log_metric(run_id, "loss", 0.42)

        # When done
        tracker.complete_run(run_id)

        # View results
        results = tracker.get_run(run_id)
    """

    def __init__(self):
        self._runs: Dict[str, ExperimentRun] = {}
        self._run_counter = 0

    def start_run(
            self,
            prototype_id: str,
            config_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new experiment run."""
        self._run_counter += 1
        run_id = f"run_{self._run_counter:04d}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        run = ExperimentRun(
            run_id=run_id,
            prototype_id=prototype_id,
            started_at=datetime.now().isoformat(),
            config_overrides=config_overrides or {},
        )

        self._runs[run_id] = run
        return run_id

    def log_metric(self, run_id: str, name: str, value: float):
        """Log a metric for a run."""
        if run_id in self._runs:
            self._runs[run_id].metrics[name] = value

    def log_message(self, run_id: str, message: str):
        """Log a message for a run."""
        if run_id in self._runs:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._runs[run_id].logs.append(f"[{timestamp}] {message}")

    def complete_run(self, run_id: str, status: str = "completed"):
        """Mark a run as complete."""
        if run_id in self._runs:
            self._runs[run_id].completed_at = datetime.now().isoformat()
            self._runs[run_id].status = status

    def fail_run(self, run_id: str, error: str):
        """Mark a run as failed."""
        if run_id in self._runs:
            self._runs[run_id].completed_at = datetime.now().isoformat()
            self._runs[run_id].status = "failed"
            self._runs[run_id].logs.append(f"ERROR: {error}")

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a run by ID."""
        return self._runs.get(run_id)

    def list_runs(
            self,
            prototype_id: Optional[str] = None,
            status: Optional[str] = None,
    ) -> List[ExperimentRun]:
        """List experiment runs."""
        runs = list(self._runs.values())

        if prototype_id:
            runs = [r for r in runs if r.prototype_id == prototype_id]
        if status:
            runs = [r for r in runs if r.status == status]

        return sorted(runs, key=lambda r: r.started_at, reverse=True)

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across runs."""
        comparison = {'runs': {}}

        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                comparison['runs'][run_id] = {
                    'prototype_id': run.prototype_id,
                    'metrics': run.metrics,
                    'status': run.status,
                }

        # Find best for each metric
        all_metrics = set()
        for run_data in comparison['runs'].values():
            all_metrics.update(run_data['metrics'].keys())

        comparison['best'] = {}
        for metric in all_metrics:
            best_run = None
            best_value = None
            for run_id, run_data in comparison['runs'].items():
                value = run_data['metrics'].get(metric)
                if value is not None:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_run = run_id
            comparison['best'][metric] = {'run_id': best_run, 'value': best_value}

        return comparison


# Global tracker
_tracker: Optional[ExperimentTracker] = None


def get_experiment_tracker() -> ExperimentTracker:
    """Get the global experiment tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PrototypeRecord',
    'PrototypeRegistry',
    'get_prototype_registry',
    'ComponentSwapper',
    'ExperimentRun',
    'ExperimentTracker',
    'get_experiment_tracker',
]