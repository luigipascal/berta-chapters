"""
File-backed model registry for Chapter 15: MLOps & Model Deployment.

This is a teaching-grade registry: a single JSON index file plus on-disk
artifacts. It mirrors the API surface of MLflow's model registry so the
patterns transfer: register, transition_stage, get_production, list_models.

Stages: None | Staging | Production | Archived
At most one model version per name may be in 'Production' at a time.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_STAGES = ("None", "Staging", "Production", "Archived")


@dataclass
class RegistryEntry:
    """One immutable artifact + its lifecycle metadata."""
    model_name: str
    version: str
    stage: str
    artifact_path: str
    framework: str = "sklearn"
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistryEntry":
        return cls(**data)


class ModelRegistry:
    """
    Append-only registry: each register() call adds a new immutable entry.
    Stage transitions update an entry's `stage` field; promoting to Production
    auto-archives the previous Production entry for the same model name.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write_index([])

    # ---------------- index I/O ----------------

    def _read_index(self) -> List[RegistryEntry]:
        with self.index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [RegistryEntry.from_dict(d) for d in data]

    def _write_index(self, entries: List[RegistryEntry]) -> None:
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in entries], f, indent=2)

    # ---------------- public API ----------------

    def register(
        self,
        model_name: str,
        version: str,
        artifact_src: str | Path,
        framework: str = "sklearn",
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> RegistryEntry:
        """
        Copy an artifact into the registry and record it. Returns the new entry.

        The artifact is stored at `<root>/<model_name>/<version>/<basename>`.
        Re-registering the same (name, version) pair raises ValueError.
        """
        entries = self._read_index()
        for e in entries:
            if e.model_name == model_name and e.version == version:
                raise ValueError(
                    f"version {version!r} already registered for {model_name!r}"
                )
        src = Path(artifact_src)
        if not src.exists():
            raise FileNotFoundError(f"artifact not found: {src}")
        dst_dir = self.root / model_name / version
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        entry = RegistryEntry(
            model_name=model_name,
            version=version,
            stage="None",
            artifact_path=str(dst.relative_to(self.root)),
            framework=framework,
            metrics=dict(metrics or {}),
            tags=dict(tags or {}),
        )
        entries.append(entry)
        self._write_index(entries)
        return entry

    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> RegistryEntry:
        """
        Move an entry to a new stage. Promoting to Production auto-archives
        any prior Production entry for the same model_name.
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}; got {stage!r}")
        entries = self._read_index()
        target: Optional[RegistryEntry] = None
        for e in entries:
            if e.model_name == model_name and e.version == version:
                target = e
                break
        if target is None:
            raise KeyError(f"({model_name}, {version}) not found in registry")
        if stage == "Production":
            for e in entries:
                if (
                    e.model_name == model_name
                    and e.stage == "Production"
                    and e is not target
                ):
                    e.stage = "Archived"
                    e.updated_at = time.time()
        target.stage = stage
        target.updated_at = time.time()
        self._write_index(entries)
        return target

    def get_production(self, model_name: str) -> Optional[RegistryEntry]:
        """Return the current Production entry for the given model, or None."""
        for e in self._read_index():
            if e.model_name == model_name and e.stage == "Production":
                return e
        return None

    def get(self, model_name: str, version: str) -> Optional[RegistryEntry]:
        for e in self._read_index():
            if e.model_name == model_name and e.version == version:
                return e
        return None

    def list_models(self, model_name: Optional[str] = None) -> List[RegistryEntry]:
        """List all entries, optionally filtered by model_name."""
        entries = self._read_index()
        if model_name is not None:
            entries = [e for e in entries if e.model_name == model_name]
        return entries

    def absolute_artifact_path(self, entry: RegistryEntry) -> Path:
        """Resolve a registry-relative artifact_path to an absolute Path."""
        return self.root / entry.artifact_path


__all__ = ["ModelRegistry", "RegistryEntry", "VALID_STAGES"]
