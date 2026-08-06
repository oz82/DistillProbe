from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProbeResult:
    module: str
    metrics: dict[str, Any] = field(default_factory=dict)
    per_category: list[dict] = field(default_factory=list)
    baselines: dict[str, Any] = field(default_factory=dict)
    tokenization_audit: list[dict] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ProbeModule(ABC):
    """Base class for probing modules."""

    name: str = "<unset>"

    @abstractmethod
    def describe(self) -> str: ...

    @abstractmethod
    def run(self, encoders: dict, data_path: str, config) -> ProbeResult: ...

    def plot(self, result_dict: dict, out_dir) -> list:
        """Optional: produce plots for this module's result. Default: no-op."""
        return []
