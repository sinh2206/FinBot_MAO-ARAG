from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentRuntime:
    name: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(self, name: str, enabled: bool = True, **metadata: Any) -> None:
        self.runtime = AgentRuntime(name=name, enabled=enabled, metadata=metadata)

    @property
    def name(self) -> str:
        return self.runtime.name

    @property
    def enabled(self) -> bool:
        return self.runtime.enabled

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
