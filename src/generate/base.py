"""
Shared data contracts for music-generation backends.

The upstream pipeline should produce one provider-agnostic generation spec.
Backends then translate that spec into provider-specific requests and return a
normalized result manifest for downstream reporting and analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class GenerationSpec:
    """Provider-agnostic music-generation request."""

    schema_version: str
    user_id: str
    provider_target: str
    prompt_version: str
    generation_prompt: str
    negative_prompt: str | None = None
    style_keywords: list[str] = field(default_factory=list)
    instrumentation: list[str] = field(default_factory=list)
    lyrics: str = ""
    sections: list[dict[str, Any]] = field(default_factory=list)
    tempo_hint_bpm: int | None = None
    duration_hint_seconds: int | None = None
    profile_paragraph: str = ""
    input_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationSpec":
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            user_id=str(data["user_id"]),
            provider_target=str(data.get("provider_target", "generic")),
            prompt_version=str(data.get("prompt_version", "v1")),
            generation_prompt=str(data["generation_prompt"]),
            negative_prompt=data.get("negative_prompt"),
            style_keywords=[str(x) for x in data.get("style_keywords", [])],
            instrumentation=[str(x) for x in data.get("instrumentation", [])],
            lyrics=str(data.get("lyrics", "")),
            sections=list(data.get("sections", [])),
            tempo_hint_bpm=data.get("tempo_hint_bpm"),
            duration_hint_seconds=data.get("duration_hint_seconds"),
            profile_paragraph=str(data.get("profile_paragraph", "")),
            input_summary=dict(data.get("input_summary", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedSample:
    """One saved media artifact returned by a generator."""

    path: str
    mime_type: str
    text_companion: str | None = None
    source_url: str | None = None
    metadata_path: str | None = None
    call_index: int | None = None
    variant_index: int | None = None
    title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Normalized result manifest for any music-generation backend."""

    provider: str
    model: str
    prompt_used: str
    negative_prompt_used: str | None
    request_payload: dict[str, Any]
    response_metadata: dict[str, Any]
    samples: list[GeneratedSample]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["samples"] = [sample.to_dict() for sample in self.samples]
        return data


class MusicGenerator(Protocol):
    """Minimal interface for API and open-source generation backends."""

    provider_name: str

    def generate(self, spec: GenerationSpec, output_dir: Path) -> GenerationResult:
        """Generate audio assets and return a normalized manifest."""
