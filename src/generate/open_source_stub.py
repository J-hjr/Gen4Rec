"""
Placeholder adapter for future open-source music-generation backends.

The current MVP is API-first. This file reserves a clean integration point so
later local models can implement the same interface without changing the
upstream retrieval/profile pipeline.
"""

from __future__ import annotations

from pathlib import Path

from src.generate.base import GenerationResult, GenerationSpec


class OpenSourceGeneratorStub:
    """Reserved hook for future local-model generation backends."""

    provider_name = "open_source_stub"

    def generate(self, spec: GenerationSpec, output_dir: Path) -> GenerationResult:
        raise NotImplementedError(
            "Open-source generation is not implemented yet. Use the Suno API path for now."
        )
