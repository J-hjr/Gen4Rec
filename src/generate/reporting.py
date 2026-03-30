"""
Helpers for saving generation manifests and writing lightweight demo reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.generate.base import GenerationResult, GenerationSpec


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_user_facing_profile(prompt_output: dict[str, Any]) -> dict[str, Any]:
    summary = dict(prompt_output.get("input_summary", {}))
    return {
        "user_id": prompt_output.get("user_id"),
        "profile_paragraph": prompt_output.get("profile_paragraph"),
        "style_keywords": prompt_output.get("style_keywords", []),
        "top_genres": summary.get("top_genres", []),
        "top_tags": summary.get("top_tags", []),
        "representative_artists": summary.get("representative_artists", []),
        "representative_tracks": summary.get("representative_tracks", []),
        "mood_summary": summary.get("mood_summary", []),
        "audio_profile": summary.get("audio_profile", {}),
    }


def write_markdown_report(
    *,
    path: Path,
    run_id: str,
    user_id: str,
    provider: str,
    prompt_output: dict[str, Any],
    spec: GenerationSpec,
    result: GenerationResult,
) -> None:
    profile = build_user_facing_profile(prompt_output)
    sample_lines = [f"- `{sample.path}` ({sample.mime_type})" for sample in result.samples] or ["- No audio samples saved"]
    report = f"""# Music Generation Run

## Summary

- Run ID: `{run_id}`
- User ID: `{user_id}`
- Provider: `{provider}`
- Model: `{result.model}`

## Methodology

- This run reuses an existing prompt JSON from the current `profile_prompt` pipeline.
- The generation backend consumes a normalized generation spec and sends it to the hosted Suno-compatible provider.

## Results

### Listener profile

{profile.get("profile_paragraph", "")}

### Style keywords

{", ".join(profile.get("style_keywords", []))}

### Generated audio artifacts

{chr(10).join(sample_lines)}

## Analysis

- Prompt used: `{result.prompt_used}`
- Negative prompt: `{result.negative_prompt_used or "None"}`
- Retrieved top genres: `{", ".join(profile.get("top_genres", []))}`
- Retrieved top tags: `{", ".join(profile.get("top_tags", []))}`

## Plan for Additional Analysis

- Compare this API-first output with a future open-source generation backend.
- Tailor the current profile-prompt output more directly for Suno-style generation.
- Add CLAP-based alignment checks between generated audio and the target user embedding.

## Work Plan

- Finalize the Suno API demo path.
- Improve prompt tailoring for Suno generation.
- Add a parallel open-source adapter later through the shared generator interface.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
