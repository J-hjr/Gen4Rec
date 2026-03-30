"""
Lyria adapter for the generation add-on pipeline.

This adapter consumes a provider-agnostic `GenerationSpec`, calls the Gemini
API, saves audio artifacts locally, and returns a normalized result manifest.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from src.generate.api_client import GeminiApiClient
from src.generate.base import GeneratedSample, GenerationResult, GenerationSpec


def _extension_for_mime_type(mime_type: str) -> str:
    if mime_type == "audio/wav":
        return ".wav"
    if mime_type == "audio/mpeg":
        return ".mp3"
    if mime_type == "audio/mp3":
        return ".mp3"
    return ".bin"


class LyriaGenerator:
    """Generate music clips with Gemini API Lyria models."""

    provider_name = "lyria"

    def __init__(self, model: str = "lyria-3-clip-preview", client: GeminiApiClient | None = None) -> None:
        self.model = model
        self.client = client or GeminiApiClient()

    def _build_prompt(self, spec: GenerationSpec) -> str:
        prompt_parts = [spec.generation_prompt.strip()]
        if spec.tempo_hint_bpm is not None:
            prompt_parts.append(f"Tempo target around {spec.tempo_hint_bpm} BPM.")
        if spec.instrumentation:
            prompt_parts.append(f"Instrumentation: {', '.join(spec.instrumentation)}.")
        if spec.lyrics.strip():
            prompt_parts.append(f"Lyrics: {spec.lyrics.strip()}")
        if spec.sections:
            section_lines = []
            for section in spec.sections:
                timestamp = section.get("timestamp", "")
                label = section.get("label", "")
                text = section.get("text", "")
                parts = [str(x).strip() for x in (timestamp, label, text) if str(x).strip()]
                if parts:
                    section_lines.append(" | ".join(parts))
            if section_lines:
                prompt_parts.append("Structure cues: " + " ; ".join(section_lines))
        if spec.negative_prompt:
            prompt_parts.append(f"Avoid: {spec.negative_prompt.strip()}.")
        return " ".join(part for part in prompt_parts if part).strip()

    def generate(self, spec: GenerationSpec, output_dir: Path) -> GenerationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_text = self._build_prompt(spec)
        response_json = self.client.generate_music(model=self.model, prompt=prompt_text)

        samples: list[GeneratedSample] = []
        response_metadata: dict[str, Any] = {
            "prompt_feedback": response_json.get("promptFeedback"),
            "candidate_count": len(response_json.get("candidates", [])),
        }

        sample_idx = 0
        for candidate in response_json.get("candidates", []):
            content = candidate.get("content", {})
            text_parts: list[str] = []
            audio_parts: list[dict[str, Any]] = []
            for part in content.get("parts", []):
                if "text" in part and part["text"] is not None:
                    text_parts.append(str(part["text"]))
                inline_data = part.get("inlineData")
                if inline_data:
                    audio_parts.append(inline_data)

            companion_text = "\n".join(text_parts).strip() or None
            for inline_data in audio_parts:
                sample_idx += 1
                mime_type = str(inline_data.get("mimeType", "application/octet-stream"))
                file_ext = _extension_for_mime_type(mime_type)
                sample_path = output_dir / f"sample_{sample_idx:02d}{file_ext}"
                sample_bytes = base64.b64decode(inline_data["data"])
                sample_path.write_bytes(sample_bytes)

                text_path = None
                if companion_text:
                    text_path = output_dir / f"sample_{sample_idx:02d}.txt"
                    text_path.write_text(companion_text, encoding="utf-8")

                samples.append(
                    GeneratedSample(
                        path=str(sample_path),
                        mime_type=mime_type,
                        text_companion=str(text_path) if text_path else None,
                    )
                )

        return GenerationResult(
            provider=self.provider_name,
            model=self.model,
            prompt_used=prompt_text,
            negative_prompt_used=spec.negative_prompt,
            request_payload={
                "model": self.model,
                "response_modalities": ["AUDIO", "TEXT"],
                "user_id": spec.user_id,
            },
            response_metadata=response_metadata,
            samples=samples,
        )
