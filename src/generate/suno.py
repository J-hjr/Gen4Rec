"""
Suno adapter backed by the ACE Data third-party API.

This adapter consumes the existing profile-prompt output and downloads the two
generated audio variants into `outputs/recSongs`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import re

import requests

from src.generate.api_client import AceSunoApiClient
from src.generate.base import GeneratedSample, GenerationResult, GenerationSpec


class SunoGenerator:
    """Generate and download Suno tracks via the ACE Data API."""

    provider_name = "suno"

    def __init__(self, model: str = "chirp-v4-5", client: AceSunoApiClient | None = None) -> None:
        self.model = model
        self.client = client or AceSunoApiClient()

    def _build_style(self, spec: GenerationSpec) -> str:
        return ", ".join(spec.style_keywords).strip()

    def _download_audio(self, url: str, target_path: Path) -> None:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        target_path.write_bytes(response.content)

    def _slugify_title(self, title: str | None, fallback: str = "generated_song") -> str:
        raw = (title or "").strip() or fallback
        slug = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
        return slug or fallback

    def generate(self, spec: GenerationSpec, output_dir: Path) -> GenerationResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = spec.generation_prompt.strip()
        lyric_prompt = spec.profile_paragraph.strip()
        lyric = spec.lyrics.strip()
        style = self._build_style(spec)

        response_json = self.client.generate_music(
            prompt=prompt_text,
            model=self.model,
            lyric_prompt=lyric_prompt,
            lyric=lyric,
            custom=bool(lyric),
            instrumental=not bool(lyric),
            title=f"{spec.user_id}_rec_song",
            style=style,
            style_negative=spec.negative_prompt or "",
        )

        response_data = response_json.get("data", [])
        response_metadata: dict[str, Any] = {
            "success": response_json.get("success"),
            "task_id": response_json.get("task_id"),
            "trace_id": response_json.get("trace_id"),
            "started_at": response_json.get("started_at"),
            "finished_at": response_json.get("finished_at"),
            "elapsed": response_json.get("elapsed"),
            "variant_count": len(response_data),
        }

        samples: list[GeneratedSample] = []
        for idx, item in enumerate(response_data, start=1):
            audio_url = str(item.get("audio_url", "")).strip()
            if not audio_url:
                continue
            title = str(item.get("title", "")).strip() or None
            base_name = f"{self._slugify_title(title)}_variant_{idx:02d}"
            sample_path = output_dir / f"{base_name}.mp3"
            lyric_path = output_dir / f"{base_name}_lyrics.txt"
            metadata_path = output_dir / f"{base_name}.json"

            self._download_audio(audio_url, sample_path)
            lyric_path.write_text(str(item.get("lyric", "")), encoding="utf-8")
            metadata_path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")

            samples.append(
                GeneratedSample(
                    path=str(sample_path),
                    mime_type="audio/mpeg",
                    text_companion=str(lyric_path),
                    source_url=audio_url,
                    metadata_path=str(metadata_path),
                    variant_index=idx,
                    title=title,
                )
            )

        return GenerationResult(
            provider=self.provider_name,
            model=self.model,
            prompt_used=prompt_text,
            negative_prompt_used=spec.negative_prompt,
            request_payload={
                "action": "generate",
                "model": self.model,
                "user_id": spec.user_id,
                "style": style,
                "custom": bool(lyric),
                "instrumental": not bool(lyric),
            },
            response_metadata=response_metadata,
            samples=samples,
        )
