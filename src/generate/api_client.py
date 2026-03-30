"""
Minimal HTTP client for hosted music-generation APIs.

This module currently targets the ACE Data Suno-compatible API. It keeps HTTP
details out of the generator adapter so future providers can follow the same
pattern.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


class AceSunoApiClient:
    """Thin wrapper around the ACE Data Suno audio-generation endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: int = 300,
        base_url: str = "https://api.acedata.cloud/suno/audios",
    ) -> None:
        self.api_key = api_key or os.getenv("ACE_SUNO_API_KEY")
        if not self.api_key:
            raise ValueError("ACE_SUNO_API_KEY not found. Please export your ACE Suno API key first.")
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url

    def generate_music(
        self,
        *,
        prompt: str,
        model: str,
        lyric_prompt: str = "",
        lyric: str = "",
        custom: bool | None = None,
        instrumental: bool | None = None,
        title: str = "",
        style: str = "",
        style_negative: str = "",
    ) -> dict[str, Any]:
        """Call ACE Data's Suno-compatible generation endpoint."""
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        payload = {
            "action": "generate",
            "prompt": prompt,
            "model": model,
        }
        if lyric_prompt:
            payload["lyric_prompt"] = lyric_prompt
        if lyric:
            payload["lyric"] = lyric
        if custom is not None:
            payload["custom"] = custom
        if instrumental is not None:
            payload["instrumental"] = instrumental
        if title:
            payload["title"] = title
        if style:
            payload["style"] = style
        if style_negative:
            payload["style_negative"] = style_negative
        response = requests.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
