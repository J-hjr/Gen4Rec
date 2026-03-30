"""
Canonical artifact locations for the generation add-on pipeline.

The goal is to keep generation outputs in one place for the current MVP:
- generation input snapshot
- generation spec
- generated audio variants
- run manifest and report
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


BASE_DIR_PATH = Path(__file__).resolve().parent
REPO_ROOT_PATH = BASE_DIR_PATH.parent.parent


def sanitize_segment(value: str) -> str:
    """Make a path-safe identifier while preserving readability."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"


def build_run_id(user_id: str, provider: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{sanitize_segment(user_id)}-{sanitize_segment(provider)}"


@dataclass
class ArtifactPaths:
    prompt_input_json: Path
    generation_spec_json: Path
    audio_dir: Path
    manifest_json: Path
    report_md: Path

    def ensure_directories(self) -> None:
        for path in (
            self.prompt_input_json,
            self.generation_spec_json,
            self.manifest_json,
            self.report_md,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def build_artifact_paths(
    user_id: str,
    provider: str,
    run_id: str | None = None,
    outputs_root: Path | None = None,
) -> tuple[str, ArtifactPaths]:
    outputs_root = outputs_root or (REPO_ROOT_PATH / "outputs" / "recSongs")
    run_id = run_id or build_run_id(user_id=user_id, provider=provider)
    run_root = outputs_root / sanitize_segment(user_id) / run_id

    paths = ArtifactPaths(
        prompt_input_json=run_root / "prompt_input.json",
        generation_spec_json=run_root / "generation_spec.json",
        audio_dir=run_root / "audio",
        manifest_json=run_root / "run_manifest.json",
        report_md=run_root / "report.md",
    )
    return run_id, paths
