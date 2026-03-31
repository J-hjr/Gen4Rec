from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.generate.artifacts import sanitize_segment


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"
PROFILES_ROOT = OUTPUTS_ROOT / "profiles"
RECSONGS_ROOT = OUTPUTS_ROOT / "recSongs"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@dataclass
class ProfileArtifacts:
    user_id: str
    raw_profile_path: Path
    summary_path: Path
    prompt_path: Path
    validation_path: Path
    raw_profile: dict[str, Any] | None
    summary: dict[str, Any] | None
    prompt: dict[str, Any] | None
    validation: dict[str, Any] | None


@dataclass
class TrackArtifact:
    title: str
    path: Path
    metadata_path: Path | None
    lyric_path: Path | None
    rerank_score: float | None
    is_selected: bool
    call_index: int | None
    variant_index: int | None
    source_url: str | None
    cover_url: str | None
    cover_large_url: str | None
    duration_seconds: float | None
    prompt: str | None
    style: str | None
    lyric_text: str | None


@dataclass
class GenerationRunArtifacts:
    user_id: str
    run_id: str
    run_root: Path
    manifest_path: Path
    rerank_path: Path
    report_path: Path
    prompt_input_path: Path | None
    generation_spec_path: Path | None
    manifest: dict[str, Any]
    rerank: dict[str, Any] | None
    prompt_input: dict[str, Any] | None
    generation_spec: dict[str, Any] | None
    report_markdown: str | None
    tracks: list[TrackArtifact]


def get_profile_paths(user_id: str) -> dict[str, Path]:
    safe_user_id = sanitize_segment(user_id)
    return {
        "raw_profile": PROFILES_ROOT / f"{safe_user_id}.json",
        "summary": PROFILES_ROOT / f"{safe_user_id}_summary.json",
        "prompt": PROFILES_ROOT / f"{safe_user_id}_prompt.json",
        "validation": PROFILES_ROOT / f"{safe_user_id}_validation.json",
    }


def load_profile_artifacts(user_id: str) -> ProfileArtifacts:
    paths = get_profile_paths(user_id)
    return ProfileArtifacts(
        user_id=user_id,
        raw_profile_path=paths["raw_profile"],
        summary_path=paths["summary"],
        prompt_path=paths["prompt"],
        validation_path=paths["validation"],
        raw_profile=_load_json_if_exists(paths["raw_profile"]),
        summary=_load_json_if_exists(paths["summary"]),
        prompt=_load_json_if_exists(paths["prompt"]),
        validation=_load_json_if_exists(paths["validation"]),
    )


def list_generation_run_dirs(user_id: str) -> list[Path]:
    user_root = RECSONGS_ROOT / sanitize_segment(user_id)
    if not user_root.exists():
        return []
    return sorted([path for path in user_root.iterdir() if path.is_dir()], reverse=True)


def _build_tracks_from_run(manifest: dict[str, Any], rerank: dict[str, Any] | None) -> list[TrackArtifact]:
    sample_meta_by_path = {
        str(sample.get("path")): sample
        for sample in manifest.get("result", {}).get("samples", [])
    }
    selected_paths = {
        str(item.get("path"))
        for item in (rerank or {}).get("final_selected_tracks", [])
    }

    if rerank and rerank.get("candidates"):
        ordered_items = rerank["candidates"]
    else:
        ordered_items = [sample_meta_by_path[path] for path in sample_meta_by_path]

    tracks: list[TrackArtifact] = []
    for item in ordered_items:
        audio_path = Path(str(item.get("path", "")))
        metadata_path = Path(item["metadata_path"]) if item.get("metadata_path") else None
        lyric_path = Path(item["lyric_path"]) if item.get("lyric_path") else None
        metadata_payload = _load_json_if_exists(metadata_path) if metadata_path else None
        lyric_text = _load_text_if_exists(lyric_path) if lyric_path else None

        title = str(
            item.get("title")
            or (metadata_payload or {}).get("title")
            or audio_path.stem
        )

        tracks.append(
            TrackArtifact(
                title=title,
                path=audio_path,
                metadata_path=metadata_path,
                lyric_path=lyric_path,
                rerank_score=item.get("clap_cosine_score"),
                is_selected=str(audio_path) in selected_paths,
                call_index=item.get("call_index"),
                variant_index=item.get("variant_index"),
                source_url=item.get("source_url"),
                cover_url=(metadata_payload or {}).get("image_url"),
                cover_large_url=(metadata_payload or {}).get("image_large_url"),
                duration_seconds=(metadata_payload or {}).get("duration"),
                prompt=(metadata_payload or {}).get("prompt"),
                style=(metadata_payload or {}).get("style"),
                lyric_text=lyric_text,
            )
        )
    return tracks


def load_generation_run(run_root: str | Path) -> GenerationRunArtifacts:
    run_root = Path(run_root)
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rerank_path = run_root / "rerank_results.json"
    report_path = run_root / "report.md"

    rerank = _load_json_if_exists(rerank_path)
    report_markdown = _load_text_if_exists(report_path)

    artifacts = manifest.get("artifacts", {})
    prompt_input_path = Path(artifacts["prompt_input_json"]) if artifacts.get("prompt_input_json") else None
    generation_spec_path = Path(artifacts["generation_spec_json"]) if artifacts.get("generation_spec_json") else None

    return GenerationRunArtifacts(
        user_id=str(manifest["user_id"]),
        run_id=str(manifest["run_id"]),
        run_root=run_root,
        manifest_path=manifest_path,
        rerank_path=rerank_path,
        report_path=report_path,
        prompt_input_path=prompt_input_path,
        generation_spec_path=generation_spec_path,
        manifest=manifest,
        rerank=rerank,
        prompt_input=_load_json_if_exists(prompt_input_path) if prompt_input_path else None,
        generation_spec=_load_json_if_exists(generation_spec_path) if generation_spec_path else None,
        report_markdown=report_markdown,
        tracks=_build_tracks_from_run(manifest, rerank),
    )


def load_latest_generation_run(user_id: str) -> GenerationRunArtifacts | None:
    run_dirs = list_generation_run_dirs(user_id)
    if not run_dirs:
        return None
    return load_generation_run(run_dirs[0])


def read_binary_file(path: str | Path) -> bytes | None:
    path = Path(path)
    if not path.exists():
        return None
    return path.read_bytes()
