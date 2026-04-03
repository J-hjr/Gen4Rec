from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.services.artifact_service import (
    OUTPUTS_ROOT,
    ProfileArtifacts,
    load_generation_run,
    load_profile_artifacts,
)
from src.embed.export_user_profile_json import export_user_profile_payload
from src.eval.run_eval import evaluate_generation_run
from src.generate.rerank import run_rerank_from_manifest
from src.generate.run_generate import run_generation_pipeline
from src.profile_prompt.build_profile_features import build_profile_features, save_summary
from src.profile_prompt.generate_user_profile_and_prompt import generate_music_prompt, save_output


EMBEDDINGS_ROOT = OUTPUTS_ROOT / "embeddings" / "music4all"
USER_IDS_PATH = EMBEDDINGS_ROOT / "user_ids.npy"


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_available_users() -> list[str]:
    if not USER_IDS_PATH.exists():
        raise FileNotFoundError(
            f"User ID array not found at {USER_IDS_PATH}. "
            "Please build or copy user embeddings before using the demo."
        )
    user_ids = np.load(USER_IDS_PATH, allow_pickle=True).astype(str)
    return sorted(user_ids.tolist())


def build_or_load_profile(
    *,
    user_id: str,
    top_k: int = 20,
    top_n: int = 20,
    exclude_recent: bool = True,
    openai_model: str = "gpt-4.1-mini",
    force_rebuild: bool = False,
) -> ProfileArtifacts:
    artifacts = load_profile_artifacts(user_id)

    raw_payload = artifacts.raw_profile
    if force_rebuild or raw_payload is None:
        raw_payload = export_user_profile_payload(
            user_id=user_id,
            top_k=top_k,
            exclude_recent=exclude_recent,
        )
        _write_json(raw_payload, artifacts.raw_profile_path)

    summary = artifacts.summary
    if force_rebuild or summary is None:
        summary = build_profile_features(raw_payload, top_n=top_n)
        save_summary(summary, artifacts.summary_path)

    prompt = artifacts.prompt
    if force_rebuild or prompt is None:
        prompt = generate_music_prompt(summary, model=openai_model)
        save_output(prompt, artifacts.prompt_path)

    return load_profile_artifacts(user_id)


def run_generation_for_user(
    *,
    user_id: str,
    prompt_output: dict[str, Any],
    generation_model: str = "chirp-v4-5",
    num_calls: int = 5,
    max_concurrency: int = 2,
    negative_prompt: str | None = None,
    lyrics: str = "",
    tempo_hint_bpm: int | None = None,
    duration_hint_seconds: int | None = None,
    prompt_version: str = "existing-profile-prompt-v1",
    rerank_top_k: int = 2,
    rerank_diversity_threshold: float | None = None,
    rerank_encoder: str = "auto",
    eval_recent_k: int = 20,
    eval_reference_top_k: int = 3,
    eval_encoder: str = "finetuned",
    eval_save_plot: bool = True,
    eval_imitation_threshold: float = 0.9,
) -> dict[str, Any]:
    run_id, manifest, manifest_path = run_generation_pipeline(
        prompt_output=prompt_output,
        provider="suno",
        generation_model=generation_model,
        user_id=user_id,
        num_calls=num_calls,
        max_concurrency=max_concurrency,
        negative_prompt=negative_prompt,
        lyrics=lyrics,
        tempo_hint_bpm=tempo_hint_bpm,
        duration_hint_seconds=duration_hint_seconds,
        prompt_version=prompt_version,
    )
    rerank_result, rerank_output_path = run_rerank_from_manifest(
        manifest_path=manifest_path,
        top_k=rerank_top_k,
        diversity_threshold=rerank_diversity_threshold,
        encoder=rerank_encoder,
    )
    eval_result = evaluate_generation_run(
        manifest_path=manifest_path,
        recent_k=eval_recent_k,
        reference_top_k=eval_reference_top_k,
        encoder=eval_encoder,
        rerank_top_k=rerank_top_k,
        diversity_threshold=rerank_diversity_threshold,
        save_plot=eval_save_plot,
        imitation_threshold=eval_imitation_threshold,
    )
    run_artifacts = load_generation_run(Path(manifest_path).parent)
    return {
        "run_id": run_id,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "rerank_result": rerank_result,
        "rerank_output_path": rerank_output_path,
        "eval_result": eval_result,
        "run_artifacts": run_artifacts,
    }
