"""
Rerank generated music candidates with CLAP cosine similarity.

This module takes a generation run manifest, embeds each generated audio clip
with the same CLAP audio encoder used upstream, compares candidate embeddings
against the target user embedding, and returns a reranked shortlist with an
optional diversity filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.clap_audio import embed_audio_paths, load_audio_encoder
from src.eval.data import load_user_embedding


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def apply_diversity_filter(
    ranked: list[dict[str, Any]],
    *,
    top_k: int,
    diversity_threshold: float | None,
) -> list[dict[str, Any]]:
    if diversity_threshold is None:
        return ranked[:top_k]

    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        keep = True
        for chosen in selected:
            sim = cosine_similarity(candidate["clip_embedding"], chosen["clip_embedding"])
            if sim >= diversity_threshold:
                keep = False
                break
        if keep:
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def rerank_candidates(
    *,
    manifest: dict[str, Any],
    top_k: int,
    diversity_threshold: float | None,
    encoder: str,
) -> dict[str, Any]:
    user_id = str(manifest["user_id"])
    candidate_audio_paths = [str(path) for path in manifest.get("candidate_audio_paths", [])]
    if not candidate_audio_paths:
        raise ValueError("Manifest does not contain candidate_audio_paths.")

    _, _, _, encoder_cfg = load_audio_encoder(encoder)
    encoder_mode = str(encoder_cfg["encoder_name"])
    user_embedding = load_user_embedding(user_id)
    clip_embeddings, _ = embed_audio_paths(candidate_audio_paths, encoder=encoder)

    sample_meta_by_path = {
        str(sample.get("path")): sample
        for sample in manifest.get("result", {}).get("samples", [])
    }

    ranked_candidates = []
    for audio_path in candidate_audio_paths:
        clip_embedding = clip_embeddings[audio_path]
        score = cosine_similarity(clip_embedding, user_embedding)
        sample_meta = sample_meta_by_path.get(audio_path, {})
        ranked_candidates.append(
            {
                "path": audio_path,
                "title": sample_meta.get("title"),
                "call_index": sample_meta.get("call_index"),
                "variant_index": sample_meta.get("variant_index"),
                "metadata_path": sample_meta.get("metadata_path"),
                "lyric_path": sample_meta.get("text_companion"),
                "source_url": sample_meta.get("source_url"),
                "clap_cosine_score": score,
                "clip_embedding": clip_embedding,
            }
        )

    ranked_candidates.sort(key=lambda item: item["clap_cosine_score"], reverse=True)
    selected = apply_diversity_filter(
        ranked_candidates,
        top_k=top_k,
        diversity_threshold=diversity_threshold,
    )

    def strip_embedding(item: dict[str, Any]) -> dict[str, Any]:
        out = dict(item)
        out.pop("clip_embedding", None)
        return out

    return {
        "user_id": user_id,
        "candidate_count": len(ranked_candidates),
        "ranking_metric": "clap_cosine_similarity",
        "encoder": encoder_mode,
        "diversity_threshold": diversity_threshold,
        "candidates": [strip_embedding(item) for item in ranked_candidates],
        "reranked_list": [item["path"] for item in ranked_candidates],
        "final_selected_tracks": [strip_embedding(item) for item in selected],
    }


def run_rerank_from_manifest(
    *,
    manifest_path: str | Path,
    top_k: int = 2,
    diversity_threshold: float | None = None,
    encoder: str = "auto",
    output_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    manifest = load_manifest(manifest_path)
    resolved_output_path = str(output_path or Path(manifest_path).with_name("rerank_results.json"))
    rerank_result = rerank_candidates(
        manifest=manifest,
        top_k=max(1, top_k),
        diversity_threshold=diversity_threshold,
        encoder=encoder,
    )
    save_json(rerank_result, resolved_output_path)
    return rerank_result, resolved_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank generated candidate clips by CLAP cosine similarity.")
    parser.add_argument("--manifest", required=True, help="Path to a generation run manifest JSON.")
    parser.add_argument("--top-k", type=int, default=2, help="How many final tracks to keep after reranking.")
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=None,
        help="Optional cosine threshold for filtering near-duplicate generated clips.",
    )
    parser.add_argument(
        "--encoder",
        choices=["auto", "finetuned", "zeroshot"],
        default="auto",
        help="CLAP encoder to use for reranking. `auto` falls back to zero-shot if the finetuned checkpoint is unavailable.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to `rerank_results.json` next to the manifest.",
    )
    args = parser.parse_args()

    rerank_result, output_path = run_rerank_from_manifest(
        manifest_path=args.manifest,
        top_k=max(1, args.top_k),
        diversity_threshold=args.diversity_threshold,
        encoder=args.encoder,
        output_path=args.output,
    )

    print("Rerank completed.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
