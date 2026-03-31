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
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embed.embed_music4all import Config as FinetunedEmbedConfig
from src.embed.embed_music4all import load_finetuned_model_and_attention
from src.embed.embed_music4all_zeroshot import Config as EmbedConfig
from src.embed.embed_music4all_zeroshot import load_zeroshot_clap


def resolve_path(env_var: str, default_path: Path) -> str:
    env_value = os.environ.get(env_var)
    if env_value:
        return os.path.abspath(os.path.expanduser(env_value))
    return str(default_path.resolve())


class RerankConfig:
    EMBEDDINGS_DIR = resolve_path(
        "GEN4REC_EMBED_OUTPUT_DIR",
        REPO_ROOT / "outputs" / "embeddings" / "music4all",
    )
    USER_EMB_PATH = resolve_path("GEN4REC_USER_EMB_PATH", Path(EMBEDDINGS_DIR) / "user_embeddings.npy")
    USER_IDS_PATH = resolve_path("GEN4REC_USER_IDS_PATH", Path(EMBEDDINGS_DIR) / "user_ids.npy")


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_user_embedding(user_id: str) -> np.ndarray:
    user_embs = np.load(RerankConfig.USER_EMB_PATH).astype(np.float32)
    user_ids = np.load(RerankConfig.USER_IDS_PATH, allow_pickle=True).astype(str)
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    if user_id not in user_to_idx:
        raise ValueError(f"user_id not found in user_ids.npy: {user_id}")
    return user_embs[user_to_idx[user_id]].astype(np.float32)


def prepare_audio_chunks(
    path: str | Path,
    sample_rate: int,
    num_chunks: int,
    chunk_samples: int,
) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)
    total_len = waveform.shape[0]
    chunks = []

    if total_len <= chunk_samples:
        pad_len = chunk_samples - total_len
        padded = F.pad(waveform, (0, pad_len))
        for _ in range(num_chunks):
            chunks.append(padded)
    else:
        max_start = total_len - chunk_samples
        start_points = np.linspace(0, max_start, num_chunks).astype(int)
        for start in start_points:
            chunks.append(waveform[start : start + chunk_samples])

    return torch.stack(chunks)


def embed_audio_file(
    path: str | Path,
    *,
    model: torch.nn.Module,
    attention_pool: torch.nn.Module | None,
    embedding_dim: int,
    device: str,
) -> np.ndarray:
    chunks = prepare_audio_chunks(
        path=path,
        sample_rate=EmbedConfig.SAMPLE_RATE,
        num_chunks=EmbedConfig.NUM_CHUNKS,
        chunk_samples=EmbedConfig.CHUNK_SAMPLES,
    )
    chunks = chunks.unsqueeze(0)  # [1, C, T]
    _, num_chunks, samples = chunks.shape

    with torch.no_grad():
        flat_audio = chunks.view(num_chunks, samples).to(device)
        audio_dict = {"waveform": flat_audio}
        output_dict = model.audio_branch(audio_dict)
        if isinstance(output_dict, dict):
            if "embedding" in output_dict:
                flat_audio_features = output_dict["embedding"]
            else:
                flat_audio_features = list(output_dict.values())[0]
        else:
            flat_audio_features = output_dict

        if hasattr(model, "audio_projection"):
            flat_audio_features = model.audio_projection(flat_audio_features)

        unflattened = flat_audio_features.view(1, num_chunks, embedding_dim)
        if attention_pool is None:
            pooled = unflattened.mean(dim=1)
        else:
            pooled = attention_pool(unflattened)
        pooled = F.normalize(pooled, dim=-1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)


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

    device = EmbedConfig.DEVICE
    encoder_mode = encoder
    if encoder_mode == "auto":
        encoder_mode = "finetuned" if os.path.exists(FinetunedEmbedConfig.FINETUNED_CKPT) else "zeroshot"

    if encoder_mode == "finetuned":
        model, attention_pool, embedding_dim = load_finetuned_model_and_attention(device)
    elif encoder_mode == "zeroshot":
        model = load_zeroshot_clap(device)
        attention_pool = None
        with torch.no_grad():
            dummy_wave = torch.zeros(1, EmbedConfig.CHUNK_SAMPLES).to(device)
            dummy_out = model.audio_branch({"waveform": dummy_wave})
            if isinstance(dummy_out, dict):
                dummy_emb = dummy_out.get("embedding", list(dummy_out.values())[0])
            else:
                dummy_emb = dummy_out
            if hasattr(model, "audio_projection"):
                dummy_emb = model.audio_projection(dummy_emb)
            embedding_dim = int(dummy_emb.shape[-1])
    else:
        raise ValueError(f"Unsupported encoder mode: {encoder_mode}")

    user_embedding = load_user_embedding(user_id)

    sample_meta_by_path = {
        str(sample.get("path")): sample
        for sample in manifest.get("result", {}).get("samples", [])
    }

    ranked_candidates = []
    for audio_path in candidate_audio_paths:
        clip_embedding = embed_audio_file(
            path=audio_path,
            model=model,
            attention_pool=attention_pool,
            embedding_dim=embedding_dim,
            device=device,
        )
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
