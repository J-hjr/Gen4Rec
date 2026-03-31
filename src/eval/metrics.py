from __future__ import annotations

from typing import Any

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def compute_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot compute centroid from an empty embedding list.")
    centroid = np.mean(np.stack(embeddings), axis=0)
    return normalize_vector(centroid)


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_reference_similarity_metrics(
    clip_embedding: np.ndarray,
    reference_embeddings: list[np.ndarray],
    *,
    top_n: int,
) -> dict[str, float | None]:
    if not reference_embeddings:
        return {
            "reference_mean_cosine": None,
            "reference_max_cosine": None,
            "reference_topn_mean_cosine": None,
        }

    sims = [cosine_similarity(clip_embedding, ref_embedding) for ref_embedding in reference_embeddings]
    sorted_sims = sorted(sims, reverse=True)
    top_slice = sorted_sims[: max(1, min(top_n, len(sorted_sims)))]
    return {
        "reference_mean_cosine": float(np.mean(sims)),
        "reference_max_cosine": float(np.max(sims)),
        "reference_topn_mean_cosine": float(np.mean(top_slice)),
    }


def _pairwise_similarities(embeddings: list[np.ndarray]) -> list[float]:
    if len(embeddings) < 2:
        return []
    values = []
    for idx in range(len(embeddings)):
        for jdx in range(idx + 1, len(embeddings)):
            values.append(cosine_similarity(embeddings[idx], embeddings[jdx]))
    return values


def _nearest_neighbor_similarities(embeddings: list[np.ndarray]) -> list[float]:
    if len(embeddings) < 2:
        return []
    values = []
    for idx in range(len(embeddings)):
        sims = [
            cosine_similarity(embeddings[idx], embeddings[jdx])
            for jdx in range(len(embeddings))
            if jdx != idx
        ]
        values.append(max(sims))
    return values


def compute_diversity_metrics(embeddings: list[np.ndarray], *, prefix: str) -> dict[str, float | None]:
    pairwise = _pairwise_similarities(embeddings)
    nearest_neighbor = _nearest_neighbor_similarities(embeddings)
    out: dict[str, float | None] = {
        f"{prefix}_count": float(len(embeddings)),
        f"{prefix}_mean_pairwise_cosine": None,
        f"{prefix}_max_pairwise_cosine": None,
        f"{prefix}_mean_nearest_neighbor_cosine": None,
    }
    if pairwise:
        out[f"{prefix}_mean_pairwise_cosine"] = float(np.mean(pairwise))
        out[f"{prefix}_max_pairwise_cosine"] = float(np.max(pairwise))
    if nearest_neighbor:
        out[f"{prefix}_mean_nearest_neighbor_cosine"] = float(np.mean(nearest_neighbor))
    return out


def compute_gain(selected_summary: dict[str, float | None], candidate_summary: dict[str, float | None]) -> dict[str, float | None]:
    gain: dict[str, float | None] = {}
    for key, selected_value in selected_summary.items():
        candidate_value = candidate_summary.get(key)
        gain_key = f"gain_{key}"
        if selected_value is None or candidate_value is None:
            gain[gain_key] = None
        else:
            gain[gain_key] = float(selected_value - candidate_value)
    return gain


def build_candidate_metrics(
    *,
    candidates: list[dict[str, Any]],
    user_embedding: np.ndarray,
    reference_embeddings: list[np.ndarray],
    reference_labels: list[str],
    recent_centroid: np.ndarray | None,
    top_reference_k: int,
    imitation_threshold: float,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in candidates:
        clip_embedding = item["clip_embedding"]
        row = {key: value for key, value in item.items() if key != "clip_embedding"}
        row["user_embedding_cosine"] = cosine_similarity(clip_embedding, user_embedding)
        row["recent_centroid_cosine"] = (
            cosine_similarity(clip_embedding, recent_centroid) if recent_centroid is not None else None
        )
        reference_metrics = compute_reference_similarity_metrics(
            clip_embedding,
            reference_embeddings,
            top_n=top_reference_k,
        )
        row.update(reference_metrics)

        if reference_embeddings:
            reference_sims = [
                cosine_similarity(clip_embedding, ref_embedding)
                for ref_embedding in reference_embeddings
            ]
            best_idx = int(np.argmax(reference_sims))
            row["nearest_reference_label"] = reference_labels[best_idx]
            row["too_close_to_reference"] = bool(reference_sims[best_idx] >= imitation_threshold)
        else:
            row["nearest_reference_label"] = None
            row["too_close_to_reference"] = None

        enriched.append(row)
    return enriched
