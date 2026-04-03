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
from src.eval.data import load_recent_reference_tracks, load_user_embedding
from src.eval.metrics import (
    build_candidate_metrics,
    compute_centroid,
    compute_diversity_metrics,
    summarize_values,
)
from src.eval.reporting import save_csv, save_json, write_eval_report
from src.eval.viz import (
    build_generation_space_figure,
    build_generation_space_plot_df,
    save_figure,
)
from src.generate.rerank import load_manifest, rerank_candidates, save_json as save_rerank_json


def _load_or_create_rerank(
    *,
    run_root: Path,
    manifest: dict[str, Any],
    encoder: str,
    rerank_top_k: int,
    diversity_threshold: float | None,
) -> tuple[dict[str, Any], Path]:
    rerank_path = run_root / "rerank_results.json"
    if rerank_path.exists():
        return json.loads(rerank_path.read_text(encoding="utf-8")), rerank_path

    rerank_result = rerank_candidates(
        manifest=manifest,
        top_k=max(1, rerank_top_k),
        diversity_threshold=diversity_threshold,
        encoder=encoder,
    )
    save_rerank_json(rerank_result, rerank_path)
    return rerank_result, rerank_path


def _build_eval_artifact_paths(run_root: Path, output_dir: str | Path | None = None) -> dict[str, str]:
    if output_dir:
        base_dir = Path(output_dir)
    else:
        user_id = run_root.parent.name
        run_id = run_root.name
        base_dir = REPO_ROOT / "outputs" / "eval" / user_id / run_id
    return {
        "eval_summary_json": str(base_dir / "eval_summary.json"),
        "eval_report_md": str(base_dir / "eval_report.md"),
        "reference_alignment_csv": str(base_dir / "reference_alignment.csv"),
        "embedding_space_png": str(base_dir / "embedding_space.png"),
    }


def _build_candidate_track_rows(
    rerank_result: dict[str, Any],
    candidate_embeddings: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    selected_paths = {
        str(item.get("path"))
        for item in rerank_result.get("final_selected_tracks", [])
    }

    rows = []
    for item in rerank_result.get("candidates", []):
        row = dict(item)
        row["is_selected"] = str(item["path"]) in selected_paths
        row["clip_embedding"] = candidate_embeddings[str(item["path"])]
        rows.append(row)
    return rows


def _aggregate_candidate_metrics(candidates: list[dict[str, Any]], *, selected_only: bool) -> dict[str, float | None]:
    rows = [row for row in candidates if row["is_selected"]] if selected_only else candidates

    def collect(metric: str) -> list[float]:
        return [float(row[metric]) for row in rows if row.get(metric) is not None]

    summary = {
        "user_embedding_cosine_mean": summarize_values(collect("user_embedding_cosine"))["mean"],
        "recent_centroid_cosine_mean": summarize_values(collect("recent_centroid_cosine"))["mean"],
        "reference_mean_cosine_mean": summarize_values(collect("reference_mean_cosine"))["mean"],
        "reference_max_cosine_mean": summarize_values(collect("reference_max_cosine"))["mean"],
        "reference_topk_mean_cosine_mean": summarize_values(collect("reference_topk_mean_cosine"))["mean"],
    }
    return summary


def _build_metric_panels(
    *,
    aggregate_metrics: dict[str, float | None],
    diversity_metrics: dict[str, float | None],
    candidates_with_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    selected_rows = [row for row in candidates_with_metrics if row["is_selected"]]
    selected_too_close_count = int(sum(1 for row in selected_rows if row.get("too_close_to_reference") is True))
    return {
        "personalization": {
            "selected_user_embedding_cosine_mean": aggregate_metrics.get("selected_user_embedding_cosine_mean"),
            "gain_user_embedding_cosine_mean": aggregate_metrics.get("gain_user_embedding_cosine_mean"),
            "selected_recent_centroid_cosine_mean": aggregate_metrics.get("selected_recent_centroid_cosine_mean"),
            "gain_recent_centroid_cosine_mean": aggregate_metrics.get("gain_recent_centroid_cosine_mean"),
            "selected_reference_topk_mean_cosine_mean": aggregate_metrics.get(
                "selected_reference_topk_mean_cosine_mean"
            ),
            "gain_reference_topk_mean_cosine_mean": aggregate_metrics.get("gain_reference_topk_mean_cosine_mean"),
        },
        "diversity": {
            "selected_mean_pairwise_cosine": diversity_metrics.get("selected_mean_pairwise_cosine"),
            "selected_mean_nearest_neighbor_cosine": diversity_metrics.get("selected_mean_nearest_neighbor_cosine"),
            "candidate_mean_pairwise_cosine": diversity_metrics.get("candidate_mean_pairwise_cosine"),
        },
        "risk": {
            "candidate_too_close_to_reference_count": diversity_metrics.get("too_close_to_reference_count"),
            "selected_too_close_to_reference_count": selected_too_close_count,
        },
    }


def evaluate_generation_run(
    *,
    manifest_path: str | Path,
    recent_k: int = 20,
    reference_top_k: int = 3,
    encoder: str = "finetuned",
    rerank_top_k: int = 2,
    diversity_threshold: float | None = None,
    output_dir: str | Path | None = None,
    save_plot: bool = False,
    imitation_threshold: float = 0.9,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    run_root = manifest_path.parent
    rerank_result, rerank_path = _load_or_create_rerank(
        run_root=run_root,
        manifest=manifest,
        encoder=encoder,
        rerank_top_k=rerank_top_k,
        diversity_threshold=diversity_threshold,
    )

    reference_tracks = load_recent_reference_tracks(
        user_id=str(manifest["user_id"]),
        top_recent=recent_k,
    )
    if not reference_tracks:
        raise ValueError("No recent reference tracks with local audio files were found for this user.")

    candidate_paths = [str(item["path"]) for item in rerank_result.get("candidates", [])]
    if not candidate_paths:
        raise ValueError("No candidate tracks found in rerank results.")

    _, _, _, encoder_cfg = load_audio_encoder(encoder)
    resolved_encoder = str(encoder_cfg["encoder_name"])

    reference_embeddings_map, _ = embed_audio_paths([track["path"] for track in reference_tracks], encoder=resolved_encoder)
    candidate_embeddings_map, _ = embed_audio_paths(candidate_paths, encoder=resolved_encoder)
    reference_embeddings = [reference_embeddings_map[track["path"]] for track in reference_tracks]
    reference_labels = [track["label"] for track in reference_tracks]
    recent_centroid = compute_centroid(reference_embeddings) if reference_embeddings else None
    user_embedding = load_user_embedding(str(manifest["user_id"]))

    candidate_rows = _build_candidate_track_rows(rerank_result, candidate_embeddings_map)
    candidates_with_metrics = build_candidate_metrics(
        candidates=candidate_rows,
        user_embedding=user_embedding,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        recent_centroid=recent_centroid,
        reference_top_k=reference_top_k,
        imitation_threshold=imitation_threshold,
    )

    candidate_summary = _aggregate_candidate_metrics(candidates_with_metrics, selected_only=False)
    selected_summary = _aggregate_candidate_metrics(candidates_with_metrics, selected_only=True)
    aggregate_metrics = {}
    for metric_name, candidate_value in candidate_summary.items():
        selected_value = selected_summary.get(metric_name)
        aggregate_metrics[f"candidate_{metric_name}"] = candidate_value
        aggregate_metrics[f"selected_{metric_name}"] = selected_value
        aggregate_metrics[f"gain_{metric_name}"] = (
            None
            if candidate_value is None or selected_value is None
            else float(selected_value - candidate_value)
        )

    candidate_clip_embeddings = [row["clip_embedding"] for row in candidate_rows]
    selected_clip_embeddings = [row["clip_embedding"] for row in candidate_rows if row["is_selected"]]
    diversity_metrics = {
        **compute_diversity_metrics(candidate_clip_embeddings, prefix="candidate"),
        **compute_diversity_metrics(selected_clip_embeddings, prefix="selected"),
        "too_close_to_reference_count": int(
            sum(1 for row in candidates_with_metrics if row.get("too_close_to_reference") is True)
        ),
    }
    metric_panels = _build_metric_panels(
        aggregate_metrics=aggregate_metrics,
        diversity_metrics=diversity_metrics,
        candidates_with_metrics=candidates_with_metrics,
    )

    plot_reference_tracks = []
    for track in reference_tracks:
        plot_reference_tracks.append(
            {
                **track,
                "embedding": reference_embeddings_map[track["path"]],
            }
        )

    plot_candidate_tracks = []
    for row in candidate_rows:
        plot_candidate_tracks.append(
            {
                "path": row["path"],
                "title": row.get("title"),
                "clap_cosine_score": row.get("clap_cosine_score"),
                "is_selected": row.get("is_selected", False),
                "embedding": row["clip_embedding"],
            }
        )

    plot_df = build_generation_space_plot_df(
        user_id=str(manifest["user_id"]),
        reference_tracks=plot_reference_tracks,
        candidate_tracks=plot_candidate_tracks,
        encoder_name=resolved_encoder,
    )
    figure = build_generation_space_figure(
        plot_df=plot_df,
        user_id=str(manifest["user_id"]),
    )

    artifact_paths = _build_eval_artifact_paths(run_root, output_dir)
    if save_plot:
        save_figure(figure, artifact_paths["embedding_space_png"])
        saved_plot_path: str | None = artifact_paths["embedding_space_png"]
    else:
        saved_plot_path = None

    summary = {
        "run": {
            "run_id": str(manifest["run_id"]),
            "user_id": str(manifest["user_id"]),
            "manifest_path": str(manifest_path),
            "rerank_path": str(rerank_path),
            "encoder": resolved_encoder,
            "candidate_count": len(candidates_with_metrics),
            "selected_count": len([row for row in candidates_with_metrics if row["is_selected"]]),
        },
        "reference_set": {
            "recent_k": int(recent_k),
            "reference_track_count": len(reference_tracks),
            "reference_top_k": int(reference_top_k),
            "labels": reference_labels,
        },
        "metric_panels": metric_panels,
        "aggregate_metrics": aggregate_metrics,
        "diversity_metrics": diversity_metrics,
        "candidates": [
            {key: value for key, value in row.items() if key != "clip_embedding"}
            for row in candidates_with_metrics
        ],
        "selected_tracks": [
            {key: value for key, value in row.items() if key != "clip_embedding"}
            for row in candidates_with_metrics
            if row["is_selected"]
        ],
        "artifacts": {
            **artifact_paths,
            "embedding_space_png": saved_plot_path,
        },
    }

    save_json(summary, artifact_paths["eval_summary_json"])
    save_csv(summary["candidates"], artifact_paths["reference_alignment_csv"])
    write_eval_report(summary, artifact_paths["eval_report_md"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automatic evaluation for a generated music run.")
    parser.add_argument("--manifest", required=True, help="Path to a generation run manifest JSON.")
    parser.add_argument("--recent-k", type=int, default=20, help="How many recent listened songs to embed as references.")
    parser.add_argument(
        "--reference-top-k",
        type=int,
        default=3,
        help="How many nearest reference tracks to average for the reference top-k mean metric.",
    )
    parser.add_argument(
        "--top-reference-k",
        dest="reference_top_k",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--encoder",
        choices=["auto", "finetuned", "zeroshot"],
        default="finetuned",
        help="Embedding encoder for automatic eval. `finetuned` is the project default.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=2,
        help="If rerank_results.json is missing, use this top-k when recomputing rerank.",
    )
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=None,
        help="Optional diversity threshold if rerank must be recomputed.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save `embedding_space.png` next to the run-level eval artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional custom output directory for eval artifacts. Defaults to outputs/eval/<USER_ID>/<RUN_ID>/.",
    )
    parser.add_argument(
        "--imitation-threshold",
        type=float,
        default=0.9,
        help="Reference similarity threshold used to flag potentially over-close generations.",
    )
    args = parser.parse_args()

    summary = evaluate_generation_run(
        manifest_path=args.manifest,
        recent_k=max(1, args.recent_k),
        reference_top_k=max(1, args.reference_top_k),
        encoder=args.encoder,
        rerank_top_k=max(1, args.rerank_top_k),
        diversity_threshold=args.diversity_threshold,
        output_dir=args.output_dir,
        save_plot=args.save_plot,
        imitation_threshold=args.imitation_threshold,
    )
    print("Eval completed.")
    print(f"Eval summary: {summary['artifacts']['eval_summary_json']}")
    print(f"Eval report: {summary['artifacts']['eval_report_md']}")
    if summary["artifacts"]["embedding_space_png"]:
        print(f"Embedding plot: {summary['artifacts']['embedding_space_png']}")


if __name__ == "__main__":
    main()
