from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_eval_report(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    run = summary["run"]
    panels = summary.get("metric_panels", {})
    aggregate = summary["aggregate_metrics"]
    reference = summary["reference_set"]
    diversity = summary["diversity_metrics"]
    reference_top_k = reference.get("reference_top_k", reference.get("top_reference_k"))
    personalization_panel = panels.get("personalization", {})
    selected_reference_topk = personalization_panel.get(
        "selected_reference_topk_mean_cosine_mean",
        personalization_panel.get("selected_reference_topn_mean_cosine_mean"),
    )
    gain_reference_topk = personalization_panel.get(
        "gain_reference_topk_mean_cosine_mean",
        personalization_panel.get("gain_reference_topn_mean_cosine_mean"),
    )

    report = f"""# Eval Report

## Run

- Run ID: `{run['run_id']}`
- User ID: `{run['user_id']}`
- Manifest: `{run['manifest_path']}`
- Encoder: `{run['encoder']}`
- Candidate count: `{run['candidate_count']}`
- Selected count: `{run['selected_count']}`

## Reference Set

- Recent-K used: `{reference['recent_k']}`
- Valid reference tracks embedded: `{reference['reference_track_count']}`
- Reference top-k: `{reference_top_k}`

## Three-Panel Summary

### Personalization

- Selected mean user-embedding cosine: `{_fmt(panels.get('personalization', {}).get('selected_user_embedding_cosine_mean'))}`
- Gain vs candidate pool: `{_fmt(panels.get('personalization', {}).get('gain_user_embedding_cosine_mean'))}`
- Selected mean recent-centroid cosine: `{_fmt(panels.get('personalization', {}).get('selected_recent_centroid_cosine_mean'))}`
- Gain vs candidate pool: `{_fmt(panels.get('personalization', {}).get('gain_recent_centroid_cosine_mean'))}`
- Selected mean reference top-k cosine: `{_fmt(selected_reference_topk)}`
- Gain vs candidate pool: `{_fmt(gain_reference_topk)}`

### Diversity

- Selected mean pairwise cosine: `{_fmt(panels.get('diversity', {}).get('selected_mean_pairwise_cosine'))}`
- Selected mean nearest-neighbor cosine: `{_fmt(panels.get('diversity', {}).get('selected_mean_nearest_neighbor_cosine'))}`
- Candidate mean pairwise cosine: `{_fmt(panels.get('diversity', {}).get('candidate_mean_pairwise_cosine'))}`

### Risk

- Candidate too-close-to-reference count: `{_fmt(panels.get('risk', {}).get('candidate_too_close_to_reference_count'))}`
- Selected too-close-to-reference count: `{_fmt(panels.get('risk', {}).get('selected_too_close_to_reference_count'))}`

## Full Diagnostics

- Candidate mean user-embedding cosine: `{_fmt(aggregate['candidate_user_embedding_cosine_mean'])}`
- Selected mean user-embedding cosine: `{_fmt(aggregate['selected_user_embedding_cosine_mean'])}`
- Candidate mean centroid cosine: `{_fmt(aggregate['candidate_recent_centroid_cosine_mean'])}`
- Selected mean centroid cosine: `{_fmt(aggregate['selected_recent_centroid_cosine_mean'])}`
- Candidate mean reference cosine: `{_fmt(aggregate['candidate_reference_mean_cosine_mean'])}`
- Selected mean reference cosine: `{_fmt(aggregate['selected_reference_mean_cosine_mean'])}`
- Candidate mean pairwise cosine: `{_fmt(diversity['candidate_mean_pairwise_cosine'])}`
- Selected mean pairwise cosine: `{_fmt(diversity['selected_mean_pairwise_cosine'])}`
- Candidate mean nearest-neighbor cosine: `{_fmt(diversity['candidate_mean_nearest_neighbor_cosine'])}`
- Selected mean nearest-neighbor cosine: `{_fmt(diversity['selected_mean_nearest_neighbor_cosine'])}`

## Artifacts

- Eval summary JSON: `{summary['artifacts']['eval_summary_json']}`
- Eval report MD: `{summary['artifacts']['eval_report_md']}`
- Reference alignment CSV: `{summary['artifacts']['reference_alignment_csv']}`
- Embedding plot: `{summary['artifacts'].get('embedding_space_png') or 'Not saved'}`
"""
    path.write_text(report, encoding="utf-8")
