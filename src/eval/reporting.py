from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


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
    aggregate = summary["aggregate_metrics"]
    reference = summary["reference_set"]
    diversity = summary["diversity_metrics"]

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
- Top-reference-k metric: `{reference['top_reference_k']}`

## Aggregate Alignment

- Candidate mean user-embedding cosine: `{aggregate['candidate_user_embedding_cosine_mean']}`
- Selected mean user-embedding cosine: `{aggregate['selected_user_embedding_cosine_mean']}`
- Gain: `{aggregate['gain_user_embedding_cosine_mean']}`
- Candidate mean centroid cosine: `{aggregate['candidate_recent_centroid_cosine_mean']}`
- Selected mean centroid cosine: `{aggregate['selected_recent_centroid_cosine_mean']}`
- Gain: `{aggregate['gain_recent_centroid_cosine_mean']}`
- Candidate mean reference cosine: `{aggregate['candidate_reference_mean_cosine_mean']}`
- Selected mean reference cosine: `{aggregate['selected_reference_mean_cosine_mean']}`
- Gain: `{aggregate['gain_reference_mean_cosine_mean']}`

## Diversity

- Candidate mean pairwise cosine: `{diversity['candidate_mean_pairwise_cosine']}`
- Selected mean pairwise cosine: `{diversity['selected_mean_pairwise_cosine']}`
- Candidate mean nearest-neighbor cosine: `{diversity['candidate_mean_nearest_neighbor_cosine']}`
- Selected mean nearest-neighbor cosine: `{diversity['selected_mean_nearest_neighbor_cosine']}`

## Artifacts

- Eval summary JSON: `{summary['artifacts']['eval_summary_json']}`
- Eval report MD: `{summary['artifacts']['eval_report_md']}`
- Reference alignment CSV: `{summary['artifacts']['reference_alignment_csv']}`
- Embedding plot: `{summary['artifacts'].get('embedding_space_png') or 'Not saved'}`
"""
    path.write_text(report, encoding="utf-8")
