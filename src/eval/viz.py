from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from src.eval.clap_audio import embed_audio_paths, load_audio_encoder
from src.eval.data import load_recent_reference_tracks
from src.eval.metrics import compute_centroid


def project_to_2d(matrix: np.ndarray, seed: int = 42) -> np.ndarray:
    if HAS_UMAP and matrix.shape[0] >= 5:
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(matrix)
    reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(matrix)


def build_generation_space_plot_df(
    *,
    user_id: str,
    reference_tracks: list[dict[str, Any]],
    candidate_tracks: list[dict[str, Any]],
    encoder_name: str,
) -> pd.DataFrame:
    records = []
    reference_embeddings = [track["embedding"] for track in reference_tracks]

    for track in reference_tracks:
        records.append(
            {
                "label": track["label"],
                "group": "recent_listens",
                "path": track["path"],
                "rerank_score": np.nan,
                "embedding": track["embedding"],
            }
        )

    for track in candidate_tracks:
        records.append(
            {
                "label": track.get("title") or Path(track["path"]).stem,
                "group": "selected_generated" if track.get("is_selected") else "generated_candidates",
                "path": track["path"],
                "rerank_score": track.get("clap_cosine_score"),
                "embedding": track["embedding"],
            }
        )

    if reference_embeddings:
        recent_centroid = compute_centroid(reference_embeddings)
        records.append(
            {
                "label": f"{user_id} recent centroid",
                "group": "recent_centroid",
                "path": "",
                "rerank_score": np.nan,
                "embedding": recent_centroid,
            }
        )

    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        raise ValueError("No points available to visualize.")
    coords = project_to_2d(np.stack(plot_df["embedding"].to_list()))
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["encoder"] = encoder_name
    return plot_df.drop(columns=["embedding"])


def build_generation_space_figure(
    *,
    plot_df: pd.DataFrame,
    user_id: str,
    recent_label_count: int = 5,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 9))
    style_map = {
        "recent_listens": {"color": "#4C78A8", "marker": "o", "size": 70, "alpha": 0.8},
        "generated_candidates": {"color": "#F58518", "marker": "^", "size": 90, "alpha": 0.75},
        "selected_generated": {"color": "#E45756", "marker": "D", "size": 120, "alpha": 0.95},
        "recent_centroid": {"color": "#111111", "marker": "*", "size": 260, "alpha": 1.0},
    }

    for group, style in style_map.items():
        group_df = plot_df.loc[plot_df["group"] == group]
        if group_df.empty:
            continue
        ax.scatter(
            group_df["x"],
            group_df["y"],
            c=style["color"],
            marker=style["marker"],
            s=style["size"],
            alpha=style["alpha"],
            label=group.replace("_", " "),
        )

    annotate_df = plot_df.loc[plot_df["group"].isin(["selected_generated", "recent_centroid"])].copy()
    annotate_df = pd.concat(
        [
            annotate_df,
            plot_df.loc[plot_df["group"] == "recent_listens"].head(recent_label_count),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["label", "group"])

    for _, row in annotate_df.iterrows():
        ax.text(
            row["x"] + 0.015,
            row["y"] + 0.01,
            row["label"],
            fontsize=9,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
        )

    encoder_name = str(plot_df["encoder"].iloc[0])
    ax.set_title(f"User generation space for {user_id} ({encoder_name} CLAP)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def load_candidate_tracks_from_run(run_root: str | Path) -> list[dict[str, Any]]:
    run_root = Path(run_root)
    rerank_path = run_root / "rerank_results.json"
    manifest_path = run_root / "run_manifest.json"
    if not rerank_path.exists():
        raise FileNotFoundError(f"Rerank results not found at {rerank_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found at {manifest_path}")

    rerank = json.loads(rerank_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_meta_by_path = {
        str(sample.get("path")): sample
        for sample in manifest.get("result", {}).get("samples", [])
    }
    selected_paths = {
        str(item.get("path"))
        for item in rerank.get("final_selected_tracks", [])
    }

    candidates = []
    for item in rerank.get("candidates", []):
        sample_meta = sample_meta_by_path.get(str(item["path"]), {})
        candidates.append(
            {
                "path": str(item["path"]),
                "title": item.get("title") or sample_meta.get("title"),
                "clap_cosine_score": item.get("clap_cosine_score"),
                "is_selected": str(item["path"]) in selected_paths,
            }
        )
    return candidates


def build_user_generation_plot_data(
    *,
    user_id: str,
    run_root: str | Path,
    top_recent: int = 20,
    encoder: str = "auto",
) -> tuple[pd.DataFrame, str]:
    encoder_bundle = load_audio_encoder(encoder)
    encoder_name = str(encoder_bundle[3]["encoder_name"])

    reference_tracks = load_recent_reference_tracks(user_id=user_id, top_recent=top_recent)
    candidate_tracks = load_candidate_tracks_from_run(run_root)

    reference_embeddings, _ = embed_audio_paths([track["path"] for track in reference_tracks], encoder=encoder_name)
    candidate_embeddings, _ = embed_audio_paths([track["path"] for track in candidate_tracks], encoder=encoder_name)

    for track in reference_tracks:
        track["embedding"] = reference_embeddings[track["path"]]
    for track in candidate_tracks:
        track["embedding"] = candidate_embeddings[track["path"]]

    plot_df = build_generation_space_plot_df(
        user_id=user_id,
        reference_tracks=reference_tracks,
        candidate_tracks=candidate_tracks,
        encoder_name=encoder_name,
    )
    return plot_df, encoder_name


def build_user_generation_figure(
    *,
    user_id: str,
    run_root: str | Path,
    top_recent: int = 20,
    recent_label_count: int = 5,
    encoder: str = "auto",
) -> tuple[plt.Figure, pd.DataFrame, str]:
    plot_df, encoder_name = build_user_generation_plot_data(
        user_id=user_id,
        run_root=run_root,
        top_recent=top_recent,
        encoder=encoder,
    )
    figure = build_generation_space_figure(
        plot_df=plot_df,
        user_id=user_id,
        recent_label_count=recent_label_count,
    )
    return figure, plot_df, encoder_name
