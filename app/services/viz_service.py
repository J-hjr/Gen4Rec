from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.decomposition import PCA

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embed.embed_music4all import Config as FinetunedConfig
from src.embed.embed_music4all import load_finetuned_model_and_attention
from src.embed.embed_music4all_zeroshot import Config as ZeroShotConfig
from src.embed.embed_music4all_zeroshot import load_zeroshot_clap


LISTENING_HISTORY_PATH = REPO_ROOT / "music4all" / "listening_history.csv"
ID_INFO_PATH = REPO_ROOT / "music4all" / "id_information.csv"
AUDIO_DIR = REPO_ROOT / "music4all" / "audios"


def load_listening_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) == 1 and "\t" in df.columns[0]:
        fixed = pd.read_csv(path, sep="\t", header=None)
        fixed.columns = ["user", "song", "timestamp"][: fixed.shape[1]]
        df = fixed
    lower_map = {c.lower(): c for c in df.columns}
    user_col = lower_map.get("user") or lower_map.get("user_id")
    song_col = lower_map.get("song") or lower_map.get("song_id")
    ts_col = lower_map.get("timestamp")
    if user_col is None or song_col is None:
        raise ValueError("Could not find user/song columns in listening_history.csv")
    out = df.rename(columns={user_col: "user_id", song_col: "song_id"}).copy()
    out["user_id"] = out["user_id"].astype(str)
    out["song_id"] = out["song_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out[ts_col], errors="coerce") if ts_col is not None else pd.NaT
    return out[["user_id", "song_id", "timestamp"]]


def load_id_information(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        df = pd.read_csv(path, sep="\t", header=None, names=["id", "artist", "song", "album_name"])
    df["id"] = df["id"].astype(str)
    return df


def get_recent_unique_song_ids(history_df: pd.DataFrame, user_id: str, top_recent: int) -> list[str]:
    user_df = history_df.loc[history_df["user_id"] == user_id].copy()
    user_df = user_df.sort_values("timestamp", ascending=False, na_position="last")
    seen = set()
    recent_ids: list[str] = []
    for song_id in user_df["song_id"].tolist():
        if song_id not in seen:
            recent_ids.append(song_id)
            seen.add(song_id)
        if len(recent_ids) >= top_recent:
            break
    return recent_ids


@lru_cache(maxsize=1)
def load_preferred_audio_encoder() -> tuple[torch.nn.Module, torch.nn.Module | None, int, dict[str, object]]:
    try:
        model, attention_pool, embedding_dim = load_finetuned_model_and_attention(FinetunedConfig.DEVICE)
        return model, attention_pool, int(embedding_dim), {
            "encoder_name": "finetuned",
            "device": FinetunedConfig.DEVICE,
            "sample_rate": FinetunedConfig.SAMPLE_RATE,
            "num_chunks": FinetunedConfig.NUM_CHUNKS,
            "chunk_samples": FinetunedConfig.CHUNK_SAMPLES,
        }
    except Exception as exc:
        print(f"Falling back to zero-shot CLAP: {exc}")
        model = load_zeroshot_clap(ZeroShotConfig.DEVICE)
        with torch.no_grad():
            dummy_wave = torch.zeros(1, ZeroShotConfig.CHUNK_SAMPLES).to(ZeroShotConfig.DEVICE)
            dummy_out = model.audio_branch({"waveform": dummy_wave})
            if isinstance(dummy_out, dict):
                dummy_emb = dummy_out.get("embedding", list(dummy_out.values())[0])
            else:
                dummy_emb = dummy_out
            if hasattr(model, "audio_projection"):
                dummy_emb = model.audio_projection(dummy_emb)
            embedding_dim = int(dummy_emb.shape[-1])
        return model, None, embedding_dim, {
            "encoder_name": "zeroshot",
            "device": ZeroShotConfig.DEVICE,
            "sample_rate": ZeroShotConfig.SAMPLE_RATE,
            "num_chunks": ZeroShotConfig.NUM_CHUNKS,
            "chunk_samples": ZeroShotConfig.CHUNK_SAMPLES,
        }


def prepare_audio_chunks(path: Path, sample_rate: int, num_chunks: int, chunk_samples: int) -> torch.Tensor:
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
        padded = F.pad(waveform, (0, chunk_samples - total_len))
        for _ in range(num_chunks):
            chunks.append(padded)
    else:
        max_start = total_len - chunk_samples
        start_points = np.linspace(0, max_start, num_chunks).astype(int)
        for start in start_points:
            chunks.append(waveform[start : start + chunk_samples])
    return torch.stack(chunks)


def embed_audio_file(
    path: Path,
    *,
    model: torch.nn.Module,
    attention_pool: torch.nn.Module | None,
    embedding_dim: int,
    device: str,
    sample_rate: int,
    num_chunks: int,
    chunk_samples: int,
) -> np.ndarray:
    chunks = prepare_audio_chunks(
        path,
        sample_rate=sample_rate,
        num_chunks=num_chunks,
        chunk_samples=chunk_samples,
    ).unsqueeze(0)
    _, observed_num_chunks, samples = chunks.shape
    with torch.no_grad():
        flat_audio = chunks.view(observed_num_chunks, samples).to(device)
        output_dict = model.audio_branch({"waveform": flat_audio})
        if isinstance(output_dict, dict):
            flat_audio_features = output_dict.get("embedding", list(output_dict.values())[0])
        else:
            flat_audio_features = output_dict
        if hasattr(model, "audio_projection"):
            flat_audio_features = model.audio_projection(flat_audio_features)
        unflattened = flat_audio_features.view(1, observed_num_chunks, embedding_dim)
        if attention_pool is None:
            pooled = unflattened.mean(dim=1)
        else:
            pooled = attention_pool(unflattened)
        pooled = F.normalize(pooled, dim=-1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def project_to_2d(matrix: np.ndarray, seed: int = 42) -> np.ndarray:
    if HAS_UMAP and matrix.shape[0] >= 5:
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(matrix)
    reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(matrix)


def build_user_generation_plot_data(
    *,
    user_id: str,
    run_root: str | Path,
    top_recent: int = 20,
) -> tuple[pd.DataFrame, str]:
    run_root = Path(run_root)
    rerank_path = run_root / "rerank_results.json"
    if not rerank_path.exists():
        raise FileNotFoundError(f"Rerank results not found at {rerank_path}")

    rerank = json.loads(rerank_path.read_text(encoding="utf-8"))
    history_df = load_listening_history(LISTENING_HISTORY_PATH)
    id_info_df = load_id_information(ID_INFO_PATH)
    id_to_label = {
        row["id"]: f"{row.get('artist', 'Unknown')} - {row.get('song', row['id'])}"
        for _, row in id_info_df.iterrows()
    }

    recent_song_ids = get_recent_unique_song_ids(history_df, user_id=user_id, top_recent=top_recent)
    recent_audio_paths = []
    for song_id in recent_song_ids:
        audio_path = AUDIO_DIR / f"{song_id}.mp3"
        if audio_path.exists():
            recent_audio_paths.append((song_id, audio_path))

    candidate_rows = rerank["candidates"]
    selected_paths = {item["path"] for item in rerank["final_selected_tracks"]}

    model, attention_pool, embedding_dim, encoder_cfg = load_preferred_audio_encoder()
    device = str(encoder_cfg["device"])
    sample_rate = int(encoder_cfg["sample_rate"])
    num_chunks = int(encoder_cfg["num_chunks"])
    chunk_samples = int(encoder_cfg["chunk_samples"])

    records = []
    recent_embeddings = []
    for song_id, audio_path in recent_audio_paths:
        emb = embed_audio_file(
            audio_path,
            model=model,
            attention_pool=attention_pool,
            embedding_dim=embedding_dim,
            device=device,
            sample_rate=sample_rate,
            num_chunks=num_chunks,
            chunk_samples=chunk_samples,
        )
        recent_embeddings.append(emb)
        records.append(
            {
                "label": id_to_label.get(song_id, song_id),
                "group": "recent_listens",
                "path": str(audio_path),
                "rerank_score": np.nan,
                "embedding": emb,
            }
        )

    for item in candidate_rows:
        audio_path = Path(item["path"])
        emb = embed_audio_file(
            audio_path,
            model=model,
            attention_pool=attention_pool,
            embedding_dim=embedding_dim,
            device=device,
            sample_rate=sample_rate,
            num_chunks=num_chunks,
            chunk_samples=chunk_samples,
        )
        records.append(
            {
                "label": item.get("title") or audio_path.stem,
                "group": "selected_generated" if str(audio_path) in selected_paths else "generated_candidates",
                "path": str(audio_path),
                "rerank_score": item.get("clap_cosine_score"),
                "embedding": emb,
            }
        )

    if recent_embeddings:
        recent_centroid = np.mean(np.stack(recent_embeddings), axis=0)
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
    plot_matrix = np.stack(plot_df["embedding"].to_list())
    coords = project_to_2d(plot_matrix)
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["encoder"] = str(encoder_cfg["encoder_name"])
    plot_df = plot_df.drop(columns=["embedding"])
    return plot_df, str(encoder_cfg["encoder_name"])


def build_user_generation_figure(
    *,
    user_id: str,
    run_root: str | Path,
    top_recent: int = 20,
    recent_label_count: int = 5,
) -> tuple[plt.Figure, pd.DataFrame, str]:
    plot_df, encoder_name = build_user_generation_plot_data(
        user_id=user_id,
        run_root=run_root,
        top_recent=top_recent,
    )

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

    ax.set_title(f"User generation space for {user_id} ({encoder_name} CLAP)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, plot_df, encoder_name
