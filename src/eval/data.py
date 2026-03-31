from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent


def resolve_path(env_var: str, default_path: Path) -> str:
    env_value = os.environ.get(env_var)
    if env_value:
        return os.path.abspath(os.path.expanduser(env_value))
    return str(default_path.resolve())


class EvalDataConfig:
    DATASET_PATH = resolve_path("GEN4REC_DATASET_PATH", REPO_ROOT / "music4all")
    EMBEDDINGS_DIR = resolve_path(
        "GEN4REC_EMBED_OUTPUT_DIR",
        REPO_ROOT / "outputs" / "embeddings" / "music4all",
    )
    LISTENING_HISTORY_PATH = resolve_path(
        "GEN4REC_LISTENING_HISTORY_PATH",
        Path(DATASET_PATH) / "listening_history.csv",
    )
    ID_INFORMATION_PATH = resolve_path(
        "GEN4REC_ID_INFORMATION_PATH",
        Path(DATASET_PATH) / "id_information.csv",
    )
    AUDIO_DIR = resolve_path("GEN4REC_AUDIO_DIR", Path(DATASET_PATH) / "audios")
    USER_EMB_PATH = resolve_path("GEN4REC_USER_EMB_PATH", Path(EMBEDDINGS_DIR) / "user_embeddings.npy")
    USER_IDS_PATH = resolve_path("GEN4REC_USER_IDS_PATH", Path(EMBEDDINGS_DIR) / "user_ids.npy")


def ensure_local_file(path: str | Path, description: str) -> Path:
    path = Path(path)
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{description} not found at {path}. "
        "Please download or copy this file and place it at that path, "
        "or override the default location with the corresponding GEN4REC_* environment variable."
    )


def load_listening_history(path: str | Path | None = None) -> pd.DataFrame:
    path = ensure_local_file(path or EvalDataConfig.LISTENING_HISTORY_PATH, "Listening history table")
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


def load_id_information(path: str | Path | None = None) -> pd.DataFrame:
    path = ensure_local_file(path or EvalDataConfig.ID_INFORMATION_PATH, "Song information table")
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        df = pd.read_csv(path, sep="\t", header=None, names=["id", "artist", "song", "album_name"])
    df["id"] = df["id"].astype(str)
    return df


def build_id_to_label_map(path: str | Path | None = None) -> dict[str, str]:
    info_df = load_id_information(path)
    return {
        row["id"]: f"{row.get('artist', 'Unknown')} - {row.get('song', row['id'])}"
        for _, row in info_df.iterrows()
    }


def get_recent_unique_song_ids(
    history_df: pd.DataFrame,
    *,
    user_id: str,
    top_recent: int,
) -> list[str]:
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


def load_recent_reference_tracks(
    *,
    user_id: str,
    top_recent: int = 20,
) -> list[dict[str, Any]]:
    history_df = load_listening_history()
    label_map = build_id_to_label_map()
    audio_dir = Path(EvalDataConfig.AUDIO_DIR)
    song_ids = get_recent_unique_song_ids(history_df, user_id=user_id, top_recent=top_recent)

    reference_tracks: list[dict[str, Any]] = []
    for song_id in song_ids:
        audio_path = audio_dir / f"{song_id}.mp3"
        if not audio_path.exists():
            continue
        reference_tracks.append(
            {
                "song_id": song_id,
                "label": label_map.get(song_id, song_id),
                "path": str(audio_path),
            }
        )
    return reference_tracks


def load_user_embedding(user_id: str) -> np.ndarray:
    user_emb_path = ensure_local_file(EvalDataConfig.USER_EMB_PATH, "User embedding matrix")
    user_ids_path = ensure_local_file(EvalDataConfig.USER_IDS_PATH, "User ID array")
    user_embs = np.load(user_emb_path).astype(np.float32)
    user_ids = np.load(user_ids_path, allow_pickle=True).astype(str)
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    if user_id not in user_to_idx:
        raise ValueError(f"user_id not found in user_ids.npy: {user_id}")
    return user_embs[user_to_idx[user_id]].astype(np.float32)
