"""
Export Top-K songs nearest to a user embedding in CLAP space as JSON.

Output is info + metadata (+ genres/tags) only, for the User Profile & LLM phase.
No natural-language summary here — teammates consume this JSON downstream.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR_PATH = Path(__file__).resolve().parent
REPO_ROOT_PATH = BASE_DIR_PATH.parent.parent


def resolve_path(env_var: str, default_path: Path) -> str:
    env_value = os.environ.get(env_var)
    if env_value:
        return os.path.abspath(os.path.expanduser(env_value))
    return str(default_path.resolve())


class Config:
    DATASET_PATH = resolve_path("GEN4REC_DATASET_PATH", REPO_ROOT_PATH / "music4all")
    EMBEDDINGS_DIR = resolve_path(
        "GEN4REC_EMBED_OUTPUT_DIR",
        REPO_ROOT_PATH / "outputs" / "embeddings" / "music4all",
    )
    LISTENING_HISTORY_PATH = resolve_path(
        "GEN4REC_LISTENING_HISTORY_PATH",
        Path(DATASET_PATH) / "listening_history.csv",
    )
    SONG_EMB_PATH = resolve_path("GEN4REC_SONG_EMB_PATH", Path(EMBEDDINGS_DIR) / "music4all_embeddings.npy")
    SONG_IDS_PATH = resolve_path("GEN4REC_SONG_IDS_PATH", Path(EMBEDDINGS_DIR) / "music4all_ids.npy")
    USER_EMB_PATH = resolve_path("GEN4REC_USER_EMB_PATH", Path(EMBEDDINGS_DIR) / "user_embeddings.npy")
    USER_IDS_PATH = resolve_path("GEN4REC_USER_IDS_PATH", Path(EMBEDDINGS_DIR) / "user_ids.npy")
    ID_INFORMATION_PATH = resolve_path(
        "GEN4REC_ID_INFORMATION_PATH",
        Path(DATASET_PATH) / "id_information.csv",
    )
    ID_METADATA_PATH = resolve_path(
        "GEN4REC_ID_METADATA_PATH",
        Path(DATASET_PATH) / "id_metadata.csv",
    )
    ID_GENRES_PATH = resolve_path("GEN4REC_ID_GENRES_PATH", Path(DATASET_PATH) / "id_genres.csv")
    ID_TAGS_PATH = resolve_path("GEN4REC_ID_TAGS_PATH", Path(DATASET_PATH) / "id_tags.csv")


def ensure_local_file(path: str, description: str) -> str:
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"{description} not found at {path}. "
        "Please download or copy this file and place it at that path, "
        "or override the default location with the corresponding GEN4REC_* environment variable."
    )


def load_listening_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) == 1 and "\t" in df.columns[0]:
        fixed = pd.read_csv(path, sep="\t", header=None)
        fixed.columns = ["user", "song", "timestamp"][: fixed.shape[1]]
        df = fixed
    lower_map = {c.lower(): c for c in df.columns}
    user_col = lower_map.get("user")
    song_col = lower_map.get("song")
    ts_col = lower_map.get("timestamp")
    if user_col is None or song_col is None:
        raise ValueError("listening_history.csv must contain 'user' and 'song' columns.")
    out = df.rename(columns={user_col: "user_id", song_col: "song_id"}).copy()
    if ts_col is not None:
        out = out.rename(columns={ts_col: "timestamp"})
    out["user_id"] = out["user_id"].astype(str)
    out["song_id"] = out["song_id"].astype(str)
    return out[["user_id", "song_id"]]


def _read_tsv_id_df(path: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        df = pd.read_csv(path, sep="\t", header=None, names=["id", value_col])
    df["id"] = df["id"].astype(str)
    return df


def load_id_information(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        df = pd.read_csv(path, sep="\t", header=None, names=["id", "artist", "song", "album_name"])
    df["id"] = df["id"].astype(str)
    return df


def load_id_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=[
                "id",
                "spotify_id",
                "popularity",
                "release",
                "danceability",
                "energy",
                "key",
                "mode",
                "valence",
                "tempo",
                "duration_ms",
            ],
        )
    df["id"] = df["id"].astype(str)
    return df


def jsonable_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if pd.isna(v):
        return None
    if isinstance(v, str):
        return v
    return str(v)


def build_export_payload(
    user_id: str,
    top_k: int,
    song_ids: np.ndarray,
    scores: np.ndarray,
    ranks: np.ndarray,
    info_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    genres_df: pd.DataFrame,
    tags_df: pd.DataFrame,
) -> dict[str, Any]:
    base = pd.DataFrame({"song_id": song_ids, "similarity_score": scores, "rank": ranks})
    info_df = info_df.rename(columns={"id": "song_id"})
    meta_df = meta_df.rename(columns={"id": "song_id"})
    genres_df = genres_df.rename(columns={"id": "song_id"})
    tags_df = tags_df.rename(columns={"id": "song_id"})

    merged = base.merge(info_df, on="song_id", how="left")
    merged = merged.merge(meta_df, on="song_id", how="left", suffixes=("", "_meta_dup"))
    merged = merged.merge(genres_df, on="song_id", how="left")
    merged = merged.merge(tags_df, on="song_id", how="left")

    # Drop duplicate id columns if any suffix collision
    drop_cols = [c for c in merged.columns if c.endswith("_meta_dup")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    songs_out: list[dict[str, Any]] = []
    info_key_order = ("artist", "song", "album_name")
    meta_key_order = (
        "spotify_id",
        "popularity",
        "release",
        "danceability",
        "energy",
        "key",
        "mode",
        "valence",
        "tempo",
        "duration_ms",
    )

    for _, row in merged.iterrows():
        info_block = {
            k: jsonable_value(row[k]) if k in row.index and pd.notna(row.get(k)) else None for k in info_key_order
        }
        meta_block = {k: jsonable_value(row[k]) if k in row.index else None for k in meta_key_order}
        entry: dict[str, Any] = {
            "rank": int(row["rank"]),
            "song_id": str(row["song_id"]),
            "similarity_score": float(row["similarity_score"]),
            "info": info_block,
            "metadata": meta_block,
            "genres": jsonable_value(row["genres"]) if "genres" in row else None,
            "tags": jsonable_value(row["tags"]) if "tags" in row else None,
        }
        songs_out.append(entry)

    return {
        "schema_version": "1.1",
        "user_id": user_id,
        "top_k": top_k,
        "retrieval": {
            "space": "clap_embedding_cosine",
            "note": "similarity_score is dot product on L2-normalized 512-d vectors (equals cosine similarity).",
        },
        "songs": songs_out,
    }


def export_user_profile_payload(
    *,
    user_id: str,
    top_k: int = 20,
    exclude_recent: bool = False,
) -> dict[str, Any]:
    song_embs = np.load(ensure_local_file(Config.SONG_EMB_PATH, "Song embedding matrix")).astype(np.float32)
    song_ids_arr = np.load(ensure_local_file(Config.SONG_IDS_PATH, "Song ID array"), allow_pickle=True).astype(str)
    user_embs = np.load(ensure_local_file(Config.USER_EMB_PATH, "User embedding matrix")).astype(np.float32)
    user_ids = np.load(ensure_local_file(Config.USER_IDS_PATH, "User ID array"), allow_pickle=True).astype(str)

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    if user_id not in user_to_idx:
        raise ValueError(f"user_id not found: {user_id}")
    user_vec = user_embs[user_to_idx[user_id]]

    scores = song_embs @ user_vec

    if exclude_recent:
        history = load_listening_history(ensure_local_file(Config.LISTENING_HISTORY_PATH, "Listening history table"))
        listened = set(history.loc[history["user_id"] == user_id, "song_id"].astype(str).tolist())
        if listened:
            song_to_idx = {sid: i for i, sid in enumerate(song_ids_arr)}
            listened_idxs = [song_to_idx[sid] for sid in listened if sid in song_to_idx]
            scores = scores.copy()
            scores[np.array(listened_idxs, dtype=np.int64)] = -1e9

    k = max(1, top_k)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    sel_scores = scores[idx].astype(np.float64)
    sel_song_ids = song_ids_arr[idx]
    ranks = np.arange(1, len(idx) + 1)

    info_df = load_id_information(ensure_local_file(Config.ID_INFORMATION_PATH, "Song information table"))
    meta_df = load_id_metadata(ensure_local_file(Config.ID_METADATA_PATH, "Song metadata table"))
    genres_df = _read_tsv_id_df(ensure_local_file(Config.ID_GENRES_PATH, "Song genre table"), "genres")
    tags_df = _read_tsv_id_df(ensure_local_file(Config.ID_TAGS_PATH, "Song tag table"), "tags")

    return build_export_payload(
        user_id=user_id,
        top_k=k,
        song_ids=sel_song_ids,
        scores=sel_scores,
        ranks=ranks,
        info_df=info_df,
        meta_df=meta_df,
        genres_df=genres_df,
        tags_df=tags_df,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Top-K nearest songs as JSON (info + metadata + genres/tags) for LLM downstream."
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument(
        "--top-k",
        "--top-m",
        type=int,
        default=20,
        dest="top_k",
        help="Number of nearest songs to retrieve (Top-K). Alias --top-m is deprecated but still works.",
    )
    parser.add_argument("--exclude-recent", action="store_true", help="Exclude songs already in user listening history.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write JSON to this path; default prints to stdout only.",
    )
    args = parser.parse_args()

    payload = export_user_profile_payload(
        user_id=args.user_id,
        top_k=args.top_k,
        exclude_recent=args.exclude_recent,
    )

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
