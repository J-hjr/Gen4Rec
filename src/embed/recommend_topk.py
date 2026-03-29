import argparse
import os
from pathlib import Path

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
    else:
        out["timestamp"] = np.nan
    out["user_id"] = out["user_id"].astype(str)
    out["song_id"] = out["song_id"].astype(str)
    return out[["user_id", "song_id", "timestamp"]]


def load_song_metadata(path: str) -> pd.DataFrame:
    """Load tab-separated id_information: id, artist, song, album_name."""
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns and len(df.columns) > 0:
        df = pd.read_csv(path, sep="\t", header=None, names=["id", "artist", "song", "album_name"])
    df["id"] = df["id"].astype(str)
    return df


def load_audio_metadata(path: str) -> pd.DataFrame:
    """
    Load tab-separated id_metadata: Spotify id, popularity, release year,
    and audio features (danceability, energy, key, mode, valence, tempo, duration_ms).
    """
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns and len(df.columns) > 0:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend top-k songs for a user from embedding space.")
    parser.add_argument("--user-id", required=True, help="Target user ID, e.g. user_007XIjOr")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--exclude-recent", action="store_true", help="Exclude songs the user has already listened to")
    parser.add_argument(
        "--with-info",
        action="store_true",
        help="Join id_information.csv for artist, song title, and album_name",
    )
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help="Join id_metadata.csv for Spotify/audio features (popularity, tempo, etc.)",
    )
    args = parser.parse_args()

    song_embs = np.load(Config.SONG_EMB_PATH).astype(np.float32)
    # IDs may be stored as object arrays depending on how npy was created.
    song_ids = np.load(Config.SONG_IDS_PATH, allow_pickle=True).astype(str)
    user_embs = np.load(Config.USER_EMB_PATH).astype(np.float32)
    user_ids = np.load(Config.USER_IDS_PATH, allow_pickle=True).astype(str)

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    if args.user_id not in user_to_idx:
        raise ValueError(f"user_id not found: {args.user_id}")
    user_vec = user_embs[user_to_idx[args.user_id]]

    # Dot product equals cosine similarity here because vectors are normalized.
    scores = song_embs @ user_vec

    if args.exclude_recent:
        history = load_listening_history(Config.LISTENING_HISTORY_PATH)
        listened = set(history.loc[history["user_id"] == args.user_id, "song_id"].astype(str).tolist())
        if listened:
            song_to_idx = {sid: i for i, sid in enumerate(song_ids)}
            listened_idxs = [song_to_idx[sid] for sid in listened if sid in song_to_idx]
            scores[np.array(listened_idxs, dtype=np.int64)] = -1e9

    top_k = max(1, args.top_k)
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]

    result = pd.DataFrame(
        {
            "rank": np.arange(1, len(idx) + 1),
            "song_id": song_ids[idx],
            "score": scores[idx],
        }
    )

    if args.with_info:
        info_df = load_song_metadata(Config.ID_INFORMATION_PATH)
        result = result.merge(info_df, left_on="song_id", right_on="id", how="left").drop(columns=["id"])

    if args.with_metadata:
        meta_df = load_audio_metadata(Config.ID_METADATA_PATH)
        result = result.merge(meta_df, left_on="song_id", right_on="id", how="left").drop(columns=["id"])

    if args.with_info or args.with_metadata:
        # Stable readable order: rank, ids, human-readable info, audio features, score last.
        preferred = [
            "rank",
            "song_id",
            "artist",
            "song",
            "album_name",
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
            "score",
        ]
        cols = [c for c in preferred if c in result.columns]
        extra = [c for c in result.columns if c not in cols]
        result = result[cols + extra]

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

