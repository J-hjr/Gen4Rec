"""
Build user embedding vectors from recent listening history.

This module converts per-user listening events into dense user representations
that can be used for nearest-neighbor retrieval against song embeddings.

Inputs
- `listening_history.csv`: user-song interaction history, optionally with
  timestamps for recency ordering.
- `music4all_embeddings.npy`: normalized song embedding matrix with one row per
  song.
- `music4all_ids.npy` or `id_genres.csv`: mapping from song IDs to embedding
  rows.

Processing overview
1. Load and normalize listening-history columns into `user_id`, `song_id`, and
   `timestamp`.
2. For each user, keep only the most recent `K` events.
3. Collapse repeated songs, preserving first recency rank and play count.
4. Remove incoherent outliers using a medoid-based similarity filter so a
   single off-topic listen does not dominate the user profile.
5. Compute a weighted average of the remaining song embeddings using:
   - exponential recency decay
   - logarithmic frequency boost for repeated listens
6. L2-normalize the final vector so cosine similarity can be used directly for
   retrieval.

Outputs
- `user_embeddings.npy`: row-aligned matrix of user embedding vectors.
- `user_ids.npy`: user IDs corresponding to `user_embeddings.npy`.
- `user_embedding_stats.csv`: per-user diagnostics for recent events and songs
  kept after coherence filtering.

Operational notes
- If `music4all_ids.npy` is missing, the script can recreate it from
  `id_genres.csv`, but the row order must match `music4all_embeddings.npy`.
- Environment variables such as `GEN4REC_DATASET_PATH` and
  `GEN4REC_EMBED_OUTPUT_DIR` can override the default repo-relative paths.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


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
    ID_GENRES_PATH = resolve_path("GEN4REC_ID_GENRES_PATH", Path(DATASET_PATH) / "id_genres.csv")
    SONG_EMB_PATH = resolve_path(
        "GEN4REC_SONG_EMB_PATH",
        Path(EMBEDDINGS_DIR) / "music4all_embeddings.npy",
    )
    SONG_IDS_PATH = resolve_path(
        "GEN4REC_SONG_IDS_PATH",
        Path(EMBEDDINGS_DIR) / "music4all_ids.npy",
    )


def load_listening_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    if len(df.columns) == 1 and "\t" in df.columns[0]:
        # Handles malformed header where pandas saw one literal "user\tsong\ttimestamp".
        fixed = pd.read_csv(path, sep="\t", header=None)
        fixed.columns = ["user", "song", "timestamp"][: fixed.shape[1]]
        df = fixed

    lower_map = {c.lower(): c for c in df.columns}
    user_col = lower_map.get("user")
    song_col = lower_map.get("song")
    ts_col = lower_map.get("timestamp")

    if user_col is None or song_col is None:
        raise ValueError(
            "listening_history.csv must contain 'user' and 'song' columns."
        )

    out = df.rename(columns={user_col: "user_id", song_col: "song_id"}).copy()
    if ts_col is not None:
        out = out.rename(columns={ts_col: "timestamp"})
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    else:
        out["timestamp"] = pd.NaT

    out["user_id"] = out["user_id"].astype(str)
    out["song_id"] = out["song_id"].astype(str)
    return out[["user_id", "song_id", "timestamp"]]


def ensure_song_ids(song_ids_path: str, id_genres_path: str, expected_n: int) -> np.ndarray:
    if os.path.exists(song_ids_path):
        song_ids = np.load(song_ids_path)
        if len(song_ids) != expected_n:
            raise ValueError(
                f"music4all_ids.npy length mismatch: {len(song_ids)} vs embeddings rows {expected_n}."
            )
        return song_ids.astype(str)

    # Support both header and no-header variants of id_genres.csv.
    id_df = pd.read_csv(id_genres_path, sep="\t")
    if "id" in id_df.columns:
        song_ids = id_df["id"].astype(str).to_numpy()
    else:
        id_df = pd.read_csv(
            id_genres_path,
            sep="\t",
            header=None,
            names=["id", "genres"],
            usecols=[0],
        )
        song_ids = id_df["id"].astype(str).to_numpy()
    if len(song_ids) != expected_n:
        raise ValueError(
            "Cannot auto-create music4all_ids.npy: id count does not match embeddings rows."
        )
    np.save(song_ids_path, song_ids)
    print(f"Saved missing ids file to {song_ids_path}")
    return song_ids


def build_user_embeddings(
    listening_df: pd.DataFrame,
    song_ids: np.ndarray,
    song_embs: np.ndarray,
    recent_k: int,
    decay_lambda: float,
    medoid_threshold: float,
    min_keep: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # Build a fast lookup from song_id -> row index in the embedding matrix.
    # This avoids repeated joins and keeps per-user processing cheap.
    song2idx = {sid: i for i, sid in enumerate(song_ids)}

    available = listening_df["song_id"].isin(song2idx)
    filtered = listening_df.loc[available].copy()
    if filtered.empty:
        raise ValueError("No listening records matched song embedding IDs.")

    # Sort by recency so Recent-K is literally the top K rows per user.
    # If a timestamp is missing (NaT), we treat it as least recent.
    filtered = filtered.sort_values(["user_id", "timestamp"], ascending=[True, False], na_position="last")

    user_ids_out = []
    user_embs_out = []
    stats = []

    for user_id, user_df in tqdm(filtered.groupby("user_id", sort=False), desc="Building user embeddings"):
        recent = user_df.head(recent_k).copy()
        if recent.empty:
            continue

        # Rank 0 means most recent interaction. We later convert rank to "age"
        # for exponential decay, where larger age should get lower weight.
        recent["rank"] = np.arange(len(recent), dtype=np.int32)
        agg = (
            recent.groupby("song_id", as_index=False)
            .agg(min_rank=("rank", "min"), play_count=("song_id", "count"))
            .sort_values("min_rank")
        )

        idxs = np.array([song2idx[sid] for sid in agg["song_id"].tolist()], dtype=np.int64)
        embs = song_embs[idxs].astype(np.float32)

        # Medoid-based coherence filtering:
        # We find the song that is most central to the Recent-K set (highest
        # average cosine similarity to others). Songs far from this medoid are
        # treated as short-term outliers and removed before averaging.
        #
        # Why: without this step, very mixed histories can pull the user vector
        # toward an "in-between" point that does not represent a real taste mode.
        if embs.shape[0] > 1:
            sims = embs @ embs.T
            mean_sims = sims.mean(axis=1)
            medoid_idx = int(np.argmax(mean_sims))
            keep_mask = sims[medoid_idx] >= medoid_threshold
            if keep_mask.sum() < min_keep:
                # Safety fallback: always keep at least `min_keep` nearest songs
                # to preserve enough signal for stable user representation.
                top_idx = np.argsort(sims[medoid_idx])[::-1][: min(min_keep, embs.shape[0])]
                keep_mask = np.zeros(embs.shape[0], dtype=bool)
                keep_mask[top_idx] = True
            embs = embs[keep_mask]
            agg = agg.iloc[np.where(keep_mask)[0]].copy()

        if embs.shape[0] == 0:
            continue

        ages = agg["min_rank"].to_numpy(dtype=np.float32)
        counts = agg["play_count"].to_numpy(dtype=np.float32)

        # Weight design:
        # 1) Time decay: w_time = exp(-lambda * age), so recent songs dominate.
        # 2) Frequency boost: w_freq = 1 + log(1 + play_count), so repeated listens
        #    matter more, but logarithm prevents a single looped song from dominating.
        # Final weight is multiplicative and then normalized to sum to 1.
        time_w = np.exp(-decay_lambda * ages)
        freq_w = 1.0 + np.log1p(counts)
        weights = time_w * freq_w
        weights = weights / (weights.sum() + 1e-12)

        # Weighted average in song-embedding space, then L2-normalize so cosine
        # similarity remains the primary distance metric for retrieval.
        user_vec = (embs * weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(user_vec))
        if norm > 0:
            user_vec = user_vec / norm

        user_ids_out.append(user_id)
        user_embs_out.append(user_vec.astype(np.float32))
        stats.append(
            {
                "user_id": user_id,
                "recent_events": int(len(recent)),
                "unique_recent_songs": int(len(idxs)),
                "kept_songs_after_coherence": int(len(embs)),
            }
        )

    if not user_embs_out:
        raise ValueError("No user embeddings were generated.")

    return np.array(user_ids_out, dtype=str), np.vstack(user_embs_out), pd.DataFrame(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build user weighted embeddings from listening history.")
    # Recent-K tradeoff:
    # - Smaller K (e.g., 5-10): more responsive to short-term taste shifts,
    #   but more sensitive to noise/outlier listens.
    # - Larger K (e.g., 50): more stable profile, but slower to adapt.
    # In practice, K=20 is often a good balance for short-term recommendation.
    parser.add_argument("--recent-k", type=int, default=50)
    parser.add_argument("--decay-lambda", type=float, default=0.08)
    parser.add_argument("--medoid-threshold", type=float, default=0.2)
    parser.add_argument("--min-keep", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)

    print(f"Loading song embeddings from {Config.SONG_EMB_PATH}")
    song_embs = np.load(Config.SONG_EMB_PATH).astype(np.float32)
    print(f"Song embeddings shape: {song_embs.shape}")

    song_ids = ensure_song_ids(Config.SONG_IDS_PATH, Config.ID_GENRES_PATH, expected_n=song_embs.shape[0])
    print(f"Song ids count: {len(song_ids)}")

    print(f"Loading listening history from {Config.LISTENING_HISTORY_PATH}")
    listening_df = load_listening_history(Config.LISTENING_HISTORY_PATH)
    print(f"Listening events: {len(listening_df)}")

    user_ids, user_embs, stats_df = build_user_embeddings(
        listening_df=listening_df,
        song_ids=song_ids,
        song_embs=song_embs,
        recent_k=args.recent_k,
        decay_lambda=args.decay_lambda,
        medoid_threshold=args.medoid_threshold,
        min_keep=args.min_keep,
    )

    user_emb_path = Path(Config.EMBEDDINGS_DIR) / "user_embeddings.npy"
    user_ids_path = Path(Config.EMBEDDINGS_DIR) / "user_ids.npy"
    stats_path = Path(Config.EMBEDDINGS_DIR) / "user_embedding_stats.csv"

    np.save(user_emb_path, user_embs)
    np.save(user_ids_path, user_ids)
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved user embeddings to {user_emb_path}")
    print(f"Saved user ids to {user_ids_path}")
    print(f"Saved user stats to {stats_path}")
    print(f"User embeddings shape: {user_embs.shape}")


if __name__ == "__main__":
    main()

