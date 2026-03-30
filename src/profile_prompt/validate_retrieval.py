'''这段是检查我们的embedding, 对指定 user 构建历史听歌集合, 重新在 embedding 空间里做 retrieval
再对比: history top genres/tags & retrieved top genres/tags
        history vs retrieved 的 audio feature 均值
输出验证指标: genre overlap; tag overlap; audio feature difference; retrieval coherence summary'''

import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# Paths (default project layout)
EMBED_DIR = Path("outputs/embeddings/music4all")
DATA_DIR = Path("music4all")


# Helpers
def split_csv_like_field(value: Any) -> List[str]:
    if pd.isna(value) or value is None:
        return []
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def top_items(counter: Counter, k: int = 10) -> List[str]:
    return [x for x, _ in counter.most_common(k)]


def safe_mean(series: pd.Series) -> float | None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return None
    return round(float(series.mean()), 3)


def jaccard_overlap(a: List[str], b: List[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return round(len(set_a & set_b) / len(set_a | set_b), 3)


def cosine_topk(
    user_vec: np.ndarray,
    song_matrix: np.ndarray,
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    # Assume embeddings are already L2-normalized.
    scores = song_matrix @ user_vec
    top_idx = np.argsort(-scores)[:top_k]
    return top_idx, scores[top_idx]


# ----------------------------
# Loaders
# ----------------------------
def load_embeddings() -> Dict[str, Any]:
    return {
        "song_ids": np.load(EMBED_DIR / "music4all_ids.npy", allow_pickle=True),
        "song_embeds": np.load(EMBED_DIR / "music4all_embeddings.npy", allow_pickle=True),
        "user_ids": np.load(EMBED_DIR / "user_ids.npy", allow_pickle=True),
        "user_embeds": np.load(EMBED_DIR / "user_embeddings.npy", allow_pickle=True),
    }


def load_tables() -> Dict[str, pd.DataFrame]:
    id_genres = pd.read_csv(DATA_DIR / "id_genres.csv", sep=None, engine="python")
    id_tags = pd.read_csv(DATA_DIR / "id_tags.csv", sep=None, engine="python")
    id_metadata = pd.read_csv(DATA_DIR / "id_metadata.csv", sep=None, engine="python")
    listening_history = pd.read_csv(DATA_DIR / "listening_history.csv", sep=None, engine="python")
    return {
        "id_genres": id_genres,
        "id_tags": id_tags,
        "id_metadata": id_metadata,
        "listening_history": listening_history,
    }


# Column detection
def detect_column(df: pd.DataFrame, candidates: List[str], df_name: str) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise ValueError(
        f"Could not detect expected column in {df_name}. "
        f"Available columns: {list(df.columns)}. "
        f"Tried: {candidates}"
    )


def prepare_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    id_genres = tables["id_genres"].copy()
    id_tags = tables["id_tags"].copy()
    id_metadata = tables["id_metadata"].copy()
    listening_history = tables["listening_history"].copy()

    song_id_col_genres = detect_column(id_genres, ["id", "song_id", "track_id"], "id_genres")
    genre_col = detect_column(id_genres, ["genres", "genre"], "id_genres")

    song_id_col_tags = detect_column(id_tags, ["id", "song_id", "track_id"], "id_tags")
    tags_col = detect_column(id_tags, ["tags", "tag"], "id_tags")

    song_id_col_meta = detect_column(id_metadata, ["id", "song_id", "track_id"], "id_metadata")

    user_col_hist = detect_column(listening_history, ["user_id", "userid", "user"], "listening_history")
    song_col_hist = detect_column(listening_history, ["song", "id", "song_id", "track_id"], "listening_history")

    # Try to detect a timestamp / order column for recency logic if available
    history_sort_col = None
    for cand in ["timestamp", "ts", "time", "played_at", "datetime", "date"]:
        if cand in [c.lower() for c in listening_history.columns]:
            history_sort_col = detect_column(listening_history, [cand], "listening_history")
            break

    genre_map = dict(zip(id_genres[song_id_col_genres].astype(str), id_genres[genre_col]))
    tag_map = dict(zip(id_tags[song_id_col_tags].astype(str), id_tags[tags_col]))

    id_metadata[song_id_col_meta] = id_metadata[song_id_col_meta].astype(str)

    return {
        "genre_map": genre_map,
        "tag_map": tag_map,
        "id_metadata": id_metadata,
        "song_id_col_meta": song_id_col_meta,
        "listening_history": listening_history,
        "user_col_hist": user_col_hist,
        "song_col_hist": song_col_hist,
        "history_sort_col": history_sort_col,
    }


# Feature extraction
def aggregate_genres_tags(song_ids: List[str], genre_map: Dict[str, Any], tag_map: Dict[str, Any]) -> Dict[str, Any]:
    genre_counter = Counter()
    tag_counter = Counter()

    for sid in song_ids:
        for g in split_csv_like_field(genre_map.get(str(sid))):
            genre_counter[g] += 1
        for t in split_csv_like_field(tag_map.get(str(sid))):
            tag_counter[t] += 1

    return {
        "top_genres": top_items(genre_counter, 10),
        "top_tags": top_items(tag_counter, 12),
        "genre_counter": genre_counter,
        "tag_counter": tag_counter,
    }


def aggregate_audio(song_ids: List[str], id_metadata: pd.DataFrame, song_id_col_meta: str) -> Dict[str, Any]:
    subset = id_metadata[id_metadata[song_id_col_meta].isin([str(x) for x in song_ids])].copy()

    return {
        "danceability_mean": safe_mean(subset["danceability"]) if "danceability" in subset.columns else None,
        "energy_mean": safe_mean(subset["energy"]) if "energy" in subset.columns else None,
        "valence_mean": safe_mean(subset["valence"]) if "valence" in subset.columns else None,
        "tempo_mean": safe_mean(subset["tempo"]) if "tempo" in subset.columns else None,
    }


def compute_audio_deltas(hist_audio: Dict[str, Any], ret_audio: Dict[str, Any]) -> Dict[str, Any]:
    deltas = {}
    for key in ["danceability_mean", "energy_mean", "valence_mean", "tempo_mean"]:
        h = hist_audio.get(key)
        r = ret_audio.get(key)
        if h is None or r is None:
            deltas[key.replace("_mean", "_delta")] = None
        else:
            deltas[key.replace("_mean", "_delta")] = round(abs(h - r), 3)
    return deltas


def generate_validation_summary(
    genre_overlap: float,
    tag_overlap: float,
    audio_deltas: Dict[str, Any],
) -> str:
    score_notes = []

    if genre_overlap >= 0.4:
        score_notes.append("strong genre overlap")
    elif genre_overlap >= 0.2:
        score_notes.append("moderate genre overlap")
    else:
        score_notes.append("limited genre overlap")

    if tag_overlap >= 0.3:
        score_notes.append("good tag consistency")
    elif tag_overlap >= 0.15:
        score_notes.append("some tag consistency")
    else:
        score_notes.append("weak tag consistency")

    energy_delta = audio_deltas.get("energy_delta")
    valence_delta = audio_deltas.get("valence_delta")
    tempo_delta = audio_deltas.get("tempo_delta")

    if energy_delta is not None and valence_delta is not None:
        if energy_delta <= 0.12 and valence_delta <= 0.12:
            score_notes.append("audio mood is closely aligned")
        elif energy_delta <= 0.2 and valence_delta <= 0.2:
            score_notes.append("audio mood is reasonably aligned")
        else:
            score_notes.append("audio mood differs noticeably")

    if tempo_delta is not None:
        if tempo_delta <= 15:
            score_notes.append("tempo profile is similar")
        elif tempo_delta <= 30:
            score_notes.append("tempo profile is somewhat similar")
        else:
            score_notes.append("tempo profile differs")

    return "; ".join(score_notes)


# Main validation logic
def get_recent_history_song_ids(
    listening_history: pd.DataFrame,
    user_id: str,
    user_col: str,
    song_col: str,
    sort_col: str | None,
    recent_k: int = 20,
) -> List[str]:
    df = listening_history[listening_history[user_col].astype(str) == str(user_id)].copy()

    if sort_col is not None:
        df = df.sort_values(sort_col, ascending=False)

    song_ids = df[song_col].astype(str).tolist()

    # keep order, deduplicate
    seen = set()
    deduped = []
    for sid in song_ids:
        if sid not in seen:
            deduped.append(sid)
            seen.add(sid)

    return deduped[:recent_k]


def validate_retrieval_for_user(
    user_id: str,
    top_k: int = 20,
    recent_k: int = 20,
    exclude_recent: bool = True,
) -> Dict[str, Any]:
    embeds = load_embeddings()
    tables_raw = load_tables()
    tables = prepare_tables(tables_raw)

    user_ids = embeds["user_ids"].astype(str)
    user_embeds = embeds["user_embeds"]
    song_ids = embeds["song_ids"].astype(str)
    song_embeds = embeds["song_embeds"]

    if str(user_id) not in set(user_ids):
        raise ValueError(f"user_id {user_id} not found in user_ids.npy")

    user_idx = np.where(user_ids == str(user_id))[0][0]
    user_vec = user_embeds[user_idx]

    history_song_ids = get_recent_history_song_ids(
        listening_history=tables["listening_history"],
        user_id=user_id,
        user_col=tables["user_col_hist"],
        song_col=tables["song_col_hist"],
        sort_col=tables["history_sort_col"],
        recent_k=recent_k,
    )

    candidate_song_ids = song_ids.copy()
    candidate_embeds = song_embeds.copy()

    if exclude_recent:
        mask = ~np.isin(candidate_song_ids, np.array(history_song_ids))
        candidate_song_ids = candidate_song_ids[mask]
        candidate_embeds = candidate_embeds[mask]

    top_idx, top_scores = cosine_topk(user_vec, candidate_embeds, top_k=top_k)
    retrieved_song_ids = candidate_song_ids[top_idx].tolist()

    history_gt = aggregate_genres_tags(history_song_ids, tables["genre_map"], tables["tag_map"])
    retrieved_gt = aggregate_genres_tags(retrieved_song_ids, tables["genre_map"], tables["tag_map"])

    history_audio = aggregate_audio(history_song_ids, tables["id_metadata"], tables["song_id_col_meta"])
    retrieved_audio = aggregate_audio(retrieved_song_ids, tables["id_metadata"], tables["song_id_col_meta"])

    genre_overlap = jaccard_overlap(history_gt["top_genres"][:5], retrieved_gt["top_genres"][:5])
    tag_overlap = jaccard_overlap(history_gt["top_tags"][:8], retrieved_gt["top_tags"][:8])
    audio_deltas = compute_audio_deltas(history_audio, retrieved_audio)

    validation_summary = generate_validation_summary(
        genre_overlap=genre_overlap,
        tag_overlap=tag_overlap,
        audio_deltas=audio_deltas,
    )

    retrieved_examples = [
        {"song_id": sid, "score": round(float(score), 6)}
        for sid, score in zip(retrieved_song_ids[:10], top_scores[:10])
    ]

    return {
        "user_id": user_id,
        "settings": {
            "recent_k": recent_k,
            "top_k": top_k,
            "exclude_recent": exclude_recent,
        },
        "history_summary": {
            "num_recent_songs_used": len(history_song_ids),
            "recent_song_ids": history_song_ids,
            "top_genres": history_gt["top_genres"],
            "top_tags": history_gt["top_tags"],
            "audio_profile": history_audio,
        },
        "retrieval_summary": {
            "num_retrieved": len(retrieved_song_ids),
            "retrieved_song_ids": retrieved_song_ids,
            "top_genres": retrieved_gt["top_genres"],
            "top_tags": retrieved_gt["top_tags"],
            "audio_profile": retrieved_audio,
            "top_examples": retrieved_examples,
        },
        "validation_metrics": {
            "genre_overlap_top5_jaccard": genre_overlap,
            "tag_overlap_top8_jaccard": tag_overlap,
            **audio_deltas,
        },
        "human_readable_summary": validation_summary,
    }


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate retrieval quality for a given user.")
    parser.add_argument("--user-id", type=str, required=True, help="User ID to validate.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of retrieved songs.")
    parser.add_argument("--recent-k", type=int, default=20, help="Number of recent history songs to summarize.")
    parser.add_argument(
        "--exclude-recent",
        action="store_true",
        help="Exclude already listened songs from retrieval."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save validation JSON."
    )

    args = parser.parse_args()

    result = validate_retrieval_for_user(
        user_id=args.user_id,
        top_k=args.top_k,
        recent_k=args.recent_k,
        exclude_recent=args.exclude_recent,
    )

    print("\nRetrieval validation result:\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        save_json(result, args.output)
        print(f"\nSaved validation result to: {args.output}")