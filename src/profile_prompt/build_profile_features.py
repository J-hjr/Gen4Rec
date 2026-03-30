import json
from pathlib import Path
from collections import Counter
from statistics import mean
from typing import Any, Dict, List


def _split_csv_like_field(value: Any) -> List[str]:
    """Split comma-separated string fields like genres/tags into cleaned tokens."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")

    cleaned = []
    for item in items:
        token = str(item).strip().lower()
        if token:
            cleaned.append(token)
    return cleaned


def _safe_mean(values: List[float]) -> float | None:
    """Return mean if list is non-empty, else None."""
    return round(mean(values), 3) if values else None


def _top_items(counter: Counter, k: int = 8) -> List[str]:
    """Return top-k item names only."""
    return [item for item, _ in counter.most_common(k)]


def _build_mood_summary(
    energy_mean: float | None,
    valence_mean: float | None,
    tempo_mean: float | None,
) -> List[str]:
    """Map audio features to a few interpretable mood/style descriptors."""
    moods: List[str] = []

    if energy_mean is not None:
        if energy_mean < 0.3:
            moods.extend(["gentle", "understated"])
        elif energy_mean < 0.55:
            moods.extend(["soft", "balanced"])
        else:
            moods.extend(["energetic", "driving"])

    if valence_mean is not None:
        if valence_mean < 0.25:
            moods.extend(["melancholic", "introspective"])
        elif valence_mean < 0.5:
            moods.extend(["bittersweet", "reflective"])
        else:
            moods.extend(["warm", "uplifting"])

    if tempo_mean is not None:
        if tempo_mean < 90:
            moods.append("slow-paced")
        elif tempo_mean < 125:
            moods.append("mid-tempo")
        else:
            moods.append("flowing")

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for mood in moods:
        if mood not in seen:
            deduped.append(mood)
            seen.add(mood)
    return deduped[:6]


def _build_rule_based_profile(summary: Dict[str, Any]) -> str:
    """Create a readable paragraph before involving an LLM."""
    top_genres = summary.get("top_genres", [])[:4]
    top_tags = summary.get("top_tags", [])[:5]
    moods = summary.get("mood_summary", [])[:4]
    artists = summary.get("representative_artists", [])[:4]

    genre_text = ", ".join(top_genres) if top_genres else "indie-oriented music"
    tag_text = ", ".join(top_tags) if top_tags else "emotionally nuanced textures"
    mood_text = ", ".join(moods) if moods else "reflective and intimate moods"
    artist_text = ", ".join(artists) if artists else "similar singer-songwriters"

    return (
        f"This listener gravitates toward {genre_text}, with recurring traits such as "
        f"{tag_text}. Their retrieved songs suggest a preference for {mood_text}, often "
        f"centered on expressive vocals and restrained arrangements. Representative artists "
        f"include {artist_text}, pointing to a taste for emotionally detailed, intimate music."
    )


def build_profile_features(profile_json: Dict[str, Any], top_n: int = 20) -> Dict[str, Any]:
    """
    Convert retrieval JSON into a compact, LLM-friendly summary.
    """
    songs = profile_json.get("songs", [])[:top_n]

    genre_counter: Counter = Counter()
    tag_counter: Counter = Counter()
    artist_counter: Counter = Counter()

    danceability_vals: List[float] = []
    energy_vals: List[float] = []
    valence_vals: List[float] = []
    tempo_vals: List[float] = []

    representative_tracks: List[Dict[str, str]] = []

    for song in songs:
        info = song.get("info", {})
        metadata = song.get("metadata", {})

        artist = info.get("artist")
        title = info.get("song")

        if artist:
            artist_counter[str(artist).strip()] += 1

        for genre in _split_csv_like_field(song.get("genres")):
            genre_counter[genre] += 1

        for tag in _split_csv_like_field(song.get("tags")):
            tag_counter[tag] += 1

        if artist and title and len(representative_tracks) < 5:
            representative_tracks.append({
                "artist": str(artist).strip(),
                "song": str(title).strip()
            })

        if metadata.get("danceability") is not None:
            danceability_vals.append(float(metadata["danceability"]))
        if metadata.get("energy") is not None:
            energy_vals.append(float(metadata["energy"]))
        if metadata.get("valence") is not None:
            valence_vals.append(float(metadata["valence"]))
        if metadata.get("tempo") is not None:
            tempo_vals.append(float(metadata["tempo"]))

    audio_profile = {
        "danceability_mean": _safe_mean(danceability_vals),
        "energy_mean": _safe_mean(energy_vals),
        "valence_mean": _safe_mean(valence_vals),
        "tempo_mean": _safe_mean(tempo_vals),
    }

    summary = {
        "user_id": profile_json.get("user_id"),
        "source_top_k": profile_json.get("top_k"),
        "top_genres": _top_items(genre_counter, 8),
        "top_tags": _top_items(tag_counter, 10),
        "representative_artists": _top_items(artist_counter, 6),
        "representative_tracks": representative_tracks,
        "audio_profile": audio_profile,
        "mood_summary": _build_mood_summary(
            energy_mean=audio_profile["energy_mean"],
            valence_mean=audio_profile["valence_mean"],
            tempo_mean=audio_profile["tempo_mean"],
        ),
    }

    summary["rule_based_profile_paragraph"] = _build_rule_based_profile(summary)
    return summary


def load_profile_json(json_path: str | Path) -> Dict[str, Any]:
    """Load exported retrieval JSON."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_summary(summary: Dict[str, Any], output_path: str | Path) -> None:
    """Save condensed summary to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build compact profile features from retrieval JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to exported user profile JSON."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save condensed feature summary JSON."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many retrieved songs to summarize."
    )

    args = parser.parse_args()

    raw_profile = load_profile_json(args.input)
    summary = build_profile_features(raw_profile, top_n=args.top_n)
    save_summary(summary, args.output)

    print("\nBuilt condensed profile summary.\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))