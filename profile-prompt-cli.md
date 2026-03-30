# Profile prompt CLI

## Required directories & files (defaults)

All paths are relative to the repo root.

### What this stage does

This stage turns retrieval output into:

- a compact listener summary for downstream prompt engineering
- a human-readable user profile paragraph
- a text-to-music prompt draft for later generation
- an optional retrieval validation report for debugging and analysis

### Upstream input

The profile-prompt stage expects a retrieval JSON produced by:

```bash
python src/embed/export_user_profile_json.py --user-id <USER_ID> --top-k 20 -o outputs/profiles/<USER_ID>.json
```

That JSON contains the user ID, retrieved songs, similarity scores, and joined metadata such as genres, tags, artist/song names, and audio features.

### Environment variables

`generate_user_profile_and_prompt.py` calls the OpenAI API and expects:

```bash
export OPENAI_API_KEY="..."
```

You can place this in `.env` locally if you use `python-dotenv`.

---

## Condensed summary (`build_profile_features.py`)

```bash
python src/profile_prompt/build_profile_features.py --input outputs/profiles/<USER_ID>.json --output outputs/profiles/<USER_ID>_summary.json
python src/profile_prompt/build_profile_features.py --input outputs/profiles/<USER_ID>.json --output outputs/profiles/<USER_ID>_summary.json --top-n 10
```

What it does:

- reads the retrieval JSON
- aggregates top genres, top tags, representative artists, and representative tracks
- computes average audio traits such as danceability, energy, valence, and tempo
- derives a compact mood summary
- writes a rule-based profile paragraph before any LLM step

| Parameter | Meaning |
|-----------|---------|
| `--input` | **Required.** Path to the retrieval JSON exported from `src/embed/export_user_profile_json.py`. |
| `--output` | **Required.** Path to save the condensed summary JSON. |
| `--top-n` | How many retrieved songs to summarize (default `20`). Smaller values focus on the nearest songs only. |

Output JSON includes fields such as `top_genres`, `top_tags`, `representative_artists`, `representative_tracks`, `audio_profile`, `mood_summary`, and `rule_based_profile_paragraph`.

---

## LLM profile and prompt (`generate_user_profile_and_prompt.py`)

```bash
python src/profile_prompt/generate_user_profile_and_prompt.py --input outputs/profiles/<USER_ID>_summary.json --output outputs/profiles/<USER_ID>_prompt.json
python src/profile_prompt/generate_user_profile_and_prompt.py --input outputs/profiles/<USER_ID>_summary.json --output outputs/profiles/<USER_ID>_prompt.json --model gpt-4.1-mini
```

What it does:

- reads the condensed summary JSON
- calls the OpenAI API
- generates a polished listener profile paragraph
- generates a compact music-generation prompt
- returns a list of style keywords

| Parameter | Meaning |
|-----------|---------|
| `--input` | **Required.** Path to the condensed summary JSON. |
| `--output` | **Required.** Path to save the generated profile-and-prompt JSON. |
| `--model` | OpenAI model name for the prompt-engineering step (default `gpt-4.1-mini`). |

Output JSON includes:

- `user_id`
- `input_summary`
- `profile_paragraph`
- `suno_generation_prompt`
- `style_keywords`

Even though the current field name is `suno_generation_prompt`, you can also treat it as a first-pass music-generation prompt for later adaptation to other providers.

---

## Retrieval validation (`validate_retrieval.py`)

```bash
python src/profile_prompt/validate_retrieval.py --user-id <USER_ID>
python src/profile_prompt/validate_retrieval.py --user-id <USER_ID> --top-k 20 --recent-k 20 --exclude-recent --output outputs/profiles/<USER_ID>_validation.json
```

What it does:

- rebuilds retrieval results in embedding space for the specified user
- compares the user's recent listening history with the retrieved songs
- checks genre overlap, tag overlap, and audio-feature similarity
- writes a validation JSON for analysis

| Parameter | Meaning |
|-----------|---------|
| `--user-id` | **Required.** User ID to validate. |
| `--top-k` | Number of retrieved songs to compare against history (default `20`). |
| `--recent-k` | Number of recent history songs to use as the comparison set (default `20`). |
| `--exclude-recent` | Exclude already listened songs from retrieval before validation. |
| `--output` | Optional path to save the validation JSON. If omitted, the result is printed only. |

Validation output includes:

- `history_summary`
- `retrieval_summary`
- `validation_metrics`
- `human_readable_summary`

This script is mainly for quality checking and debugging, not for the main generation pipeline.

---

## Suggested order

```bash
python src/embed/export_user_profile_json.py --user-id <USER_ID> --top-k 20 --exclude-recent -o outputs/profiles/<USER_ID>.json
python src/profile_prompt/build_profile_features.py --input outputs/profiles/<USER_ID>.json --output outputs/profiles/<USER_ID>_summary.json
python src/profile_prompt/generate_user_profile_and_prompt.py --input outputs/profiles/<USER_ID>_summary.json --output outputs/profiles/<USER_ID>_prompt.json
```

Optional validation step:

```bash
python src/profile_prompt/validate_retrieval.py --user-id <USER_ID> --top-k 20 --recent-k 20 --exclude-recent --output outputs/profiles/<USER_ID>_validation.json
```

This gives you a clear pipeline:

1. retrieve top songs for a user
2. summarize the retrieved taste signals
3. generate a profile paragraph and prompt draft
4. optionally validate whether retrieval matches listening history well
