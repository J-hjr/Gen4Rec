# Generate CLI

## Required files

The current generate stage reuses an existing prompt JSON from the profile-prompt pipeline.

Recommended upstream command:

```bash
python src/profile_prompt/generate_user_profile_and_prompt.py --input outputs/profiles/<USER_ID>_summary.json --output outputs/profiles/<USER_ID>_prompt.json
```

This file should contain:

- `user_id`
- `profile_paragraph`
- `suno_generation_prompt`
- `style_keywords`

## Environment variables

The current API path uses the ACE Data Suno-compatible API:

```bash
export ACE_SUNO_API_KEY="..."
```

## Suno generation (`run_generate.py`)

```bash
python src/generate/run_generate.py --prompt-json outputs/profiles/<USER_ID>_prompt.json
python src/generate/run_generate.py --prompt-json outputs/profiles/<USER_ID>_prompt.json --num-calls 5
python src/generate/run_generate.py --prompt-json outputs/profiles/<USER_ID>_prompt.json --num-calls 5 --max-concurrency 2
python src/generate/run_generate.py --prompt-json outputs/profiles/<USER_ID>_prompt.json --negative-prompt "heavy EDM drops, aggressive distortion"
python src/generate/run_generate.py --prompt-json outputs/profiles/<USER_ID>_prompt.json --lyrics-file lyrics.txt --generation-model chirp-v4-5
```

What it does:

- reads an existing prompt JSON from the current profile-prompt pipeline
- converts it into a normalized generation spec
- calls the ACE Data Suno-compatible API
- can repeat the API call multiple times for the same prompt
- supports bounded parallel sampling with a low concurrency cap
- downloads all returned audio variants into one run folder
- saves a run manifest and markdown report

| Parameter | Meaning |
|-----------|---------|
| `--prompt-json` | **Required.** Path to an existing prompt JSON. |
| `--user-id` | Optional override if you want the output folder to use a different user ID than the JSON. |
| `--provider` | Provider name recorded in manifests (default `suno`). |
| `--generation-model` | Hosted music model name (default `chirp-v4-5`). |
| `--num-calls` | How many API calls to make for the same prompt (default `1`). Each Suno call can return two candidate clips. |
| `--max-concurrency` | Maximum number of API calls to run in parallel (default `2`). Keep this low to reduce the chance of API throttling or auth issues. |
| `--negative-prompt` | Optional negative style guidance passed to the API. |
| `--lyrics-file` | Optional text file whose contents are sent as lyrics. If omitted, generation defaults to instrumental mode. |
| `--tempo-hint-bpm` | Optional BPM hint stored in the generation spec for future use. |
| `--duration-hint-seconds` | Optional duration hint stored in the generation spec for future use. |
| `--prompt-version` | Version string recorded in the saved generation spec. |

## Output layout

All generated files are written under:

```text
outputs/recSongs/<USER_ID>/<RUN_ID>/
```

`RUN_ID` is an auto-generated run identifier in this format:

```text
<UTC_TIMESTAMP>-<USER_ID>-<PROVIDER>
```

Example:

```text
20260330T213650Z-user_007XIjOr-suno
```

This helps keep multiple runs for the same user separate and prevents files from being overwritten.

Each run currently saves:

- `prompt_input.json`
- `generation_spec.json`
- `audio/call_01/<song_name>_variant_01.mp3`
- `audio/call_01/<song_name>_variant_02.mp3` (if returned)
- `audio/call_02/...` through `audio/call_05/...` when you use repeated calls
- per-variant lyric and metadata sidecar files
- `run_manifest.json`
- `report.md`

`run_manifest.json` also includes a `candidate_audio_paths` list so the next rerank stage can directly consume the generated clip paths.

This stage does not modify the current `profile_prompt` implementation. It only consumes its output.

---

## Rerank generated candidates (`rerank.py`)

```bash
python src/generate/rerank.py --manifest outputs/recSongs/<USER_ID>/<RUN_ID>/run_manifest.json
python src/generate/rerank.py --manifest outputs/recSongs/<USER_ID>/<RUN_ID>/run_manifest.json --top-k 2 --diversity-threshold 0.95
python src/generate/rerank.py --manifest outputs/recSongs/<USER_ID>/<RUN_ID>/run_manifest.json --encoder zeroshot
```

What it does:

- loads `candidate_audio_paths` from the generation manifest
- embeds each generated audio clip with CLAP
- computes cosine similarity between each clip embedding and the target `user_embedding`
- sorts candidates by CLAP cosine score
- optionally filters near-duplicate candidates with a diversity threshold
- saves `rerank_results.json` next to the manifest by default

| Parameter | Meaning |
|-----------|---------|
| `--manifest` | **Required.** Path to a generation run manifest JSON. |
| `--top-k` | How many final tracks to keep after reranking (default `2`). |
| `--diversity-threshold` | Optional cosine threshold for filtering near-duplicate clips. If omitted, reranking is score-only. |
| `--encoder` | Which CLAP encoder to use: `auto`, `finetuned`, or `zeroshot` (default `auto`). `auto` falls back to zero-shot if the fine-tuned checkpoint is unavailable. |
| `--output` | Optional custom output path for the rerank JSON. |

`rerank_results.json` includes:

- per-candidate CLAP cosine scores
- the full reranked list
- the final selected tracks after optional diversity filtering
