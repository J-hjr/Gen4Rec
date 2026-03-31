# Frontend Demo

## What this is

This is a local Streamlit demo for:

- loading a `user_id`
- viewing the listener profile built from recent listening history
- triggering Suno generation with one click
- browsing generated tracks, cover art, rerank scores, and reports
- visualizing recent listens and generated songs in CLAP embedding space

The demo is meant for local research / internal presentation use. It is not a production web app.

## Entry point

```bash
streamlit run app/streamlit_app.py
```

## Required local files

The demo expects the same local artifacts used by the current pipeline:

- `music4all/listening_history.csv`
- `music4all/id_information.csv`
- `music4all/audios/`
- `outputs/embeddings/music4all/music4all_embeddings.npy`
- `outputs/embeddings/music4all/music4all_ids.npy`
- `outputs/embeddings/music4all/user_embeddings.npy`
- `outputs/embeddings/music4all/user_ids.npy`

Optional but strongly recommended:

- `weights/clap/music_audioset_epoch_15_esc_90.14.pt`
- `weights/clap/clap_finetuned_best.pt`

If files are missing, the app will surface the same path-based errors used by the underlying scripts.

## Required environment variables

For profile generation:

```bash
export OPENAI_API_KEY="..."
```

For Suno generation through the ACE-compatible API:

```bash
export ACE_SUNO_API_KEY="..."
```

Optional path overrides still work through the existing `GEN4REC_*` environment variables.

## Typical flow

1. Launch the app.
2. Select or enter a `user_id`.
3. Click `Load profile` to build or load:
   - `outputs/profiles/<USER_ID>.json`
   - `outputs/profiles/<USER_ID>_summary.json`
   - `outputs/profiles/<USER_ID>_prompt.json`
4. Review the listener profile and summary.
5. Set generation controls and click `Generate songs`.
6. The app will:
   - run Suno generation
   - save a new run under `outputs/recSongs/<USER_ID>/<RUN_ID>/`
   - rerank the generated candidates with CLAP
7. Review:
   - selected tracks
   - all candidates
   - cover art and audio
   - `report.md`
   - embedding visualization

## Notes

- The app prefers the fine-tuned CLAP checkpoint for visualization and falls back to zero-shot if needed.
- Generation is explicit and button-driven; nothing expensive runs automatically on page load.
- The current app assumes a local single-user demo workflow and reads artifacts directly from `outputs/`.
