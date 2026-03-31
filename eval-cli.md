# Eval CLI

This document explains the automatic evaluation flow under `src/eval/`.

## What It Does

Given a generation run manifest under `outputs/recSongs/<USER_ID>/<RUN_ID>/`, the eval pipeline:

- loads or recreates `rerank_results.json`
- embeds generated clips in the project CLAP space
- embeds the user's Recent-K listened songs in the same space
- computes user-embedding, recent-centroid, and reference-track similarity metrics
- reports diversity and a simple "too close to reference" novelty proxy
- optionally saves an embedding-space plot

## Default Embedding Space

By default, eval uses the finetuned CLAP encoder:

```bash
python src/eval/run_eval.py --manifest outputs/recSongs/<USER_ID>/<RUN_ID>/run_manifest.json
```

By default, artifacts are written to:

```text
outputs/eval/<USER_ID>/<RUN_ID>/
```

If you explicitly want fallback behavior, use:

```bash
python src/eval/run_eval.py \
  --manifest outputs/recSongs/<USER_ID>/<RUN_ID>/run_manifest.json \
  --encoder auto
```

If `--encoder auto` falls back to zero-shot CLAP, that encoder choice is recorded in the eval output JSON.

## Useful Flags

- `--recent-k`: how many recent listened songs to use as the reference set
- `--top-reference-k`: top-N reference similarities to average for the `reference_topn_mean_cosine` metric
- `--rerank-top-k`: only used if `rerank_results.json` is missing and rerank must be recomputed
- `--diversity-threshold`: only used if rerank must be recomputed
- `--save-plot`: save `embedding_space.png`
- `--output-dir`: write eval artifacts somewhere other than `outputs/eval/<USER_ID>/<RUN_ID>/`
- `--imitation-threshold`: threshold used for the "too close to reference" flag

Example:

```bash
python src/eval/run_eval.py \
  --manifest outputs/recSongs/<USER_ID>/<RUN_ID>/run_manifest.json \
  --recent-k 20 \
  --top-reference-k 3 \
  --save-plot
```

## Output Artifacts

By default the command writes to `outputs/eval/<USER_ID>/<RUN_ID>/`:

- `eval_summary.json`
- `eval_report.md`
- `reference_alignment.csv`
- `embedding_space.png` when `--save-plot` is enabled

## Metric Summary

- `user_embedding_cosine`: similarity to the stored `user_embeddings.npy` vector
- `recent_centroid_cosine`: similarity to the centroid of embedded Recent-K reference tracks
- `reference_mean_cosine`: mean similarity to the full Recent-K reference set
- `reference_max_cosine`: best match to any single reference track
- `reference_topn_mean_cosine`: mean of the top-N reference similarities
- diversity metrics: pairwise and nearest-neighbor similarities within candidate and selected sets
- novelty proxy: a boolean flag when a generated clip is very close to a reference track

## Notes

- The automatic metrics are proxies for personalization and coherence, not a replacement for the later human study.
- Eval and visualization now share the same reusable CLAP audio embedding helpers as rerank.
