# User embedding & retrieval CLI

## Required directories & files (defaults)

All paths are relative to the repo root.

### What to keep under version control (team workflow)

- **In Git:** `outputs/embeddings/music4all/*.npy` — embedding arrays are tracked (see `.gitignore`). The large song matrix `music4all_embeddings.npy` is stored with **Git LFS** (see `.gitattributes`); after `git clone`, run `git lfs pull` (or ensure Git LFS is installed so files are fetched automatically).
- **Not in Git:** `music4all/**/*.csv` (too large), `music4all/audios/`, and `weights/` — obtain CSVs and audio locally or via shared storage; place them under `music4all/` on each machine. You do not need weights if `music4all_embeddings.npy` already exists.

### Dataset: `music4all/`

These files are **not** in the repository (ignored by `.gitignore`); download or copy them locally. Expected layout (tab-separated CSVs where applicable):

- `listening_history.csv`
- `id_genres.csv`
- `id_information.csv`
- `id_metadata.csv`
- `id_tags.csv`

### Embeddings: `outputs/embeddings/music4all/`

| File | Where it comes from |
|------|---------------------|
| `music4all_embeddings.npy` | Produced by `src/embed/embed_music4all.py` or `embed_music4all_zeroshot.py`, or copied in from elsewhere; shape `(N, 512)`. |
| `music4all_ids.npy` | Optional at first. If missing, `build_user_embeddings.py` creates it from `id_genres.csv` (row order must match `music4all_embeddings.npy`). |
| `user_embeddings.npy` | Produced by `src/embed/build_user_embeddings.py`. |
| `user_ids.npy` | Produced by `src/embed/build_user_embeddings.py` (same row order as `user_embeddings.npy`). |

### Weights: `weights/clap/` (only if you encode audio yourself)

- `clap_finetuned_best.pt` — fine-tuned CLAP + attention for `embed_music4all.py`. **Not required** if you already have `music4all_embeddings.npy`.

---

## User embeddings (`build_user_embeddings.py`)

```bash
python src/embed/build_user_embeddings.py
python src/embed/build_user_embeddings.py --recent-k 20 --decay-lambda 0.08 --medoid-threshold 0.2 --min-keep 5
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--recent-k` | `50` | Max recent listening events per user before aggregation (smaller → more short-term). |
| `--decay-lambda` | `0.08` | Weight for older events: \( \exp(-\lambda \times \text{rank}) \); larger → stronger recency. |
| `--medoid-threshold` | `0.2` | Cosine similarity to the medoid track; songs below this are dropped as outliers. |
| `--min-keep` | `5` | After filtering, keep at least this many songs (fallback if too many are removed). |

---

## Top-K recommendations (`recommend_topk.py`)

```bash
python src/embed/recommend_topk.py --user-id <USER_ID> --top-k 20
python src/embed/recommend_topk.py --user-id <USER_ID> --top-k 10 --exclude-recent --with-info --with-metadata
```

| Parameter | Meaning |
|-----------|---------|
| `--user-id` | **Required.** User id as stored in `user_ids.npy`. |
| `--top-k` | How many nearest songs to return (default `20`). |
| `--exclude-recent` | Mask out every song that appears in `listening_history.csv` for this user so recommendations are not already heard. |
| `--with-info` | Join `id_information.csv`: `artist`, `song`, `album_name`. |
| `--with-metadata` | Join `id_metadata.csv`: Spotify id, audio features (`tempo`, `energy`, etc.). |

---

## Profile JSON (`export_user_profile_json.py`)

```bash
python src/embed/export_user_profile_json.py --user-id <USER_ID> --top-k 20
python src/embed/export_user_profile_json.py --user-id <USER_ID> --top-k 20 --exclude-recent -o outputs/profiles/<USER_ID>.json
```

| Parameter | Meaning |
|-----------|---------|
| `--user-id` | **Required.** User id in `user_ids.npy`. |
| `--top-k` | Number of nearest neighbors to include in JSON (default `20`). Same idea as `--top-k` in `recommend_topk.py`. |
| `--top-m` | Deprecated alias for `--top-k`. |
| `--exclude-recent` | Same as above: exclude songs the user already listened to. |
| `-o` / `--output` | Write JSON to this file; if omitted, print JSON to stdout only. |

JSON includes `info`, `metadata`, `genres`, and `tags` per song for downstream LLM use.
