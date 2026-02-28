# Gen4Rec

## 1) Create or update the Conda environment (`environment.yaml`)

This repository uses `environment.yaml` to manage dependencies.

### Create a new environment

```bash
conda env create -f environment.yaml
```

### Update an existing environment

```bash
conda env update -f environment.yaml --prune
```

`--prune` removes packages that are no longer listed in `environment.yaml`.

### Verify available environments

```bash
conda env list
```

If `environment.yaml` defines a name (for example `gen4rec`), activate it with:

```bash
conda activate gen4rec
```

---

## 2) Current workspace snapshot

```text
Gen4Rec/
├── .env.example
├── .gitignore
├── environment.yaml
├── readme.md
├── notebooks/
│   ├── music4all_data_check.ipynb
│   └── music4allOnion_data.ipynb
├── src/
│   ├── data/
│   │   └── build_music4all_db.py
│   ├── embed/
│   │   ├── embed_music4all.py
│   │   ├── embed_music4all_zeroshot.py
│   │   └── finetune_clap.py
│   ├── eval/
│   ├── generate/
│   └── pipeline/
├── music4all
├── music4allA+A
└── music4allOnion
```

Notes:
- Fine-tuning script: `src/embed/finetune_clap.py`
- Zero-shot embedding script: `src/embed/embed_music4all_zeroshot.py`
- Standard embedding script: `src/embed/embed_music4all.py`
- Music4All DB build script: `src/data/build_music4all_db.py`
- Data-check notebooks: `notebooks/music4all_data_check.ipynb` and `notebooks/music4allOnion_data.ipynb`

---

## 3) Repository structure (project layout)

```text
gen4rec/
├─ README.md
├─ pyproject.toml                 # or requirements.txt / environment.yml
├─ .env.example                   # DB_URL, DATA_ROOT, MODEL_CACHE, etc.
├─ .gitignore
├─ Makefile                       # common commands: lint/test/run pipeline
│
├─ configs/
│  ├─ default.yaml                # unified config entry
│  ├─ data_music4all.yaml
│  ├─ embed_clap.yaml
│  ├─ profile.yaml
│  ├─ generate.yaml               # Phase C
│  ├─ eval.yaml                   # Phase D
│  └─ prompts/
│     ├─ profile_schema.json      # profile schema (JSON)
│     ├─ profile_system.txt       # LLM system prompt
│     └─ profile_user_template.j2 # Jinja2 template
│
├─ data/                          # usually not tracked by git
│  ├─ raw/                        # Music4All raw index/metadata (or links)
│  ├─ interim/                    # intermediate cleaned outputs
│  ├─ processed/                  # processed tables (parquet/feather)
│  └─ samples/                    # tiny samples for tests/debug
│
├─ db/
│  ├─ schema.sql                  # database schema
│  ├─ seeds/                      # optional demo seed data
│  └─ migrations/                 # optional alembic migrations
│
├─ src/
│  └─ mgrec/                      # package name (customizable)
│     ├─ __init__.py
│     │
│     ├─ common/
│     │  ├─ logging.py
│     │  ├─ config.py             # merge yaml + env
│     │  ├─ paths.py              # unify DATA_ROOT/cache paths
│     │  └─ utils.py
│     │
│     ├─ data/                    # Phase 0/1: data & DB
│     │  ├─ music4all_loader.py   # map song_id -> audio_path
│     │  ├─ preprocess.py         # cleaning/normalization/export
│     │  └─ db_io.py              # write songs/users/events/tags to SQL
│     │
│     ├─ embed/                   # Phase A: embeddings
│     │  ├─ clap_embedder.py      # CLAP audio encoder wrapper
│     │  ├─ build_song_embeddings.py
│     │  ├─ build_user_embeddings.py  # recent-K + decay + normalize
│     │  └─ index_faiss.py        # optional Top-M retrieval index
│     │
│     ├─ profile/                 # Phase B: user profile
│     │  ├─ aggregate_features.py # aggregate from Top-M songs
│     │  ├─ schema.py             # Pydantic schema
│     │  └─ llm_profile.py        # structured stats -> profile text/JSON
│     │
│     ├─ generate/                # Phase C: generation (pluggable)
│     │  ├─ base.py               # generator interface
│     │  ├─ prompt_builder.py     # profile JSON -> generation prompt
│     │  ├─ musicgen.py           # example: MusicGen wrapper
│     │  ├─ audioldm.py           # example: AudioLDM wrapper
│     │  └─ suno.py               # example: Suno wrapper
│     │
│     ├─ rerank/                  # Phase D(1): rerank/selection
│     │  ├─ scorer.py             # score = cos(CLAP(gen), E_u) + objectives
│     │  └─ selector.py           # top-1/top-k, diversity penalties
│     │
│     ├─ eval/                    # Phase D(2): evaluation
│     │  ├─ metrics.py            # centroid sim, nn sim, density, etc.
│     │  ├─ fad.py                # optional FAD interface
│     │  ├─ reports.py            # markdown/json/csv reports
│     │  └─ ablation.py           # baseline comparisons
│     │
│     └─ pipeline/
│        ├─ run_embed.py          # run Phase A
│        ├─ run_profile.py        # run Phase B
│        ├─ run_generate.py       # run Phase C
│        └─ run_eval.py           # run Phase D
│
├─ scripts/
│  ├─ init_db.sh
│  ├─ ingest_music4all.py
│  ├─ build_embeddings.py
│  ├─ build_profiles.py
│  ├─ generate_candidates.py
│  └─ eval_run.py
│
├─ notebooks/                     # exploration only, not main pipeline
│  ├─ 01_data_sanity.ipynb
│  ├─ 02_embedding_space_viz.ipynb
│  ├─ 03_profile_examples.ipynb
│  └─ 04_eval_plots.ipynb
│
├─ tests/
│  ├─ test_user_embedding.py
│  ├─ test_profile_schema.py
│  └─ test_rerank_metrics.py
│
└─ outputs/                       # not tracked by git
	├─ audio/
	├─ profiles/
	├─ embeddings/
	└─ reports/
```

