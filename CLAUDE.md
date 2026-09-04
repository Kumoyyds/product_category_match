# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A config-driven tool for matching product records against a target category taxonomy (online retail), using similarity over sentence-transformer embeddings. It ships as an importable package (`catmatch/`) plus a thin batch entry point (`main.py`).

Accuracy is only good enough to produce a **category**, never an SKU name — valid directions are `cate -> cate` and `sku_name -> cate`. **English only**: there is no translation step (it was removed deliberately — do not add one back without asking).

There are **no tests and no linter/formatter**.

## Commands

Python is pinned via `.python-version` (3.12) and dependencies via `uv` (`pyproject.toml` + `uv.lock`).

```powershell
uv sync               # install / recreate .venv from the lockfile (installs catmatch itself too)
uv sync --group dev   # also install ipykernel (for play.ipynb)
uv run main.py        # run the whole match pipeline (reads config.yaml)
```

Adding a dependency: `uv add <pkg>` / `uv add --group dev <pkg>`.

## The model gotcha (read before running)

`model/all-mpnet-base-v2/` is tracked in git as a `160000` gitlink with no `.gitmodules`, so fresh clones never populate it. Download the weights once:

```powershell
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2').save('model/all-mpnet-base-v2')"
```

The model is loaded **lazily** ([catmatch/embedding.py](catmatch/embedding.py)) from `EmbeddingConfig.model_path`, so importing `catmatch` from any directory is safe. `main.py` still `os.chdir(BASE_DIR)`s so the relative paths in `config.yaml` resolve against the repo root.

## Data contract

Both sides accept **xlsx / csv / json / jsonl**, and everything downstream sees the same JSON shape (a list of dicts). Column names are fixed by convention, not configured:

- input: `level_1 .. level_n`, ascending, contiguous from 1. The last level may be free text (an SKU name); there is no special-casing for it.
- taxonomy: `cat_1 .. cat_n`, same rule. Rows are ragged — a row that runs out of levels is a leaf.
- other columns ride through untouched into the output.

Violations raise from [catmatch/io.py](catmatch/io.py) `level_columns()`.

## Runtime architecture

```
config.yaml ─► main.py ─► catmatch.Matcher.match(records, taxonomy)
   .env ──────┘              │
                             ├─ io.py         four formats → list[dict]; contract check; clean()
                             ├─ compress.py   input values > threshold_words → LLM (concurrent, retries)
                             ├─ embedding.py  lazy SentenceTransformer, L2-normalised vectors
                             ├─ store.py      sqlite cache, TAXONOMY ONLY, key (text, model_name)
                             └─ matchers.py   WeighedEmbeddingMatcher | TreeBasedMatcher
                                              → output/ (xlsx | csv | json | jsonl)
```

- **Caching**: only taxonomy element embeddings are cached (`cache/embeddings.sqlite`). Input embeddings and compression results are recomputed every run, by design.
- **Vectors are L2-normalised at encode time**, and composed vectors are normalised again in `matchers.compose()`, so every `sim` is a real cosine. (The pre-refactor code used unnormalised dot products.)
- **Level weights**: `matchers.weight(k, p, aj=2.5)` — log-scaled, sums to 1 over the p levels, deeper levels weigh more.

**Two algorithms** (`matching.algo`):
- `weighed_embedding` — each candidate path collapses into one weighted vector; compare all candidates at once. Candidate set = full-depth paths, or (flexible) every prefix of depth 1..`max_level`.
- `tree_based` — beam search: at each level, score only the children of the paths still in the beam (using the *same* whole-record input vector throughout) and keep the best `beam_width` of them, all of them when there are fewer. A node with no children drops out of the beam but stays in the candidate pool. The winner is picked from that pool the `weighed_embedding` way — one composed vector per surviving path — so `sim` means the same thing in both algorithms.

**`flexible`** (one key, two implementations):
- `false` → depth is fixed at `max_level` (a branch that hits a leaf earlier still stops there).
- `true` → `weighed_embedding` lets shallow prefixes compete in the global pool; `tree_based` puts every path the beam held at every level into the candidate pool, so the final weighed pick chooses the depth.

**Output**: original input columns + `cat_1..cat_k` + `match_level` + `sim`. `k` is the deepest level any row reached; shallower rows leave the rest empty. A record with no usable text yields `'error'` in all three result columns.

## config.yaml / .env split

- `.env` (gitignored): `api_key`, `base_url`, `model` for the compression LLM. `LLMConfig.from_env()` only overrides what `.env` actually sets; `api_key` has no default.
- `config.yaml`: everything else, one section per component — `input` / `taxonomy` / `output` / `matching` / `embedding` / `compression` (with a nested `compression.llm` for behaviour: temperature, concurrency, retries, timeout, max output tokens).
- All of it is dataclasses in [catmatch/config.py](catmatch/config.py); validation lives in `__post_init__`, so an illegal config fails at construction. As a module you can bypass YAML entirely and build `Config(...)` directly.

## Data layout conventions

- `input_data/` records to match, `data/` the taxonomy, `output/` results, `cache/` the sqlite db (last two are gitignored).
- Example data is committed: `input_data/fileexample.xlsx` (`level_1..level_5`), `data/taxonomy.xlsx` (Google product taxonomy, `cat_1..cat_7`, 5595 rows), `input_data/cate.xlsx` (small `cat_1..cat_3` sample).

## Development notes

- Design notes live in `plans/*.md`, written in Chinese (README/commits are English). [plans/0904_matching_pipeline_redesign.md](plans/0904_matching_pipeline_redesign.md) records the 21 decisions behind the current architecture — read it before changing the pipeline shape.
- The dev environment is Windows with the repo inside a OneDrive-synced folder; `.venv/` holds tens of thousands of files, slow to sync but harmless.
- `.gitignore` also ignores a misspelled legacy `playground.ipynn` entry — leave it alone.
