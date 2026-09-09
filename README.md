# product_category_match

## basic info

Match product records against a category taxonomy in the context of online retail.

contributor: Yuding Duan

the accuracy is not good enough for sku name matching (as the target), only do it to match category
cate->cate √
sku_name->cate √
any->sku_name ×

**English only** — the input must already be in English. There is no translation step.

**supports**:

1. two matching algorithms — `weighed_embedding` (collapse each taxonomy path into one weighted vector and compare them all at once) and `tree_based` (beam search down the taxonomy keeping `beam_width` candidates per level, then pick the best surviving path the weighed way)
2. two matching depths — fixed at `max_level`, or `flexible` (shallower paths compete too, so a record only goes as deep as it deserves)
3. optional `compact` step — strips marketing copy out of the input's last level (e.g. a concept-test description) down to a neutral "what it is" sentence before embedding
4. optional `selection` step — instead of taking the top-1 match, ask an LLM to pick among the matcher's top-k candidates
5. usable both as a batch script (`uv run main.py`) and as an importable module (`from catmatch import Matcher`)

## Preparation

dependencies are managed with [uv](https://docs.astral.sh/uv/)

1. install uv (if you haven't): https://docs.astral.sh/uv/getting-started/installation/

2. clone the repo
`git clone https://github.com/Kumoyyds/product_category_match.git`

3. go to the dir
`cd product_category_match`

4. create the virtual env and install the exact locked dependencies
`uv sync`  # creates `.venv/` and installs from `uv.lock` (add `--group dev` if you want the notebook kernel)

5. download the embedding model once (the `model/` folder ships empty)
```
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2').save('model/all-mpnet-base-v2')"
```

6. set up the compact/selection LLM
`cp .env.sample .env` and fill in **api_key** (plus `base_url` / `model` if you are not on the default endpoint).
Not needed if you set both `compact.enabled: false` and `selection.enabled: false` in `config.yaml`.

## data contract

Input and taxonomy files can be **xlsx, csv, json or jsonl** — but the column / key names are fixed:

| side | columns | example |
|---|---|---|
| input | `level_1`, `level_2`, … `level_n` (ascending; the last one may be free text such as an SKU name) | `[{"level_1": "food", "level_2": "healthy yogurt"}]` |
| taxonomy | `cat_1`, `cat_2`, … `cat_n` | `[{"cat_1": "food", "cat_2": "dairy"}]` |

The taxonomy may be ragged — a row that runs out of levels is a leaf, and matching stops there.

## usage

1. put your files in **input_data/** (records to match) and **data/** (the taxonomy)

2. adjust **config.yaml** — mainly `matching.algo`, `matching.max_level`, `matching.flexible` and, for `tree_based`, `matching.beam_width`; also `compact.enabled` (strip marketing copy from the last input level before embedding, only if it's over `compact.threshold_words`) and `selection.enabled` (let an LLM pick among the matcher's top `selection.top_k` candidates instead of taking the top-1)

3. run `uv run main.py` (no need to activate the venv)

4. find your output in **output/** — your original columns plus `cat_1..cat_k`, `match_level` and `sim` (plus `candidates`, the ranked top-k list, when `selection.enabled: true`)

Taxonomy embeddings are cached in a sqlite db (`cache/embeddings.sqlite`), keyed by `(text, model_name)`. When the taxonomy changes, just point `config.yaml` at the new file and run again — every run diffs the taxonomy's level texts against the cache and only embeds what's missing, so updates are incremental and automatic; no manual re-embedding step exists or is needed. Renamed/removed categories leave orphaned rows behind (harmless, just unused disk space) rather than being cleaned up. Switching `embedding.model_name` embeds everything fresh under the new key, without touching the old model's cached vectors. To force a full rebuild, delete `cache/embeddings.sqlite`. Input embeddings are computed fresh every run and are never cached.

## as a module

```python
from catmatch import Matcher, Config, MatchingConfig, read_records

records = read_records("input_data/fileexample.xlsx")
taxonomy = read_records("data/taxonomy.xlsx")

matcher = Matcher(Config(matching=MatchingConfig(algo="tree_based", max_level=5)))
results = matcher.match(records, taxonomy)   # list of dicts, one per input record
```

`Config.from_yaml("config.yaml")` loads the same settings the batch script uses; the LLM credentials always come from `.env`.
