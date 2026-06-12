# Member B Modeling Workstream: Candidate Generation

## Responsibility

Member B owns the first-stage recommender models. This stage predicts which products each household is likely to purchase next and emits Top-K product candidates for promotion-aware reranking.

The output contract is:

| column | meaning |
| --- | --- |
| `household_id` | household identifier |
| `product_id` | recommended product candidate |
| `base_score` | first-stage model relevance score |
| `model_name` | candidate-generation model name |
| `base_rank` | rank before promotion-aware reranking |

## Implemented Models

### 1. Popularity Baseline

Ranks products by global purchase frequency or quantity-weighted purchase volume in the training split. Every household receives the same global product ranking. For the main grocery next-basket task, already-purchased products are allowed because staple repeat purchases are valid recommendations.

Purpose: lower-bound baseline. If advanced models cannot beat this, the modeling setup needs to be revisited.

### 2. Category Popularity Baseline

Uses each household's historical category affinity and recommends popular items within those categories. The default category field is automatically selected from product metadata in this order:

1. `product_category`
2. `commodity_desc`
3. `sub_commodity_desc`
4. `department`
5. `category`

Cold-start households fall back to global category popularity.

Purpose: personalized baseline that reflects grocery category repeat behavior.

### 3. ItemKNN

Builds a household-product sparse matrix and computes item-item cosine similarity. A household's score for an unseen product is based on similarity to products the household previously bought.

Purpose: traditional collaborative filtering baseline before matrix factorization.

### 4. Implicit ALS

Implements implicit-feedback Alternating Least Squares. The wrapper supports:

- `backend="implicit"`: uses the optional `implicit` package when installed.
- `backend="native"`: uses the built-in scipy/numpy fallback.
- `backend="auto"`: uses `implicit` if available, otherwise native fallback.

Default tuning grid:

| factors | regularization | iterations | alpha |
| --- | --- | --- | --- |
| 16 | 0.05 | 3 | 10 |
| 32 | 0.05 | 5 | 20 |

Purpose: main traditional recommender model for the project.

### 5. BPR Matrix Factorization

Implements a lightweight Bayesian Personalized Ranking SGD model with pairwise positive-vs-negative item sampling.

Default tuning grid:

| factors | learning_rate | regularization | epochs |
| --- | --- | --- | --- |
| 16 | 0.03 | 0.01 | 3 |
| 32 | 0.03 | 0.01 | 5 |

Purpose: optional pairwise ranking comparison against ALS.

## How To Run

First generate processed files through the data workstream:

```powershell
python scripts/prepare_dataset.py --raw-dir data/raw --processed-dir data/processed
```

If the team uses the RDS/RDA cleaning workflow from the data branch, run:

```powershell
python scripts/clean_completejourney.py
```

Run the minimum B workstream models:

```powershell
python scripts/run_candidate_models.py --models popularity,category,itemknn,als --k 50
```

Run all B models including BPR:

```powershell
python scripts/run_candidate_models.py --models all --k 50
```

The runner allows repeat purchases by default. This matches grocery basket prediction, where previously purchased staples are often the most realistic next-basket items. To run a discovery-only experiment that filters previously purchased products, add:

```powershell
python scripts/run_candidate_models.py --models all --k 50 --exclude-seen
```

Run a fast smoke test after generating synthetic sample data:

```powershell
python scripts/make_sample_data.py
python scripts/prepare_dataset.py --split-mode fraction
python scripts/run_candidate_models.py --models all --k 3 --als-grid 3:0.05:2:5 --bpr-grid 4:0.05:0.01:2 --bpr-samples-per-epoch 20
```

## Generated Outputs

Generated files are written under `outputs/`, which is intentionally ignored by Git.

| file | purpose |
| --- | --- |
| `outputs/candidates_popularity.csv` | global popularity candidates |
| `outputs/candidates_category_popularity.csv` | household category-affinity candidates |
| `outputs/candidates_itemknn.csv` | item-based collaborative filtering candidates |
| `outputs/candidates_als.csv` | best ALS candidate output from tuning grid |
| `outputs/candidates_bpr.csv` | best BPR candidate output when BPR is requested |
| `outputs/als_tuning_results.csv` | ALS parameter grid and validation metrics |
| `outputs/bpr_tuning_results.csv` | BPR parameter grid and validation metrics |
| `outputs/model_comparison.csv` | final model comparison table |

## Evaluation Metrics

The model runner reports:

- `Recall@10`, `Recall@20`
- `NDCG@10`, `NDCG@20`
- `Coverage@20`
- `Diversity@20`
- `Novelty@20`

Promotion-aware metrics such as Business Utility@K are owned by Member C after reranking, but Member B's candidates provide the base scores and base ranks required by that stage.

## Acceptance Mapping

| task | status after implementation |
| --- | --- |
| B1 sparse matrix loader | implemented in `src/promomind/models/sparse.py` |
| B2 popularity baseline | implemented in `src/promomind/models/baselines.py` and CLI |
| B3 category popularity baseline | implemented in `src/promomind/models/baselines.py` and CLI |
| B4 ItemKNN | implemented in `src/promomind/models/itemknn.py` |
| B5 ALS first run | implemented in `src/promomind/models/als.py` and CLI |
| B6 ALS tuning | implemented through `--als-grid` and `als_tuning_results.csv` |
| B7 BPR first run | implemented in `src/promomind/models/bpr.py` and CLI |
| B8 BPR tuning | implemented through `--bpr-grid` and `bpr_tuning_results.csv` |
| B9 model comparison table | implemented as `outputs/model_comparison.csv` |
| B10 model PPT pages | drafted in `docs/member_b_slide_content.md` |
| B11 model report section | this file can be used as the report section |

## Notes For Final Presentation

Member B should make three points:

1. The candidate-generation stage optimizes purchase relevance before promotion logic is applied.
2. Popularity and category popularity are necessary baselines because grocery baskets have strong repeat and mass-market patterns.
3. ALS is the main model because the data is implicit feedback: purchases indicate positive preference, but missing purchases are not explicit dislikes.
