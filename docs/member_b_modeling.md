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

### 2. Personal Top Frequency

Ranks each household's own most frequently purchased products, with a small recency tie-breaker.

Purpose: strong next-basket baseline. Grocery prediction is heavily driven by repeated staple purchases, so this is much stronger than global popularity.

### 3. Category Popularity Baseline

Uses each household's historical category affinity and recommends popular items within those categories. The default category field is automatically selected from product metadata in this order:

1. `product_category`
2. `commodity_desc`
3. `sub_commodity_desc`
4. `department`
5. `category`

Cold-start households fall back to global category popularity.

Purpose: personalized baseline that reflects grocery category repeat behavior.

### 4. ItemKNN

Builds a household-product sparse matrix and computes item-item cosine similarity. A household's score for an unseen product is based on similarity to products the household previously bought.

Purpose: traditional collaborative filtering baseline before matrix factorization.

### 5. TIFU-KNN Style Next-Basket Model

Builds a time-decayed personalized item-frequency vector for each household, finds nearest households by cosine similarity, and combines personal repeat preference with neighbor preference.

This follows the TIFU-KNN family of next-basket recommenders. Public NBR literature reports that conventional frequency and KNN methods, especially TIFU-KNN and recency-aware user collaborative filtering, are highly competitive on grocery next-basket datasets.

Default tuning grid:

| neighbors | alpha | basket_decay |
| --- | --- | --- |
| 50 | 0.7 | 0.95 |
| 100 | 0.7 | 0.95 |
| 200 | 0.7 | 0.95 |

Purpose: community-level strong baseline for next-basket grocery recommendation.

### 6. Strong Hybrid Candidate Source

Combines the best validation candidate sources by rank:

```text
hybrid_score = 0.5 * TIFU-KNN rank score
             + 0.4 * Personal Top Frequency rank score
             + 0.1 * ItemKNN rank score
```

Purpose: final Member B candidate source for downstream promotion-aware reranking. It improves short-list ranking quality while keeping the model interpretable.

### 7. Implicit ALS

Implements implicit-feedback Alternating Least Squares. The wrapper supports:

- `backend="implicit"`: uses the optional `implicit` package when installed.
- `backend="native"`: uses the built-in scipy/numpy fallback.
- `backend="auto"`: uses `implicit` if available, otherwise native fallback.

Default tuning grid:

| factors | regularization | iterations | alpha |
| --- | --- | --- | --- |
| 16 | 0.05 | 3 | 10 |
| 32 | 0.05 | 5 | 20 |

Purpose: matrix-factorization comparison model. On this split, ALS is useful for coverage and novelty but is not the strongest next-basket model.

### 8. BPR Matrix Factorization

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
python scripts/run_candidate_models.py --models popularity,personal_topfreq,category,itemknn,tifu_knn,hybrid_strong --k 50
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
| `outputs/candidates_personal_topfreq.csv` | household repeat-frequency candidates |
| `outputs/candidates_category_popularity.csv` | household category-affinity candidates |
| `outputs/candidates_itemknn.csv` | item-based collaborative filtering candidates |
| `outputs/candidates_tifu_knn.csv` | TIFU-KNN style next-basket candidates |
| `outputs/candidates_hybrid_strong.csv` | final strong hybrid candidate source |
| `outputs/candidates_als.csv` | best ALS candidate output from tuning grid |
| `outputs/candidates_bpr.csv` | best BPR candidate output when BPR is requested |
| `outputs/tifu_tuning_results.csv` | TIFU-KNN parameter grid and validation metrics |
| `outputs/als_tuning_results.csv` | ALS parameter grid and validation metrics |
| `outputs/bpr_tuning_results.csv` | BPR parameter grid and validation metrics |
| `outputs/model_comparison.csv` | final model comparison table |

## Current Full Validation Result

Full run settings:

- Train split: 881,510 rows.
- Validation split: 132,638 rows.
- Product catalog: top 10,000 products.
- Candidate list: Top-50 per household.
- Repeat purchases are allowed, matching the grocery next-basket task.

| model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0369 | 0.1234 | 0.0625 | 0.1244 |
| Category Popularity | 0.0460 | 0.1542 | 0.0728 | 0.1493 |
| ItemKNN | 0.0399 | 0.1980 | 0.0591 | 0.1659 |
| Personal Top Frequency | 0.0984 | 0.3790 | 0.1462 | 0.3402 |
| TIFU-KNN style | **0.1011** | 0.3851 | **0.1503** | 0.3474 |
| Strong Hybrid | 0.1001 | **0.3888** | 0.1492 | **0.3485** |
| ALS | 0.0372 | 0.0743 | 0.0596 | 0.0788 |
| BPR | 0.0046 | 0.0143 | 0.0066 | 0.0127 |

Interpretation:

- The strongest single model is TIFU-KNN style, with the best Recall@10 and Recall@20.
- The strongest ranking source is the hybrid ensemble, with the best NDCG@10 and NDCG@20.
- ALS and BPR should be reported as matrix-factorization comparisons, not as the main result.
- The project should use `candidates_hybrid_strong.csv` or `candidates_tifu_knn.csv` as Member B's handoff to promotion-aware reranking.

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
| B9 model comparison table | implemented as `outputs/model_comparison.csv`, including strong next-basket baselines |
| B10 model PPT pages | drafted in `docs/member_b_slide_content.md` |
| B11 model report section | this file can be used as the report section |

## Notes For Final Presentation

Member B should make three points:

1. The candidate-generation stage optimizes purchase relevance before promotion logic is applied.
2. Grocery next-basket prediction is repeat-heavy; filtering previously purchased products is the wrong main-task setup.
3. The final strong candidate source is TIFU-KNN / hybrid next-basket modeling, while ALS/BPR are reported as comparison models.
