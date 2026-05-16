# PromoMind Notebook Order

Use notebooks for exploration and evidence, while reusable logic should move into `src/` once stable.

Recommended order:

1. `01_data_loading_and_schema.ipynb`
   - Verify raw The Complete Journey tables.
   - Record exact column names, row counts, primary keys, and join keys.

2. `02_data_cleaning_and_split.ipynb`
   - Clean transactions/products.
   - Apply product/household filters.
   - Create Week 1-40 train, Week 41-46 validation, and Week 47-53 test files.

3. `03_eda.ipynb`
   - Produce dataset statistics and PPT-ready figures.
   - Include coupon redemption sparsity and demographics coverage.

4. `04_candidate_generation.ipynb`
   - Run popularity, category popularity, ItemKNN, ALS, and optional BPR.
   - Save candidate files using the shared schema.

5. `05_promotion_coupon_features.ipynb`
   - Validate promotion, campaign, coupon, and redemption joins.
   - Save promotion, coupon, and discount-cost feature tables.

6. `06_reranking_and_metrics.ipynb`
   - Run reranking variants.
   - Compute Recall, NDCG, coverage, diversity, novelty, and Business Utility@K.
   - Save ablation and final result tables.

7. `07_demo_data_export.ipynb`
   - Build demo-ready recommendation and household profile extracts.
   - Check that selected demo households have complete, interpretable rows.

8. `08_lightgcn_optional.ipynb`
   - Optional bonus notebook for RecBole/LightGCN experiments.
   - Must emit the same candidate schema as other models if included.

Notebook output rule: every notebook that creates a shared artifact should end with a short "Outputs written" cell listing filenames, row counts, and any known limitations.
