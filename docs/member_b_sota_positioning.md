# Member B SOTA Positioning

## Bottom Line

The Complete Journey project setup does not have a public, single-number leaderboard comparable to MovieLens. Therefore, we should not claim an official dataset SOTA.

What we can claim defensibly:

- The task is a grocery next-basket recommendation task.
- In next-basket recommendation literature, strong conventional methods include TOP / personal frequency, TIFU-KNN, and recency-aware collaborative filtering.
- On our time-based split, repeat-aware next-basket models strongly outperform generic ALS/BPR matrix factorization.
- Member B's strongest single-model source is `candidates_cornac_tifuknn.csv`.
- Member B's strongest recall-oriented source is `candidates_hybrid_strong.csv`.
- Member B's final protocol-best source is `candidates_sota_ensemble.csv`, a rank ensemble of Cornac TIFUKNN and `hybrid_strong`.

## Literature Basis

Relevant public sources:

- TIFU-KNN original implementation and paper reference: https://github.com/HaojiHu/TIFUKNN
- Reproduction/extension paper: https://arxiv.org/abs/2402.17925
- Empirical NBR study: https://arxiv.org/html/2312.02550v1

Key implications:

- TIFU-KNN was introduced for next-basket recommendation and models personalized item-frequency information.
- A 2024 reproduction reports that TIFU-KNN outperforms Personal Top Frequency across multiple public grocery datasets and metrics.
- A 2023 empirical NBR study reports that conventional approaches such as TOP, UP-CF@r, and TIFU-KNN are highly competitive; on Instacart and Dunnhumby-style datasets, UP-CF@r and TIFU-KNN show strong performance across metrics.

## Our Full Validation Result

Full run:

```powershell
python scripts/run_candidate_models.py --models all --k 50 --tifu-grid 50:0.7:0.95,100:0.7:0.95,200:0.7:0.95,200:0.8:0.95 --als-grid 32:0.05:5:20,64:0.05:5:20 --bpr-grid 16:0.03:0.01:2,32:0.03:0.01:3 --bpr-samples-per-epoch 50000
```

Official Cornac TIFUKNN run:

```powershell
python scripts/run_cornac_nbr_models.py --k 50 --tifuknn-grid 300:0.9:0.7:0.7:7
```

Final ensemble run:

```powershell
python scripts/run_sota_ensemble.py --weight-step 0.01 --primary-metric ndcg_at_10
```

Environment used for the Cornac check:

- `cornac==2.2.2`
- `numpy==1.26.4`
- `pandas==2.1.4`
- `scipy==1.13.1`

| model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0369 | 0.1234 | 0.0625 | 0.1244 |
| Personal Top Frequency | 0.0984 | 0.3790 | 0.1462 | 0.3402 |
| Category Popularity | 0.0460 | 0.1542 | 0.0728 | 0.1493 |
| ItemKNN | 0.0399 | 0.1980 | 0.0591 | 0.1659 |
| UPCF-style | 0.0874 | 0.3278 | 0.1242 | 0.2831 |
| TIFU-KNN style | 0.1011 | 0.3851 | 0.1503 | 0.3474 |
| Strong Hybrid | 0.1029 | 0.3935 | **0.1511** | 0.3528 |
| Official Cornac TIFUKNN | 0.1009 | **0.4210** | 0.1416 | **0.3574** |
| SOTA Ensemble | **0.1051** | **0.4278** | 0.1492 | **0.3691** |
| ALS | 0.0372 | 0.0743 | 0.0596 | 0.0788 |
| BPR | 0.0046 | 0.0143 | 0.0066 | 0.0127 |

The selected ensemble weight is:

```text
0.73 * reciprocal-rank(Cornac TIFUKNN)
+ 0.27 * reciprocal-rank(hybrid_strong)
```

## Recommended Presentation Claim

Use this wording:

> Because The Complete Journey does not have a single official leaderboard, we benchmark against community-standard next-basket methods under one time-based validation protocol. After correcting the grocery task setup to allow repeat purchases, repeat-aware models dominate generic matrix factorization. Our best final candidate source is a rank ensemble of Cornac's official TIFUKNN and our recall-oriented hybrid, improving NDCG@10 from 0.4210 to 0.4278 under our protocol.

Avoid this wording:

> We achieved official SOTA on The Complete Journey.

That claim is not defensible without a standardized public split, leaderboard, and identical metric protocol.
