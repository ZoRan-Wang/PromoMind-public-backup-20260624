# Useful File Index

This index tells the team what to read and what to avoid uploading.

## Proposal Pack

| Path | Purpose |
| --- | --- |
| `deliverables/proposal_2026_06_03/README.md` | Starting point for June 3 proposal work |
| `deliverables/proposal_2026_06_03/PromoMind_project2_proposal_slides.pptx` | Slide deck for eLearn upload |
| `deliverables/proposal_2026_06_03/proposal_deck_script.md` | Speaking script and timing |
| `deliverables/proposal_2026_06_03/full_project_proposal.md` | Full written proposal |
| `deliverables/proposal_2026_06_03/research_design_and_evaluation.md` | Research questions, hypotheses, protocol, validity threats |
| `deliverables/proposal_2026_06_03/24h_team_execution_plan.md` | PM-level urgent work split |
| `deliverables/proposal_2026_06_03/two_presenter_division.md` | Two-presenter slide split and transition lines |
| `deliverables/proposal_2026_06_03/presenter_1_background_zh_en.md` | Bilingual background notes for Presenter 1 |
| `deliverables/proposal_2026_06_03/presenter_2_background_zh_en.md` | Bilingual background notes for Presenter 2 |

## Core Repo Documents

| Path | Why it matters |
| --- | --- |
| `README.md` | Project overview and setup |
| `docs/project_plan.md` | Full project plan beyond proposal |
| `docs/experiment_protocol.md` | Evaluation, split, metrics, ablations |
| `docs/data_dictionary.md` | Raw and processed table contracts |
| `docs/demo_spec.md` | Streamlit demo behavior |
| `data/raw/completejourney/SOURCE.md` | Data source and license notes |

## Raw Data

The public Complete Journey raw artifacts are under:

```text
data/raw/completejourney/
```

Important files:

- `transactions.rds`
- `promotions.rds`
- `products.rda`
- `coupons.rda`
- `coupon_redemptions.rda`
- `campaigns.rda`
- `campaign_descriptions.rda`
- `demographics.rda`

## Code Skeleton

| Path | Purpose |
| --- | --- |
| `src/promomind/data/` | Data schemas and preprocessing hooks |
| `src/promomind/models/` | Candidate model skeletons |
| `src/promomind/rerank/` | Promotion-aware reranking code |
| `src/promomind/evaluation/` | Ranking and business metrics |
| `app/streamlit_app.py` | Demo shell |
| `scripts/download_completejourney.R` | R-side data export reference |
| `scripts/prepare_dataset.py` | Processed data preparation entrypoint |
| `tests/` | Smoke tests for metrics and core behavior |

## Do Not Upload To Public GitHub

- Official school PDF/course documents marked restricted.
- eLearn private pages or screenshots.
- Any classmate private credentials or private communication.
- Generated processed outputs if they are large or unreproducible.

The announcement requirements are summarized in `course_requirements_alignment.md` instead of uploading the restricted course PDF.
