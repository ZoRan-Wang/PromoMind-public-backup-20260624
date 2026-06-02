# 24-Hour Team Execution Plan

Context: The proposal presentation is due on June 3, 2026. The team has one day to prepare a high-precision proposal deck and speaking plan. This plan prioritizes proposal quality without pretending final experiments are already done.

## PM Decision

The team should split by slide/story ownership, not only by future code ownership. Each member owns a technical workstream and the proposal materials for that workstream.

| Member | Role | Proposal responsibility | Future project responsibility | Proposal speaking time |
| --- | --- | --- | --- | --- |
| A | Data and Problem Lead | Dataset, collection method, expected size, task definition, split | Raw data, cleaning, EDA, feature tables, leakage checks | 2:45 |
| B | Model Lead | Candidate generation algorithms and library acknowledgements | Popularity, Category Popularity, ItemKNN, ALS, BPR | 1:00 |
| C | Promotion and Evaluation Lead | Reranking formula, coupon/promotion signals, Business Utility@K, experiments | Promotion/coupon features, discount proxy, ablations, metrics | 2:05 |
| D | Integration and Presentation Lead | Architecture, demo concept, GitHub pack, final deck integration, timing | Streamlit demo, optional LightGCN, result integration, final report/deck | 3:00 |

Scripted talk time target: 8:50. Rehearsal target: 9:30 or less, leaving at least 30 seconds before the 10-minute cap.

## Timebox From Now To Class

### T-minus 24 to 18 hours

Goal: lock story and assignment.

- A confirms dataset facts and avoids restricted-school-file upload.
- B confirms algorithm list and external libraries to acknowledge.
- C confirms reranking formula and business metric wording.
- D checks deck structure, GitHub folder, and slide visual consistency.

Exit criteria:

- Everyone has read `proposal_deck_script.md`.
- No one is rewriting the project idea in a different direction.
- The deck covers every announcement requirement.

### T-minus 18 to 12 hours

Goal: polish each member's owned section.

- A prepares one clean dataset table and one 20-second explanation of why the time split is valid.
- B prepares one algorithm comparison table and one sentence on why implicit feedback fits grocery data.
- C prepares reranking formula explanation and proxy-profit limitation wording.
- D prepares the system architecture explanation and demo storyboard.

Exit criteria:

- Each speaker can explain their slides without reading paragraphs.
- Each section has one clear claim and one concrete proof object.

### T-minus 12 to 6 hours

Goal: rehearse and cut.

- Run the full deck once with a timer.
- Remove repeated explanation of the same dataset/method.
- Replace vague claims with precise proposal-stage statements.
- Verify the final deck still stays under 10 minutes.

Exit criteria:

- Team can finish in 9:30 or less.
- Everyone knows transition lines.
- Known limitations are framed honestly.

### T-minus 6 hours to submission

Goal: final package.

- D uploads the PPTX to eLearn before class time.
- D pushes final GitHub updates if any names/timing changed.
- A/B/C verify their own slide content one last time.
- Everyone keeps a local copy of the deck and GitHub link.

Exit criteria:

- eLearn has the proposal slides.
- GitHub has the proposal pack.
- Team has a 5-second message ready for classmates/meeting.

## Detailed Member Tasks

### A: Data and Problem Lead

Must deliver for proposal:

- Slide 3 dataset story.
- Slide 4 recommendation task definition.
- Dataset facts checked against repo docs:
  - households
  - transaction rows
  - products
  - coupon/campaign/promotion/demographic tables
- One sentence on collection method:
  - public `completejourney` package and raw RDS/RDA files, no private scraping.
- One sentence on risks:
  - coupon redemption sparse; demographics partial; both are handled as auxiliary signals.

Acceptance criteria:

- Can answer: "What data are you collecting and how large is it?"
- Can answer: "Why is the split chronological instead of random?"
- Does not use future-week data in the proposed training story.

### B: Model Lead

Must deliver for proposal:

- Slide 6 candidate generation story.
- Algorithm ladder:
  - Popularity
  - Category Popularity
  - ItemKNN
  - Implicit ALS
  - BPR if time allows
  - LightGCN as bonus
- Library acknowledgement:
  - `implicit` for ALS/BPR-style implicit recommenders.
  - RecBole for LightGCN if used.
  - Cornac optional/course-aligned framework if used.
- One sentence on why grocery has strong popularity baselines.

Acceptance criteria:

- Can answer: "Why ALS/BPR instead of ratings?"
- Can answer: "What happens if LightGCN is not finished?"
- Does not overpromise deep learning as required for the main project line.

### C: Promotion and Evaluation Lead

Must deliver for proposal:

- Slide 7 reranking formula.
- Slide 8 experiment and metric plan.
- Business Utility wording:
  - "revenue-minus-discount proxy", not "profit".
- Reranking ablation plan:
  - base only
  - promotion
  - coupon
  - promotion + coupon
  - discount-aware
  - full reranking with diversity
- Coupon sparsity explanation:
  - redemption is too sparse to be the only target, so it is an auxiliary signal.

Acceptance criteria:

- Can answer: "What is the X-factor beyond Top-K recommendation?"
- Can answer: "How do you evaluate the business effect?"
- Can answer: "Why is this not a causal coupon uplift model?"

### D: Integration and Presentation Lead

Must deliver for proposal:

- Slide 5 architecture.
- Slide 9 demo and X-factor storyboard.
- Slide 10 ownership and plan.
- GitHub proposal pack:
  - `deliverables/proposal_2026_06_03/`
- eLearn upload readiness:
  - final PPTX under 10 minutes.
- Optional:
  - update member names in deck if placeholders A/B/C/D need replacement.

Acceptance criteria:

- Can answer: "What will the working system look like?"
- Can answer: "Where can the team find the files?"
- Can run the meeting without re-summarizing the idea from scratch.

## Meeting Rule

Do not spend the online meeting restating the whole project. Start from the GitHub pack and ask only unresolved questions:

- Does anyone object to their assigned section?
- Does anyone need a name replacement in the deck?
- Does anyone see a factual error in dataset size or library acknowledgement?
- Who uploads to eLearn?

## Escalation Rules

- If a fact is uncertain, mark it as "expected" in proposal language rather than asserting it as completed.
- If a model is risky, move it from required to bonus.
- If the deck exceeds 10 minutes, cut details from algorithm tuning first, not from dataset/problem/X-factor coverage.
- If someone cannot finish their speaker section, D reads that section from `proposal_deck_script.md` but the original owner still owns follow-up work.
