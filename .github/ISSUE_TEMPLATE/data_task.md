---
name: Data task
about: Cleaning, split, schema, EDA, or processed dataset work
title: "[Data] "
labels: ["data"]
assignees: ""
---

## Goal

Describe the data artifact or analysis this issue should produce.

## Inputs

- Raw/processed files:
- Tables used:
- Upstream issue or dependency:

## Expected Outputs

- [ ] File(s) written:
- [ ] Row counts or summary stats documented:
- [ ] Schema or dictionary updated if needed:

## Leakage Checks

- [ ] Train/valid/test weeks are respected.
- [ ] Historical features use train-only data unless explicitly labeled.
- [ ] Valid/test target labels are not used in training features.

## Acceptance Criteria

- [ ] Artifact can be read by downstream model/reranking/demo code.
- [ ] Missing values and filtering choices are documented.
- [ ] Notebook/script records exact output path and row count.
