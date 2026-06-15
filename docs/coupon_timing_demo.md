# Coupon Timing Demo

## Task

Given a household's historical grocery baskets before a campaign starts, predict time-product pairs for the next likely repurchase window.

For the coupon timing label:

```text
success = 1 if the exposed household buys the campaign coupon product within 5 days after campaign start
success = 0 otherwise
```

This reframes the project from only "which products will be bought next" to "which products may be bought in the coupon response window".

## External Model Code

The demo model step uses the external TBP/TARS next-basket repository:

```text
https://github.com/GiulioRossetti/tbp-next-basket
```

The local script only handles data conversion, Python 2 to Python 3 runtime compatibility, label construction, output formatting, and Streamlit display. It does not introduce a new model architecture or training strategy.

The external repo is GPL-3.0 licensed, so it is kept under `external/` and ignored by Git instead of being copied into this repository.

## Build Outputs

Run from the repository root:

```bash
python scripts/build_coupon_timing_demo.py --demo-events 3 --prediction-length 5 --max-label-rows 5000
```

Generated files:

- `outputs/coupon_timing_training_labels.csv`
- `outputs/demo_history_input.csv`
- `outputs/demo_time_name_recommendations.csv`

## Demo

Run:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit page shows:

- input historical purchases before coupon/campaign start
- output predicted purchase time and product name pairs
- whether the predicted item is coupon eligible in that campaign
- whether the item was actually bought within the 5-day success window

## Current Sample Size

The current local generated label file uses a controlled sample:

```text
False    4000
True     1000
```

Use a larger `--max-label-rows` value for the final experiment run.
