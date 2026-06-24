# PromoMind Local Cache Packages

These packages contain generated artifacts for the final Zixun-cleaned run. They are committed because `outputs/`, `data/processed/`, and `data/raw/*.csv` are ignored.

Restore everything from the repository root:

```powershell
python scripts/restore_local_artifacts.py --clean
```

Then run the browser demo:

```powershell
python app/web_demo/server.py --port 8766
```

Open:

```text
http://127.0.0.1:8766/
```

## Packages

| Package | Restores |
| --- | --- |
| `PromoMind_raw_csv_cache_20260624_zixun.zip` | `data/raw/*.csv` |
| `PromoMind_processed_cache_20260624_zixun.zip` | `data/processed/*` |
| `PromoMind_outputs_core_20260624_zixun.zip` | final `outputs/*` except the two large feature matrices |
| `PromoMind_outputs_coupon_response_all_features_20260624_zixun.zip` | `outputs/coupon_response_all_features.csv` |
| `PromoMind_outputs_coupon_response_features_20260624_zixun.zip` | `outputs/coupon_response_features.csv` |
