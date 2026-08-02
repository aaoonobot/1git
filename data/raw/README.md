# Raw data

Raw StatsBomb pass-event files are not committed. Recreate the fixed sample from the repository root:

```bash
python scripts/01_fetch_statsbomb_passes.py \
  --output_dir data/raw \
  --output_csv passes_all_matches_fixed.csv \
  --competitions_manifest data/manifests/competitions_selected.csv \
  --match_manifest data/manifests/xpass_match_split_manifest.csv \
  --expected_matches 3926 \
  --expected_pass_rows 3806977
```
