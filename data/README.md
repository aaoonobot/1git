# Data

This folder describes the data workflow used in this repository.

The raw event data are obtained from StatsBomb Open Data. Large raw and processed CSV files are not committed to this repository because of file-size limitations. They can be regenerated using the scripts in the `scripts/` folder.

Suggested workflow:

```bash
python scripts/01_fetch_statsbomb_passes.py \
  --output_dir data/raw \
  --output_csv passes_all_matches_fixed.csv
