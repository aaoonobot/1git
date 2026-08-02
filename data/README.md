# Data directory

`data/manifests/` contains the small fixed manifests required to reconstruct the analytical sample. Large raw and processed event files are generated locally and are excluded from version control.

Use `scripts/01_fetch_statsbomb_passes.py` with both the competition and match manifests for exact sample reconstruction. The remaining numbered scripts then write intermediate files to a local `outputs/` directory.
