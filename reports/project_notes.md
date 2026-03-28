# Project notes

This package was cleaned for portfolio upload.

## What was cleaned

- removed `.git` metadata and Python cache files
- removed duplicated source folder `src/demand/demand/`
- fixed broken README formatting
- corrected project-root path bugs in preprocessing and feature engineering entry points
- replaced random optimization parameters with deterministic rule-based assumptions
- added a reusable `forecasting/model.py`
- fixed `service_level.py` so safety penalty cost is actually calculated
- updated `main.py` to represent the full end-to-end pipeline
- clarified naming around baseline vs enhanced forecast outputs

## Important note

Large raw files and bulky intermediate artifacts were intentionally excluded from this package so it is easier to upload to GitHub.
