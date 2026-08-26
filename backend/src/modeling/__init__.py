"""CTR modelling stage: temporal splits, negative downsampling, evaluation
harness, and the logistic-regression baseline. Sits downstream of
`backend/src/features/dataset.py` (the sanctioned feature loader) and
`backend/src/ingest/` (raw -> Parquet). See `backend/src/ingest/README.md`,
"Where train/val/test splitting and downsampling would slot in", for the
design constraints this package implements.
"""
