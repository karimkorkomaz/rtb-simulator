"""Feature-loading layer: turns ingested Parquet tables into model-ready
(X, y) via `dataset.load_impression_features`, the single sanctioned entry
point that enforces `ingest.schema.feature_columns()`. See dataset.py.
"""
