import duckdb, glob
c = duckdb.connect()
imp = glob.glob("backend/data/processed/impressions/**/*.parquet", recursive=True)
print(c.execute("""
  SELECT COUNT(*) AS n,
         MIN(payprice) AS mn, MAX(payprice) AS mx,
         ROUND(AVG(payprice),2) AS mean,
         MEDIAN(payprice) AS med,
         ROUND(AVG(bidprice),2) AS avg_bid
  FROM read_parquet(?)
""", [imp]).df())
