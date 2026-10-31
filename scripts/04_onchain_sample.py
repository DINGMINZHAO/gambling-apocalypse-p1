from pathlib import Path
import duckdb

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = (
    "hf://datasets/moose-code/"
    "polymarket-onchain-v1/"
    "order_filled/year=2025/**/*.parquet"
)

OUTPUT = OUT / "onchain_sample_2025.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")
con.execute("SET enable_progress_bar=true")

# Enable remote Parquet access
con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

print("\n=== REMOTE DATA SCHEMA ===")

schema = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{SOURCE}')"
).fetchall()

for row in schema:
    print(f"{row[0]:35} {row[1]}")

print("\n=== EXTRACTING 100,000 ROWS ===")

con.execute(f"""
    COPY (
        SELECT *
        FROM read_parquet('{SOURCE}')
        LIMIT 100000
    )
    TO '{OUTPUT}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== SAMPLE SUMMARY ===")

summary = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp
    FROM read_parquet('{OUTPUT}')
""").fetchone()

print("Rows:", summary[0])
print("Earliest timestamp:", summary[1])
print("Latest timestamp:", summary[2])
print(
    "File size (MB):",
    round(OUTPUT.stat().st_size / 1024**2, 2)
)

print("\n=== FIRST RECORD ===")
row = con.execute(
    f"SELECT * FROM read_parquet('{OUTPUT}') LIMIT 1"
).df()

print(row.T.to_string(header=False))

con.close()
print("\nCompleted.")
