from pathlib import Path
import duckdb

SOURCE = "data/processed/cross_time/*.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")

con.execute(f"""
CREATE VIEW trades AS
SELECT * FROM read_parquet('{SOURCE}')
""")

print("\n=== OVERALL COVERAGE ===")

result = con.execute("""
SELECT
    COUNT(*) AS fills,
    COUNT(DISTINCT condition_id) AS unique_markets,
    COUNT(DISTINCT wallet) AS unique_wallets,
    COUNT(DISTINCT sample_day) AS sampled_days,
    COUNT(DISTINCT condition_id)
        FILTER (WHERE pre_close = TRUE)
        AS pre_close_markets,
    COUNT(DISTINCT condition_id)
        FILTER (WHERE resolution_status = 'resolved')
        AS resolved_markets
FROM trades
""")

values = dict(zip(
    [x[0] for x in result.description],
    result.fetchone()
))

for name, value in values.items():
    print(f"{name:25}: {value:,}")


print("\n=== REPEATED MARKET COVERAGE ===")

market_stats = con.execute("""
WITH activity AS (
    SELECT
        condition_id,
        COUNT(DISTINCT sample_day) AS active_days,
        COUNT(DISTINCT wallet) AS wallets
    FROM trades
    GROUP BY condition_id
)
SELECT
    COUNT_IF(active_days >= 2),
    COUNT_IF(active_days >= 3),
    MAX(wallets)
FROM activity
""").fetchone()

print("Markets in 2+ days:", market_stats[0])
print("Markets in 3+ days:", market_stats[1])
print("Max wallets in one market:", market_stats[2])


print("\n=== MARKET CATEGORIES ===")

categories = con.execute("""
SELECT
    category_refined,
    COUNT(DISTINCT condition_id) AS markets,
    COUNT(*) AS fills
FROM trades
GROUP BY category_refined
ORDER BY markets DESC
""").fetchall()

for cat, markets, fills in categories:
    print(
        f"{str(cat):20} "
        f"Markets: {markets:>6,} "
        f"Fills: {fills:>9,}"
    )


# Update project handover state
state_path = Path("PROJECT_STATE.md")
state = state_path.read_text(encoding="utf-8")

start = "<!-- PILOT_COVERAGE_START -->"
end = "<!-- PILOT_COVERAGE_END -->"

checkpoint = f"""
{start}
## Pilot Market Coverage

- Source: TimeSeventeen/Polymarket-v1, daily_aligned
- Sample: six separate days in 2025 (not continuous)
- Trade fills: {values['fills']:,}
- Unique markets (condition_id): {values['unique_markets']:,}
- Unique wallets: {values['unique_wallets']:,}
- Markets with pre-close fills: {values['pre_close_markets']:,}
- Resolved markets: {values['resolved_markets']:,}
- Markets appearing in 2+ sample days: {market_stats[0]:,}
- Markets appearing in 3+ sample days: {market_stats[1]:,}
- These are pilot coverage statistics, not the final
  valid prediction sample or independent event count.

## Next Development Stage

Build continuous historical data for
2025-01-01 through 2026-03-31.

Primary candidate: TimeSeventeen/Polymarket-v1.
Supplement: official historical market probabilities.

Use Colab Pro for heavy processing and Google Drive
for persistent datasets. All cloud runs must record
the Git commit and data/run manifest.

Do not begin final skill inference until pre-trade
market probability and information cutoff are valid.
{end}
"""

if start in state and end in state:
    before = state.split(start)[0]
    after = state.split(end, 1)[1]
    state = before + checkpoint + after
else:
    state = state.rstrip() + "\n\n" + checkpoint

state_path.write_text(state, encoding="utf-8")

print("\nPROJECT_STATE.md updated.")
print("Completed.")

con.close()
