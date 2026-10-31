# Gambling Apocalypse — Part I — Project State

## Project Overview
Wallet-level predictive intelligence on Polymarket.
Research scope: A, B, C.
Parallel dimensions: Forecasting Skill and BIA.

## Development Environment
- Local: MacBook Air M1, 8GB RAM
- IDE: VS Code
- Conda environment: gambling-p1
- Python: 3.11
- Storage: Local-first; Colab Pro and Google Drive if needed

## Completed Work
- Research scope and development principles established
- Local Conda environment created

## Data Inventory
No datasets acquired or verified yet.

## Technical Decisions
- DuckDB and Parquet preferred
- Research-first, minimal necessary data engineering
- Temporal validation required
- GitHub as the formal code version source

## Open Issues
- Historical data accessibility and coverage
- Trade-to-market and resolution mapping
- Historical market probability availability

## Current Status & Next Step
Phase 1: Obtain and inspect a small real Polymarket data sample.

## Version Information
Initial setup. Dataset version not yet assigned.


## Phase 1 Price Feasibility Checkpoint

### Historical Market Probability
- Official source: Polymarket Data API v2 /prices-history
- Resolution: 3-hour historical price buckets
- Audited 24 tokens and 3,669 trade fills
- Matched: 3,651 / 3,669 (99.5%)
- Within 1 hour: 1,071
- Within 3 hours: 1,954
- Median price age: 2.53 hours
- Mean absolute execution-vs-benchmark difference: 0.0992
- Historical prices are aggregate proxies, not orderbook quotes.
- Initial benchmark rule: completed historical bucket,
  with age <= 3 hours.
- Sensitivity analysis: age <= 1 hour.
- Temporal validity and benchmark accuracy require
  further verification on the full research cohort.

### Formal Data Plan
- Candidate window: 2025-01 through 2026-03 inclusive.
- Full continuous trade history, not six sampled days.
- Primary candidate source: TimeSeventeen/Polymarket-v1.
- Supplement: official market probabilities and metadata.
- Wallet-level analysis; preserve trade-level observations.
- Require valid pre-outcome information cutoff.
- Avoid treating repeated fills as independent outcomes.
- Preserve market/event dependencies in validation.

### Cloud Execution
- VS Code on Mac: authoritative code editing environment.
- GitHub: authoritative version-controlled code.
- Colab Pro High-RAM: large-scale execution when needed.
- Google Drive: persistent data and checkpoints.
- Every cloud run uses a fixed Git commit.
- Each run records commit, source partitions, parameters,
  output locations and completed work in a Run Manifest.
- No unsynchronized Colab-only code changes.

### Current Status
Historical probability feasibility established on pilot.
No final Forecasting Skill or BIA estimates yet.

### Next Step
Prepare GitHub-to-Colab execution, then build the
continuous historical research dataset.


<!-- PILOT_COVERAGE_START -->
## Pilot Market Coverage

- Source: TimeSeventeen/Polymarket-v1, daily_aligned
- Sample: six separate days in 2025 (not continuous)
- Trade fills: 1,004,808
- Unique markets (condition_id): 8,225
- Unique wallets: 103,589
- Markets with pre-close fills: 7,584
- Resolved markets: 8,225
- Markets appearing in 2+ sample days: 881
- Markets appearing in 3+ sample days: 351
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
<!-- PILOT_COVERAGE_END -->


<!-- OFFICIAL_API_AUDIT_START -->
## Official API Feasibility Audit

- Git commit: 8b0b52a368e11d8cffe5b0b06823a8869400001f
- Endpoint: Data API v2 /trades
- Historical market scope: fixed three-year window.
- Market queries ignore start/end filters.
- taker_only=false includes both wallet-side records.
- Markets tested: 12
- Markets reaching pilot date: 8
- Pages fetched: 41
- API rows: 40369
- Request attempts: 41
- Retries: 0
- Final historical data source decision: PENDING.
- Detailed results: data/audit/official_api_feasibility.json
<!-- OFFICIAL_API_AUDIT_END -->

## Phase 2 — Full History Optimized Extraction

Decision:
- Full UTC window: 2025-01-01 to 2026-03-31.
- Source: TimeSeventeen/Polymarket-v1 cleaned trade layers.
- Include daily_aligned and daily_aligned_multi.
- No day sampling and no wallet profitability selection.
- Extraction is split internally into daily partitions.
- Preserve all source-cleaned fills in the selected window.

Optimized storage:
- Script: scripts/14_full_trade_extract.py
- Trades: data/research_raw/trades/{market_type}/
- Metadata: data/research_raw/market_metadata/{market_type}/
- Market metadata is stored separately to reduce duplication.
- Both maker and taker addresses remain available.
- No premature wallet-market aggregation.

Execution:
- Mac VS Code is the only formal code editing environment.
- GitHub is the authoritative code repository.
- Colab Pro is the execution environment for large data.
- Google Drive stores persistent Parquet and RUN_MANIFEST.json.
- Git commit and Hugging Face revision are pinned.
- Runs are resumable using the same commit.

Outstanding research dependencies:
- Obtain official historical market probabilities.
- Reconstruct point-in-time market benchmarks.
- Validate outcome-information cutoffs.
- Validate wallet-side behavior and dataset filtering.
- Identify related markets within shared events.
- Avoid future outcome leakage in wallet features.
- Forecasting Skill and BIA are not yet estimated.
