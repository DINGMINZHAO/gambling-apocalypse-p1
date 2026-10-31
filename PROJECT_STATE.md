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
