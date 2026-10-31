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

## Full Historical Trade Extraction — Completed

- Status: COMPLETE (2026-10-31)
- Source Git commit: 824c80ae2d2df1c879845b9447a99580ba1e2080
- Source dataset revision:
  5aa1b9d52316a8b2e789e81c8ae42c7ed532e8aa
- Period: 2025-01-01 to 2026-03-31 UTC
- Completed partitions: 912 / 912
- Missing partitions: 0
- Failed partitions: 0
- Extracted fills: 569,536,397
- Persistent storage: 10.59 GB
- Drive: gambling-apocalypse-p1/data/research_raw/
- RUN_MANIFEST.json verified by extraction script.
- Coverage refers to selected cleaned source layers.

Next:
- Build market and outcome-token registry.
- Obtain historical market probabilities.
- Validate information cutoffs before skill inference.
- Move into Part I-A research as soon as valid
  market-adjusted observations are available.

## Historical Probability Acquisition — 2026-10-31

Priority: acquire independent official market probability
history before further skill or BIA modeling.

- Script: scripts/16_market_probabilities.py
- Source: Polymarket Data API v2 /prices-history.
- Unit: YES token per condition_id.
- Frequency: historical 3-hour buckets.
- available_ts = bucket_start + resolution_seconds.
- Store only completed historical buckets.
- Candidate priority: >=50 fills, >=6h observed activity.
- Stratification: first active month and market type.
- These are acquisition priorities, not final inference
  sample eligibility criteria.
- Short-duration and less-active markets remain in
  the original transaction database.
- Output: Drive/data/market_probabilities/.
- Probability output budget: 1.5 GiB initially.
- Resume tracking: PROBABILITY_RUN_MANIFEST.json.
- Research-ready status: NOT YET.
- Next: strict as-of matching to trade timestamps and
  benchmark-age/coverage validation.

## Probability Acquisition Completed — 2026-10-31
- Official historical price requests: 24,000.
- Markets returning usable price history: 23,932.
- Markets without qualifying points: 68.
- Historical price points: 1,795,673.
- Output: Drive/data/market_probabilities/.
- Probability coverage is for prioritized candidates, not
  the complete 612,892-market population.
- Historical points use 3-hour aggregate buckets.
- Pre-trade alignment and temporal quality remain unverified.

## Probability Alignment
- Script: scripts/17_probability_alignment.py
- Scope: full 2025-01 through 2026-03 window.
- First BUY per wallet-market-month for initial alignment.
- Preserve original full trade database for later analysis.
- Match latest completed historical probability bucket
  strictly before each trade.
- Evaluate 1h and 3h benchmark freshness.
- Pre-close cutoff is provisional, not a verified
  first-public-outcome timestamp.
- No Forecasting Skill or BIA estimates yet.

# PROJECT HANDOFF — 2026-10-31
## Gambling Apocalypse — Part I

### CURRENT STATUS
Data acquisition, historical probability alignment,
and unified 12-hour sample validation are COMPLETE.

NEXT PHASE: Part I-A — Skill vs. Luck.
Do NOT repeat completed extraction or alignment work.

### 1. RESEARCH SCOPE

Project positioning:
Web3 Data Science / On-chain Behavioral Analytics.

Research unit:
Polymarket on-chain wallet address, NOT real identity.

Part I:
A. Skill vs. Luck
B. Predicting Smart Money
C. Specialists vs. Generalists

Two parallel research dimensions:
- Forecasting Skill: persistent market-adjusted
  predictive ability.
- Behavioral Information Alpha: incremental
  out-of-sample information from wallet behavior
  beyond the market benchmark.

Predictable Fool is an important research idea:
wallets with systematically negative forecasting
skill may still provide useful inverted information.

Part II (F/H) and Part III (D/E/I) are separate
future projects. Do not expand Part I into them.

### 2. COMPLETED HISTORICAL DATABASE

Period: 2025-01-01 to 2026-03-31 UTC.
Source: TimeSeventeen/Polymarket-v1.
Layers: daily_aligned + daily_aligned_multi.
Total fills: 569,536,397.
Partitions: 912/912 complete.
Raw data size: approximately 10.59 GB.

Market registry:
- Markets: 612,892
- Outcome token records: 1,108,682
- Multi-outcome event groups: 19,843

Relevant scripts:
- 14_full_trade_extract.py
- 15_market_registry.py

Drive:
gambling-apocalypse-p1/data/research_raw/
gambling-apocalypse-p1/data/research_index/

Original source commit:
824c80ae2d2df1c879845b9447a99580ba1e2080

Original Hugging Face revision:
5aa1b9d52316a8b2e789e81c8ae42c7ed532e8aa

### 3. COMPLETED MARKET PROBABILITIES

Script: scripts/16_market_probabilities.py

Source:
Polymarket official Data API v2 /prices-history.

Candidate markets requested: 24,000
Markets with usable history: 23,932
Markets without qualifying history: 68
Historical probability observations: 1,795,673
API requests: 24,000

Historical price frequency: 3-hour buckets.
Timestamp rule:
available_ts = bucket_start + resolution_seconds.

Important:
Probability coverage is restricted to prioritized
markets, not the entire historical universe.

Drive:
data/market_probabilities/

Manifest:
PROBABILITY_RUN_MANIFEST.json

### 4. COMPLETED PROBABILITY ALIGNMENT

Script: scripts/17_probability_alignment.py

Aligned period: all 15 months.
Sampling unit:
First taker BUY per wallet-market-month.

Total wallet-market-month observations: 10,342,265
Matched historical probability: 10,212,600
Within 1 hour: 876,039
Within 3 hours: 2,459,690

Matching rule:
Latest completed market probability bucket strictly
before the observed trade timestamp.

Output:
data/probability_alignment/wallet_market_YYYY_MM.parquet

All 15 monthly files completed.
ALIGNMENT_MANIFEST.json exists.
Process exited successfully.

### 5. UNIFIED RESEARCH SAMPLE — DECIDED

Script: scripts/18_sample_12h.py

Main sample selection:
- Historical market probability precedes trade.
- Maximum probability age: 12 hours.
- Valid binary market settlement label.
- Trade at least 1 hour before recorded close_at.
- First taker BUY per wallet-market-month,
  following the current alignment dataset.

Final results:
Total observations: 10,342,265
Matched probability: 10,212,600
Within 12 hours: 9,928,254
Valid under 3h rule: 1,777,424
Valid under unified 12h rule: 7,143,340
Distinct wallets: 1,169,132
Distinct markets: 14,925

Valid 12h / total observations: approximately 69.1%.

Price freshness diagnostic:

0-3h:
  Observations: 1,777,424
  Mean absolute execution/benchmark gap: 0.0244
  Median absolute gap: 0.0050

3-6h:
  Observations: 1,858,039
  Mean absolute gap: 0.0289
  Median absolute gap: 0.0050

6-12h:
  Observations: 3,507,877
  Mean absolute gap: 0.0340
  Median absolute gap: 0.0059

Decision:
Use ONE unified 12-hour main sample.
Do not repeatedly redesign sample filters.
No elaborate three-tier sample framework required.

Report:
data/research_results/P1_SAMPLE_12H_REPORT.json

### 6. IMPORTANT LIMITATIONS

- 12-hour prices are proxies for earlier market beliefs.
- close_at is not a verified result-publication time.
- Markets were prioritized for probability acquisition;
  the resulting sample is not the full market universe.
- Multiple wallets can trade the same market.
  Observations are not statistically independent.
- First BUY per wallet-market-month is an initial
  representation, not a complete wallet-level forecast.
- Taker-side direction and outcome mapping should be
  validated before making formal skill claims.
- No final Forecasting Skill, BIA, or ML results yet.
- Wallet skill features must only use outcomes
  known before each feature timestamp.

### 7. NEXT RESEARCH TASK

BEGIN Part I-A: WHO IS ACTUALLY SMART MONEY?

First research questions:
1. How many wallets have enough independent
   prediction opportunities for skill estimation?
2. How does market-adjusted wallet performance vary?
3. How much apparent performance can be explained
   by luck and correlated predictions?
4. Is performance persistent out of sample?
5. Are there systematically wrong but informative
   wallets (Predictable Fools)?

Research priorities:
- Statistical inference
- Skill vs. luck
- Wallet-level aggregation
- Market/event dependence
- Temporal out-of-sample validation
- Behavioral Information Alpha as parallel dimension

Start from descriptive diagnostics and wallet-level
sample structure before choosing statistical models.

DO NOT:
- Re-download 569M trades.
- Re-fetch 24K market probability histories.
- Rebuild the 15-month probability alignment.
- Reintroduce unnecessarily strict sample thresholds.
- Confuse exploratory statistics with proven skill.
- Start large ML pipelines before defining the target.

### 8. DEVELOPMENT WORKFLOW

Local environment:
MacBook Air M1, 8GB RAM.
VS Code is the authoritative code editing environment.
GitHub is the authoritative version control repository.
Colab Pro High-RAM executes heavier computations.
Google Drive stores persistent Parquet and manifests.

Repository:
https://github.com/DINGMINZHAO/gambling-apocalypse-p1

Local project:
~/Desktop/gambling-p1

Conda environment:
gambling-p1

Drive:
MyDrive/gambling-apocalypse-p1/

Workflow:
Write scripts locally in VS Code.
Git commit and push.
Pull exact commit in Colab.
Execute with progress output.
Persist compact results to Drive.
Update PROJECT_STATE.md at major milestones.

Communication:
Chinese with English technical terminology.
Explain Web3 concepts clearly.
Assume strong statistics and ML background.
Be rigorous but practical: portfolio first.
Move quickly toward meaningful empirical results.
Discuss important research decisions step by step.

END OF HANDOFF.


---

## Q2 CHECKPOINT — 2026-10-31 (after script 30)

**This section supersedes older "next step / not yet analyzed" statements above.**
Verified against the conversation's completed Colab logs. Scripts 19–30 are present on GitHub.
The Google Drive Parquet outputs are persisted according to successful Colab \`[DONE]\` logs; remote file-level re-verification has not been performed here.

### Research scope
Part I-A has parallel Q1 Forecasting Skill and Q2 Behavioral Information Alpha (BIA). Wallet-level only, not identities. Part I-B/C and Part III remain separate.
Q2 asks whether wallet actions and point-in-time historical characteristics improve YES/NO outcome probabilities beyond market probability. Prediction unit is **first qualifying taker BUY per wallet × market**.

### Data — do not rebuild
- Time window: 2025-01 to 2026-03.
- Raw fills: 569,536,397; registry: 612,892 markets.
- Official histories: 23,932 markets; 1,795,673 probability points.
- 12h unified sample: 7,143,340 wallet × market × month rows; 6,728,149 distinct wallet × market first entries.
- Data at \`MyDrive/gambling-apocalypse-p1/data/\`; preserve \`probability_alignment/\`, \`research_index/\`, \`research_results/\`.

### Q1 completed (scripts 19–23)
Wallet opportunity diagnostics, market-adjusted Skill, time-safe persistence, skill-vs-luck permutation, Empirical Bayes ranking.
Highlights reported: winner's curse for sparse-history wallets; stronger persistence for experienced wallets; EB improves ranking; stable negative Forecasting Skill / Predictable Fools has NOT been established. Existing Q1 result files are in \`data/research_results/\`.

### Q2 completed
- \`24_bia_features.py\`: time-aware history and fixed-prior shrunk-skill proxy, 7-day outcome-availability proxy. Outputs:
  \`data/research_results/p1a_bia_features_7d/\` with \`train.parquet\`, \`validation.parquet\`, \`final_test.parquet\`, and \`BIA_FEATURES_MANIFEST.json\`.
  Counts: train **2,139,267**, validation **568,424**, final_test **1,973,839**.
  Time splits by trade: train 2025 Jan–Sep, validation 2025 Oct–Dec, final test 2026 Jan–Mar. Excluded outcomes not available by appropriate split cutoff.
- \`25_bia_models.py\`: M0 raw market, M1 calibrated market, M2 anonymous action, M3 wallet intelligence (logistic). Outputs \`p1a_bia_models_7d/\`.
- \`26_bia_analysis.py\`: event-cluster bootstrap and heterogeneity; results under \`p1a_bia_models_7d/analysis/\`.
- \`27_bia_ablation.py\`: Experience vs Skill incremental comparison; \`p1a_bia_ablation_7d/\`.
- \`28_bia_lightgbm.py\`: nested L1 market, L2 action, L3 experience, L4 skill; trained models, predictions, figures under \`p1a_bia_lightgbm_7d/\`.
  Final-test Brier improvements: L2 vs L1 +0.0005846; L3 vs L2 -0.0002974; L4 vs L3 +0.0001813; L4 vs L2 -0.0001160. Event-cluster uncertainty evaluated.
- \`29_bia_freshness.py\`: existing-model prediction evaluation at 12h, 3h, 1h history-age caps, no refitting; \`p1a_bia_freshness_7d/\`.
  L2 vs L1 final-test Brier improvement: 12h +0.0005846, 3h +0.0004381, 1h +0.0003286; evidence gets weaker with fresher price subsets and market composition changes. Historical price is a lagged proxy, not an exact executable pretrade quote.
- \`30_bia_sequences.py\`: **COMPLETED successfully** in Colab CPU + High RAM, and writes to
  \`data/research_results/p1a_bia_sequences_7d/\`.
  Log: targets final_test 1,973,839 / train 2,139,267 / validation 568,424; processed >6 million wallet-market actions; final sequence-output counts exactly match all three target counts.
  Deliverables: \`train.parquet\`, \`validation.parquet\`, \`final_test.parquet\`, \`SEQUENCE_MANIFEST.json\`.
  Historical actions: up to 16 strictly earlier qualifying buys per wallet, with 5 sequence features: direction, log_trade_size, historical p_market_before, price_age_hours, log1p(days before current). No future settlement labels in sequence inputs. Use manifest \`history_ge_1\` / \`history_ge_5\` to inspect coverage; these counts have NOT yet been reported.

### Next step after reconnect: script 31 DL experiments
1. Verify Drive outputs and inspect \`SEQUENCE_MANIFEST.json\`, especially \`history_ge_1\` and \`history_ge_5\`. **Do not rerun script 30 unless files are missing or invalid**.
2. Change Colab runtime to GPU for PyTorch training (L4 / T4 / G4 as available); mounting Drive and recloning/pulling GitHub are sufficient. Colab \`/content\` is ephemeral; Drive outputs persist.
3. Train **S0 Static MLP**, **S1 History Pooling (order-insensitive)**, **S2 GRU (order-sensitive)** using identical splits/market-equal evaluation; compare to Logistic and LightGBM on the same sample; report Brier, Log Loss, event-cluster uncertainty and memory/sequence coverage. Keep one lightweight DL model rather than an unnecessarily heavy transformer until evidence warrants it.
4. Use temporal safety and market/event dependence safeguards. Past actions may be unclosed, but their future outcomes may NOT enter sequence features. The final test was already inspected, so further comparisons are exploratory OOS, not untouched final confirmation.

### Workflow
Mac \`~/Desktop/gambling-p1\` VS Code is normal script editing place; repository \`DINGMINZHAO/gambling-apocalypse-p1\` is formal code version source; Colab Pro is compute; Drive holds permanent data.
When resuming, synchronize local VS Code and Colab with GitHub HEAD; do not rely on old \`/content\` files. Continue with script 31, not any prior extraction/alignment step.
