# Gambling Apocalypse — Part I-A
## Research Results Checkpoint | 2026-11-01

**Status:** Analysis through Step 35.1 complete. Next action is research-report drafting, not further model training. This document records analysis results from completed Colab logs, GitHub scripts and checked Drive output folders. Statistics should be re-read from source CSV/JSON when writing final prose or figures. Do not silently revise numbers.

### 1. Questions and boundaries
- Q1, Forecasting Skill vs Luck: Do historically successful wallet addresses predict market outcomes better than market-implied probabilities, consistently out of sample?
- Q2, Behavioral Information Alpha (BIA): Does current and historical wallet behavior add predictive information beyond lagged market probabilities and basic wallet features?
- Behavioral extension: When wallets suddenly bet bigger, do they know better? Explore behavior, not PnL.
- Unit: pseudonymous wallet, no deanonymization; Q2 uses first qualifying taker BUY per wallet × market. Market/event clusters are correlated. This is a Web3 Data Science portfolio, not a claim of scientific first discovery or a live trading system.
- Separate work: Part I-B specialists/generalists; Part II price discovery and lead-follow networks; Part III trading signals and copy trading.

### 2. Existing data (NO REBUILD)
- Raw trades/fills 569,536,397; dates Jan 2025–Mar 2026; registry 612,892 markets; official probability histories 23,932 markets / 1,795,673 data points.
- 12h unified observations 7,143,340 wallet×market×month, 1,169,132 distinct wallets, 14,925 markets; 6,728,149 distinct wallet×market first entries.
- Q2 eligible samples: train 2,139,267 (Jan–Sep 2025); validation 568,424 (Oct–Dec 2025); final_test 1,973,839 (Jan–Mar 2026, 3,770 markets). Outcome-availability approximation uses grace days=7; 30-day sensitivity was used for Q1.
- Historical sequence: up to 16 strictly earlier eligible BUYs, right-aligned, 5 fields: direction, log_trade_size, p_market_before, price_age_hours, log1p(days before current); no future settlement labels. History >=1: train 75.6%, validation 82.9%, final 86.9%; history >=5: train 38.0%, validation 53.9%, final 63.5%.
- Probability proxy: latest completed 3h bucket, earlier than trade, up to 12h old. NOT necessarily an actual executable market quote at trade time.

### 3. Q1: Skill versus Luck (Steps 19–23)
**Do not conflate these cohorts or claim individual wallets are proven skillful.**

- 7d OOS persistence (min 5 train and min 5 future markets): 14,870 eligible wallets. Train–future wallet score correlation ~0.0234; D10 minus D1 future score ~+0.00414 with event-cluster CI [-0.00631, +0.01506]. Small/uncertain decile spread.
- Coverage-adjusted correlation ~+0.02273; 1,000 permutation null 95% interval [-0.01747, +0.01520], two-sided p ~0.007. Statistically detectable but small effect; distinguish correlation finding from decile-contrast uncertainty.
- In more experienced group (>=50 train and >=50 future markets), prior reported 7d Top10 future advantage ~+0.0405 vs Bottom10 ~+0.00252 (1,367 wallets); 30d ~+0.04674 vs Bottom10 ~-0.00223 (1,255 wallets). Confirm exact subgroup columns before quoting.
- Empirical Bayes top-decile ranking (7d, 14,870 wallets): future raw Top10 mean +0.0130085, EB Top10 +0.0187527, absolute lift +0.0057442 (~44% relative). Top10 rank overlap ~49.97%.
- Empirical Bayes (30d, 14,034 wallets): raw Top10 +0.0100304, EB +0.0180052, lift +0.0079748 (~79.5%). Top10 overlap ~50.96%.
- Interpretation: strong winner's curse for sparse-history wallets; modest persistence in full cohort, stronger effects in experienced cohorts; EB helps stabilize selection. Ex-post future-active-cohort selection and approximate label timing remain important. Stable negative forecasting skill (Predictable Fools) NOT established.
- Source Drive folders under data/research_results/: p1a_wallet_diagnostics, p1a_wallet_skill, p1a_skill_persistence_proxy_7d, p1a_skill_persistence_proxy_30d, p1a_score_permutation, p1a_empirical_bayes_7d, p1a_empirical_bayes_30d.
- Q1 visual candidates: p1a_empirical_bayes_7d/figures/03_oos_top10_comparison.png, 04_ranking_turnover_future.png, 05_oos_eb_minus_raw.png; p1a_skill_persistence_proxy_7d/figures/01_oos_skill_deciles.png and 05_top_bottom_contrast.png; p1a_score_permutation/figures/01_adjusted_correlation_null.png and 05_experienced_wallet_null.png.

### 4. Q2: Baselines, DL and historical information (Steps 24–32)
**Final-test market-equal Brier (lower better; models are on matched 1,973,839 Q2 rows):**

| Model | Brier | Log Loss |
| --- | ---: | ---: |
| Raw market | 0.129032 | 0.394562 |
| Logistic M1 market calibration | 0.127967 | 0.389999 |
| Logistic M2 anonymous action | 0.127915 | 0.389659 |
| Logistic M3 wallet intelligence | 0.127858 | 0.389461 |
| Step31 S0 static MLP | 0.128425 | 0.390696 |
| Step31 S1 masked history mean pooling | 0.127890 | 0.388933 |
| Step31 S2 GRU history order | 0.128600 | 0.390620 |
| Step32 C0 count-only 3-seed ensemble | 0.127466 | 0.387960 |
| Step32 C1 real history 3-seed ensemble | 0.127282 | 0.387081 |

- Caveat: raw market is a lagged 12h proxy; gains can shrink on fresher subsamples. Step29 LightGBM action L2 vs market L1 Brier improvement: within 12h +0.0005846; 3h +0.0004381; 1h +0.0003286 (not composition-identical).
- Step31 S1 vs S0 all-sample gain: validation -0.0006217, final_test +0.0005353, both event-cluster 95% CIs spanning zero. GRU S2 vs S1 final -0.0007100 CI [-0.001331, -0.000176], no established order advantage.
- Cold-start negative control: S1 vs S0 final +14.730 x10^-4 even with no history, demonstrating model-comparison confounding. Within frozen S1, disabling history changes cold-start predictions exactly zero; real history vs disabled gain on final history>=1 +2.862 x10^-4 CI [+0.891,+5.099] x10^-4; history>=5 +3.165 x10^-4 CI [+0.782,+5.669] x10^-4.
- Length-preserving matched history shuffle (frozen S1): real vs shuffled histories, history>=1 validation +2.942 x10^-4 CI [+0.249,+6.077], final +5.907 x10^-4 CI [+2.640,+9.226]. History>=5 validation +2.877 x10^-4 CI [-0.088,+6.036], final +5.856 x10^-4 CI [+2.313,+9.113]. Diagnostic, not final causal identification.
- Step32 controlled same-architecture C0 vs C1 (real history event content, actual count held fixed): seeds 13/42/2026 all yield positive Brier gains in both periods. Mean individual-seed gains x10^-4: validation +2.113, final +3.525. Ensemble gain validation +0.000084 CI [-0.000283,+0.000399]; final +0.000184 CI [-0.000057,+0.000391]. The ensemble-level uncertainty includes zero; do not claim confirmed incremental BIA.
- Step32 output directory: p1a_bia_history_control_7d/. Source: controlled_scores.csv, controlled_incremental.csv, 01_history_content_incremental.png; checkpoints and ensemble predictions.
- Step31 outputs: p1a_bia_deep_learning_7d/model_scores.csv, incremental_bootstrap.csv, 01_incremental_brier.png, 02_history_coverage_brier.png.
- Logistic outputs: p1a_bia_models_7d/model_scores.csv and analysis/; history/freshness and LightGBM each have separate folders.

### 5. Step33–34: Where is history useful?
- Step33 final-test subgroup C1 minus C0 market-equal Brier improvement (x10^-4), history>=5:
  - Recency <=7d: +2.393, CI [+0.287,+4.562]; 7–30d +0.893; >30d +0.278.
  - Higher stake: +3.734, CI [+1.101,+6.702]; similar stake +1.742; lower stake +1.339.
  - Lower chosen odds +2.573; similar chosen odds +0.450; higher chosen odds +1.968.
  - Same YES/NO side +2.370; changed side +0.777. YES/NO across unrelated markets does not represent semantically equivalent beliefs.
- Step34 market-paired higher vs similar gain gap (x10^-4): validation +1.462 CI [-1.484,+4.678]; final +1.735 CI [-0.365,+3.591]. Market FE with controls final Higher vs Similar within recent group -0.582 CI [-2.707,+1.542] x10^-4. Not independently robust.
- Important distinction: this section measures *incremental C1 vs C0 predictive benefit*, not wallet forecasting accuracy itself.
- Files: p1a_bia_behavior_patterns_7d/behavior_subgroup_scores.csv and 01_behavior_heterogeneity.png; p1a_bia_stake_heterogeneity_7d/paired_market_contrasts.csv, adjusted_contrasts.csv, fixed_effects_coefficients.csv and 01_stake_context_effects.png.

### 6. Step35/35.1: Betting conviction vs forecasting skill
Define market-adjusted purchased-outcome performance A = Z - q, where Z is whether the PURCHASED outcome occurred and q is lagged market probability of that purchased outcome (BUY NO uses q=1-p_yes). Values below in percentage points (pp); not PnL.

**Step35 final-test history>=5, market-equal group means:**

| Stake group | Market implied (%) | Realized win rate (%) | A (pp) | 95% CI (pp) |
| --- | ---: | ---: | ---: | ---: |
| Lower | 28.508 | 28.729 | +0.222 | [-0.327,+0.794] |
| Similar | 42.430 | 42.976 | +0.546 | [+0.042,+0.987] |
| Higher | 66.337 | 67.871 | +1.534 | [+0.825,+2.222] |

- Step35 matched-market Higher vs Similar: validation +0.049 pp CI [-1.069,+1.109]; final +1.086 pp CI [+0.483,+1.669]. Higher vs Lower: validation -0.617 pp CI [-2.313,+1.043]; final +1.274 pp CI [+0.393,+2.042]. Cannot claim temporal stability from this unstratified comparison.
- Final-test probability mix: Higher group's purchased-outcome probability 80–100% share **64.1%**; Similar 26.7%; Lower 10.0%. Higher stake often selects already favored outcomes; raw higher win rate is heavily confounded by this distribution.
- Step35.1 Higher vs Similar market-matched AND chosen-probability-bin-matched advantage gap (pp):
  
| Purchased-outcome probability | Validation gap / 95% CI | Final gap / 95% CI | Shared markets (val/test) |
| --- | --- | --- | ---: |
| 0–20% | -0.102 [-0.603,+0.494] | -0.095 [-0.397,+0.191] | 807/1,995 |
| 20–40% | +0.770 [+0.124,+1.488] | +0.671 [+0.070,+1.261] | 653/1,416 |
| 40–60% | **+3.050 [+0.258,+5.685]** | **+3.686 [+2.153,+5.279]** | 409/957 |
| 60–80% | +1.319 [+0.530,+2.127] | +0.386 [-0.050,+0.872] | 699/1,516 |
| 80–100% | +0.100 [-0.076,+0.284] | +0.047 [-0.121,+0.233] | 1,051/2,451 |

- Strongest exploratory finding: within 40–60% market probability bin, sudden higher-than-usual stakes correlate with larger market-adjusted purchased-outcome advantage, in BOTH validation and final. While most Higher Stake bets occur at 80–100% (final), their additional matched-market predictive advantage is close to zero in that bin.
- Crucial caveat: 20 pp bins do NOT control differences within bin; non-transaction-time 12h probability proxy may lag, shared-market paired means are conditional on participation, multiple probability-bin checks unadjusted. Do NOT label as proven private information, causal confidence or profit.

- Step35 files VERIFIED under Drive p1a_bia_betting_conviction_7d/: group_market_adjusted_scores.csv, common_market_contrasts.csv, 01_group_advantage.png, 02_common_market_contrasts.png, CONVICTION_MANIFEST.json, probability_bin_composition.csv, probability_bin_contrasts.csv, 03_probability_bin_robustness.png.
- The Step35.1 robustness experiment was a Colab cell (not yet a standalone GitHub script). Its saved CSV/PNG are present and were verified in Google Drive via folder listing.

### 7. Data provenance and links
Drive persistent root: MyDrive/gambling-apocalypse-p1/data/research_results/.
- Q1 EB 7d: https://drive.google.com/drive/folders/1sS5lWMNcIkb_OiX_GEemcQCJFZxdiMsz
- Q1 EB 30d: https://drive.google.com/drive/folders/1drjuSCR6pcXNaSFFU7IPWA8XLFJZnbQz
- Q1 skill persistence 7d: https://drive.google.com/drive/folders/1b9_NL02yTxaZVJzGM7-hatKeBEAI6m3v
- Q1 permutation: https://drive.google.com/drive/folders/19RiLh6cGJP0yhfeVVprlMDAJn1T-yAHX
- Q2 Step31: https://drive.google.com/drive/folders/1yE2ezOVVF8_7qlJ9ZHQ8e-E9x9OqFxfa
- Q2 Step32: https://drive.google.com/drive/folders/1iRQ3JJLIwiHqfRzimq5nabWf8lK-urqP
- Step33: https://drive.google.com/drive/folders/1iayI7ZZytMFfFQkXpRYzP9JReUyE_H2K
- Step34: https://drive.google.com/drive/folders/1awsvzM3a5AG26MbihqUKzPcN9omFX3f0
- Step35 and 35.1: https://drive.google.com/drive/folders/1fVYcaQsa9MAAoFoC5vt2AvcWiuHmCrQ_
- Code repository: https://github.com/DINGMINZHAO/gambling-apocalypse-p1 (Steps 19–35, newest script 35).
- Current last analysis-script commit before checkpoint: c81e8fe8df5a8220eadaa1ff591f9d3208ed67b4.

### 8. Report drafting protocol (MANDATORY for next chat)
- Write one coherent subsection at a time, not an entire report in one turn. Discuss narrative and evidence before drafting; user approves Chinese subsection, then draft faithful professional English counterpart. Repeat section by section. Start with a jointly agreed outline and one section.
- Chinese first, English second. Scientific terminology English in parentheses when helpful. English final report will ultimately be GitHub-ready Markdown.
- Every material number/claim MUST reference the exact CSV/JSON, matching metric, sample, cutoff, subgroup and confidence interval. Re-open sources as required; do not manufacture or smooth over results.
- Every major section SHOULD use 1–3 meaningful real plots/tables sourced from actual Drive figures or regenerated from verified data. Show figures and describe axes, cohorts and metric; avoid decorative AI charts or unsupported figures. Curate professional visuals, readable legends, white background, blue/orange/coral/teal palette.
- Clearly distinguish Q1 forecasting skill (market-relative outcomes) from Q2 incremental predictive Brier improvement; distinguish wallet profit, predictive skill, and behavior. Keep selected explanatory hypotheses labeled as hypotheses.
- Narrative priority: compelling research questions -> surprising empirical findings -> convincing methods as needed -> limitations without exhaustive paper-like digressions. Interesting, honest, technically competent portfolio over academic maximalism.
- Keep sections concise; avoid excessive infrastructure and unneeded further experiments. No more model training or chain data rebuilding unless the user later explicitly requests it.
- Suggested report: (1) Title + Executive Summary/teaser (2) Dataset and Polymarket primer (3) Q1 skill vs luck (4) Q2 behavioral information model ladder and history controls (5) Betting conviction finding with probability mix and bin plots (6) Caveats/what this implies (7) Reproducibility & artifact map.
- Critical: final_test has been repeatedly inspected and used to guide exploration. Treat all recent findings as exploratory out-of-sample evidence, not untouched confirmatory inference.
