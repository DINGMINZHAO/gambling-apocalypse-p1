# Figure sources and provenance

The report includes eight empirical figures after removal of the overview pipeline diagram. Original analysis images and figures redrawn from archived CSV files contain no invented data. `generate_figures.py` reproduces the figures from the files in `figure_data/`.

| Report figure | Image | Source and interpretation |
|---|---|---|
| 1. Persistence / permutation | `02_skill_persistence.png` | Composition of original Q1 `01_oos_skill_deciles.png` and `01_adjusted_correlation_null.png`; 7d OOS / 1,000 permutations |
| 2. Experience | `03_experience.png` | Original Q1 `02_top_bottom_by_experience.png` |
| 3. Empirical Bayes | `04_empirical_bayes.png` | Original Q1 `03_oos_top10_comparison.png`; seven day specification |
| 4. Predictive models | `05_model_comparison.png` | Recreated from Q2 `model_scores.csv`, final_test/all rows for Logistic M0-M3 and DeepLearning S0-S2 |
| 5. Controlled history | `06_controlled_history.png` | Original Step32 `01_history_content_incremental.png`, C1 vs C0 market equal Brier gain |
| 6. Heterogeneity | `07_behavioral_heterogeneity.png` | Recreated from Step33 `behavior_subgroup_scores.csv`, recency and stake groups for `brier` |
| 7. Stake composition | `08_stake_probability.png` | Recreated from Step35.1 `probability_bin_composition.csv`, separate validation and final test composition |
| 8. Conviction paradox | `09_probability_bin.png` | Recreated from Step35.1 `probability_bin_contrasts.csv`, matched markets and purchased outcome probability bin; 95% event cluster bootstrap |

## Original result folders

- Skill persistence: https://drive.google.com/drive/folders/1b9_NL02yTxaZVJzGM7-hatKeBEAI6m3v
- Q1 permutation: https://drive.google.com/drive/folders/19RiLh6cGJP0yhfeVVprlMDAJn1T-yAHX
- Empirical Bayes: https://drive.google.com/drive/folders/1sS5lWMNcIkb_OiX_GEemcQCJFZxdiMsz
- Deep learning model scores: https://drive.google.com/drive/folders/1yE2ezOVVF8_7qlJ9ZHQ8e-E9x9OqFxfa
- Controlled historical information: https://drive.google.com/drive/folders/1iRQ3JJLIwiHqfRzimq5nabWf8lK-urqP
- Step33 heterogeneity: https://drive.google.com/drive/folders/1iayI7ZZytMFfFQkXpRYzP9JReUyE_H2K
- Step35 betting conviction: https://drive.google.com/drive/folders/1fVYcaQsa9MAAoFoC5vt2AvcWiuHmCrQ_

Research checkpoint commit: `97e0989ec852a196c2add91280bb6cfb2e977acc`.

The figure captions state metrics and sample restrictions. Values displayed in original research PNGs have not been adjusted. All redrawn numerical displays are derived from included CSV files or published checkpoint counts, and do not involve fitting, extrapolation, or interpolating new experimental results.
