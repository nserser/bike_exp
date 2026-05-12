# Threshold schedule and AUC/top-tail metric fix

This patch fixes two issues identified after the new M3 experiments.

## 1. Threshold schedule mismatch

The experiment configuration files now use the schedules selected by the corresponding threshold sweeps:

| Experiment config | Previous schedule | Selected sweep schedule now used |
|---|---:|---:|
| `configs/m3_new_r47_d5_t2_k256.toml` | `4;4;3;3` | `4;3;3;3` |
| `configs/m3_new_r47_d7_t2_k128.toml` | `4;4;3;3` | `5;5;4;3` |
| `configs/m3_new_r31_d5_t3_k128.toml` | `5;4;3;3` | `5;5;4;4;3` |

The run scripts did not need structural changes because they already point to these config files.

## 2. AUC/top-10 risk-direction audit

`src/m3.rs::build_ranking_row` now computes `auc_bad_any` and `auc_top10_label` with score and label vectors aligned by `key_id`.

Metric convention:

```text
larger model score = higher predicted failure risk
```

The previous `auc_top10_label` computation paired score-sorted scores with labels derived from a separately score-ranked list. This could invert or scramble AUC values even when Spearman/top-tail recall were correct.

The CSV `model_ranking_summary.csv` now also annotates rows with:

```text
risk_direction=larger_score_higher_risk
```

## Validation

A unit test was added for the AUC/top-10 alignment fix. I could not execute `cargo test` in the current environment because neither `cargo` nor `docker` is installed here; run `./build.sh` or `docker build -t dependency-aware-dfrcert:v01 .` in a Docker-enabled environment to validate the patched implementation.
