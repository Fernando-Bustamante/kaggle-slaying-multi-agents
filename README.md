# Kaggle Slaying Multi-Agents

An autonomous multi-agent system that participates in Kaggle competitions end-to-end — from data acquisition to leaderboard monitoring — without human intervention.

Built as part of the **CloudWalk Nimbus Challenge**.

---

## Results

| Competition | Domain | Score | Leaderboard |
|---|---|---|---|
| Titanic | Survival prediction | — | **Top 11%** (1,422 / 12,620) |
| Santander Customer Satisfaction | Financial / credit risk | AUC 0.83943 | **Above historical winner** (0.82907) |
| Spaceship Titanic | Binary classification | 0.80617 accuracy | Top 29% |
| Bank Churn (Playground s4e1) | Customer retention | — | Top 29% |

The system achieves its strongest results on **tabular binary classification with financial and behavioral data** — the core domain of credit risk and fraud detection.

---

## Architecture

Six specialized agents run sequentially, each with a single responsibility:

```
ConfigAgent → OrchestratorAgent → DataAgent → FeatureAgent → ModelingAgent → SubmissionAgent
```

| Agent | Responsibility |
|---|---|
| **ConfigAgent** | Downloads competition data via Kaggle API and auto-detects problem structure from `sample_submission.csv` |
| **OrchestratorAgent** | Calls Claude Haiku (Anthropic) to analyze the dataset profile and decide the full modeling strategy |
| **DataAgent** | Loads train/test data with adaptive sampling for large datasets |
| **FeatureAgent** | Full feature engineering pipeline with leak-free per-fold imputation, log1p transforms, target encoding, interaction features, and LightGBM-based selection |
| **ModelingAgent** | Tunes LightGBM with Optuna (TPE sampler, early stopping, time budget), trains final OOF ensemble, searches optimal classification threshold |
| **SubmissionAgent** | Generates the submission file in the correct format and submits via Kaggle API |

### Intelligence layer — OrchestratorAgent

The OrchestratorAgent uses Claude to make all key decisions dynamically based on the dataset profile:

- Dataset size classification → algorithm selection and sampling strategy
- Task type detection → metric, CV strategy (StratifiedKFold vs TimeSeriesSplit)
- Hyperparameter search budget → trials, folds, time limit
- Feature selection aggressiveness

This allows the system to adapt to any tabular competition without manual configuration.

### Leaderboard monitoring

A dedicated terminal UI fetches the full public leaderboard, computes the user's real-time position and top percentile, and displays submission history with score progression bars.

---

## Stack

- **Models:** LightGBM · XGBoost · CatBoost · RandomForest · LogisticRegression
- **Tuning:** Optuna (TPE sampler, early stopping, timeout)
- **Orchestration:** Claude Haiku via Anthropic API
- **Acceleration:** GPU auto-detected and used when available
- **Interface:** Kaggle API · Rich terminal

---

## Usage

```bash
# Run on any Kaggle competition
python main.py <competition-slug>

# Check leaderboard position (a few minutes after submission)
python leaderboard.py <competition-slug>
```

Requires a Kaggle API key (`~/.kaggle/kaggle.json`) and an Anthropic API key (`.env`).
