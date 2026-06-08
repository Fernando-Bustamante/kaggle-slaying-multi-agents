# Kaggle Slaying Multi-Agents

Autonomous multi-agent system that participates in Kaggle competitions end-to-end — downloads data, detects the problem type, engineers features, tunes and trains models, submits predictions, and monitors the leaderboard — all from a single command.

Built for the CloudWalk Nimbus Challenge.

## Demo

```bash
python main.py titanic
# → Downloads data, trains, submits, monitors leaderboard automatically
```

## Results

| Competition | Domain | Metric | Score | Leaderboard |
|---|---|---|---|---|
| Titanic | Survival prediction | Accuracy | — | **Top 11%** (1,422 / 12,620) |
| Santander Customer Satisfaction | Financial / credit | AUC | **0.83943** | Above historical winner (0.82907) |
| Spaceship Titanic | Binary classification | Accuracy | 0.80617 | Top 29% |
| Bank Churn (Playground s4e1) | Customer retention | AUC | — | Top 29% |

The system performs strongest on **tabular binary classification with financial and behavioral data** — the core domain of credit risk and fraud detection.

## Architecture

Six specialized agents run sequentially, each with a single responsibility:

```
ConfigAgent → OrchestratorAgent → DataAgent → FeatureAgent → ModelingAgent → SubmissionAgent
```

| Agent | What it does |
|---|---|
| **ConfigAgent** | Downloads data via Kaggle API, auto-detects `id_col`, `target_col`, `predict_type` from `sample_submission.csv` |
| **OrchestratorAgent** | Calls Claude Haiku to analyze the dataset profile and decide the full modeling strategy: algorithm, trials, folds, metric, time budget, sample size |
| **DataAgent** | Loads train/test CSV with optional sampling for large datasets (100k+ rows) |
| **FeatureAgent** | Full feature engineering pipeline: missing value imputation per fold (no leakage), log1p on skewed features, target encoding with 5-fold CV, delimiter splitting, datetime extraction, pairwise interaction features, LightGBM gain-based feature selection |
| **ModelingAgent** | Tunes LightGBM with Optuna (TPE sampler, early stopping, time budget), trains final ensemble via OOF cross-validation, searches optimal threshold for accuracy competitions |
| **SubmissionAgent** | Generates `submission.csv` in the correct format and submits via Kaggle CLI |

### Intelligence layer

The **OrchestratorAgent** uses Claude to make all key decisions dynamically:

- Dataset size classification (small / medium / large / very large)
- Algorithm selection (LightGBM ensemble vs diverse ensemble)
- Cross-validation strategy (StratifiedKFold vs TimeSeriesSplit)
- Hyperparameter search budget
- Feature selection aggressiveness

This means the system adapts automatically to any tabular competition without manual configuration.

### Leaderboard monitoring

```bash
python leaderboard.py <competition-name>
```

Rich terminal UI showing position, score, top percentile, and submission history with score bars. Downloads the full public leaderboard to compute accurate rankings.

## Stack

- **Models:** LightGBM, XGBoost, CatBoost, RandomForest, LogisticRegression
- **Tuning:** Optuna (TPE sampler, early stopping, timeout budget)
- **Orchestration:** Claude Haiku via Anthropic API
- **GPU:** Auto-detected — uses GPU for binary/regression, CPU for multiclass
- **Interface:** Kaggle API + Rich terminal

## Running locally (VS Code / terminal)

```bash
git clone https://github.com/Fernando-Bustamante/kaggle-slaying-multi-agents
cd kaggle-slaying-multi-agents
pip install -r requirements.txt
```

Configure credentials:
```bash
# Kaggle API — place at ~/.kaggle/kaggle.json
# Anthropic API key
echo "ANTHROPIC_API_KEY=your_key" > .env
```

Run:
```bash
python main.py titanic
python main.py santander-customer-satisfaction

# Check leaderboard a few minutes after submission
python leaderboard.py titanic
```

## Running on Kaggle Notebook

```python
# Cell 1 — Install
!pip install anthropic optuna lightgbm xgboost catboost python-dotenv pyyaml rich -q

# Cell 2 — Clone/update
import os, subprocess
os.chdir("/kaggle/working")
if not os.path.exists("kaggle-slaying-multi-agents"):
    subprocess.run(["git", "clone", "https://github.com/Fernando-Bustamante/kaggle-slaying-multi-agents"], check=True)
else:
    subprocess.run(["git", "-C", "kaggle-slaying-multi-agents", "pull"], check=True)
os.chdir("/kaggle/working/kaggle-slaying-multi-agents")

# Cell 3 — API key
from kaggle_secrets import UserSecretsClient
os.environ["ANTHROPIC_API_KEY"] = UserSecretsClient().get_secret("Anthropic")

# Cell 4 — Run
!python main.py <competition-slug>
```
