# Kaggle Slaying Multi-Agents

Autonomous multi-agent system that participates in Kaggle competitions end-to-end: downloads data, detects the problem type, engineers features, tunes and trains models, submits predictions, and monitors the leaderboard — all from a single command.

## Usage

```bash
# Run on any competition
python main.py <competition-name>

# Check leaderboard (run a few minutes after submission)
python leaderboard.py <competition-name>
```

**Examples:**
```bash
python main.py titanic
python main.py spaceship-titanic
python main.py santander-customer-satisfaction
```

## Results

| Competition | Type | Score | Result |
|---|---|---|---|
| Titanic | Binary classification | — | **Top 11%** |
| Spaceship Titanic | Binary classification | 0.80617 accuracy | Top 29% |
| Bank Churn (s4e1) | Binary classification | — | Top 29% |
| House Prices | Regression | 0.13335 RMSE | Top 46% |
| Santander Customer Satisfaction | Binary classification (financial) | 0.83943 AUC | Above historical winner (0.82907) |

## Architecture

The system is composed of 6 specialized agents that run sequentially:

```
ConfigAgent → OrchestratorAgent → DataAgent → FeatureAgent → ModelingAgent → SubmissionAgent
```

### Agents

| Agent | Responsibility |
|---|---|
| **ConfigAgent** | Downloads competition data, detects `id_col`, `target_col`, `predict_type` from `sample_submission.csv` |
| **OrchestratorAgent** | Calls Claude API to analyze the dataset profile and decide modeling strategy (`algorithm`, `n_trials`, `cv_folds`, `metric`, `time_budget`) |
| **DataAgent** | Loads train/test data with optional sampling for large datasets |
| **FeatureAgent** | Cleans data, applies log1p on skewed features, target encoding, datetime extraction, interaction features, and LightGBM-based feature selection |
| **ModelingAgent** | Tunes LightGBM hyperparameters with Optuna (early stopping), trains final ensemble with OOF cross-validation, searches optimal classification threshold |
| **SubmissionAgent** | Generates `submission.csv` and submits via Kaggle CLI |

### Leaderboard UI

```bash
python leaderboard.py <competition-name>
```

Displays a rich terminal interface showing your current position, score, top percentile, and submission history with score bars.

## Stack

- **Models:** LightGBM, XGBoost, CatBoost, RandomForest, LogisticRegression
- **Tuning:** Optuna with TPE sampler and early stopping
- **Orchestration:** Claude Haiku (Anthropic API)
- **GPU:** Auto-detected — uses GPU acceleration when available (Kaggle notebooks)
- **CLI:** Kaggle API

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/Fernando-Bustamante/kaggle-slaying-multi-agents
cd kaggle-slaying-multi-agents
pip install -r requirements.txt

# Configure Kaggle API credentials
# Place kaggle.json at ~/.kaggle/kaggle.json

# Configure Anthropic API key
echo "ANTHROPIC_API_KEY=your_key" > .env
```

## How It Works

1. **ConfigAgent** reads `sample_submission.csv` to auto-detect the prediction format (probabilities vs class labels, boolean format)
2. **OrchestratorAgent** profiles the dataset (row count, feature types, target distribution) and calls Claude to decide the optimal strategy — algorithm choice, number of trials, cross-validation folds, time budget
3. **FeatureAgent** runs a full feature engineering pipeline: missing value imputation per fold (no leakage), log1p on skewed distributions, target encoding with 5-fold CV, datetime extraction, pairwise interaction features, and LightGBM gain-based feature selection
4. **ModelingAgent** tunes LightGBM hyperparameters using Optuna on a sample, then trains the final ensemble on full data with honest OOF cross-validation. For accuracy competitions, searches the optimal classification threshold on OOF predictions
5. **SubmissionAgent** generates the submission file in the correct format and submits via Kaggle CLI
