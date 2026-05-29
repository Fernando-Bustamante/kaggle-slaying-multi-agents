# kaggle-slaying-agents
Autonomous multi-agent team for Kaggle competitions — collects data, trains models, submits, and tracks leaderboard.

## About
Multi-agent AI system competing autonomously on Kaggle.

## Usage
```bash
python main.py <competition-name>
```
The system downloads the data, detects the task type, decides the modeling strategy via Claude, tunes a LightGBM model with Optuna, and submits automatically.

## Agents
- **ConfigAgent** — detects `id_column`, `target_column`, and `predict_type` from `sample_submission.csv`
- **OrchestratorAgent** — uses Claude to decide `task_type`, `metric`, `n_trials`, `cv_folds`, and `sample_size`
- **DataAgent** — downloads and loads competition data
- **FeatureAgent** — cleans, encodes, and selects features via LightGBM importance
- **ModelingAgent** — tunes hyperparameters with Optuna and trains with seed bagging
- **SubmissionAgent** — creates and submits the final predictions

## config.yaml
For **debug/manual use only**. Ignored when a competition name is passed as argument. Fill in the fields to override the autonomous pipeline during local testing.
