import json
import re
import numpy as np
import pandas as pd
import anthropic


class OrchestratorAgent:
    def __init__(self, competition_name: str):
        self.competition_name = competition_name
        self.client = anthropic.Anthropic()

    def _detect_timeseries(self, train: pd.DataFrame, target_col: str) -> dict:
        ts_keywords = ("date", "time", "year", "month", "week", "day", "timestamp", "created", "updated")
        ts_cols = []
        for col in train.columns:
            if col == target_col:
                continue
            parsed = pd.to_datetime(train[col], errors="coerce")
            if parsed.notna().mean() > 0.5:
                col_l = col.lower()
                if any(k in col_l for k in ts_keywords) or parsed.notna().mean() > 0.8:
                    ts_cols.append(col)
        is_sorted = False
        if ts_cols:
            parsed = pd.to_datetime(train[ts_cols[0]], errors="coerce").dropna()
            is_sorted = bool(parsed.is_monotonic_increasing)
        return {
            "timeseries_candidate_cols": ts_cols[:3],
            "data_appears_sorted_by_time": is_sorted,
        }

    def _build_data_profile(self, train: pd.DataFrame, target_col: str, n_rows_total: int) -> dict:
        numeric_cols = train.select_dtypes(include=[np.number]).columns.drop(target_col, errors="ignore").tolist()
        cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
        missing_pct = round(train.isnull().sum().sum() / (train.shape[0] * train.shape[1]) * 100, 2)

        target_series = train[target_col].dropna()
        unique_count = target_series.nunique()
        if pd.api.types.is_numeric_dtype(target_series) and unique_count > 20:
            target_dist = {
                "type": "continuous",
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "mean": float(target_series.mean()),
                "unique_count": int(unique_count),
            }
        else:
            target_dist = {
                "type": "categorical",
                "value_counts": target_series.value_counts(normalize=True).round(4).head(10).to_dict(),
            }

        ts_info = self._detect_timeseries(train, target_col)

        return {
            "competition": self.competition_name,
            "n_rows_total": n_rows_total,
            "n_rows_sample": len(train),
            "n_cols": train.shape[1],
            "n_numeric_features": len(numeric_cols),
            "n_categorical_features": len(cat_cols),
            "missing_pct": missing_pct,
            "target_column": target_col,
            "target_distribution": target_dist,
            "column_names_sample": train.columns[:30].tolist(),
            "timeseries_candidate_cols": ts_info["timeseries_candidate_cols"],
            "data_appears_sorted_by_time": ts_info["data_appears_sorted_by_time"],
        }

    def decide(self, train: pd.DataFrame, config: dict, actual_n_rows: int = 0) -> dict:
        target_col = config["competition"]["target_column"]
        predict_type = config["competition"].get("predict_type", "proba")

        print("[OrchestratorAgent] Building data profile...")
        profile = self._build_data_profile(train, target_col, n_rows_total=actual_n_rows or len(train))
        print(f"[OrchestratorAgent] Profile built: {profile['n_rows_total']:,} total rows, {profile['n_cols']} cols")

        prompt = f"""You are an expert Kaggle data scientist. Analyze this competition dataset and decide the optimal modeling strategy.

Dataset profile:
{json.dumps(profile, indent=2)}

NOTE: n_rows_sample is a 5000-row probe used only for feature profiling.
      n_rows_total is the ACTUAL full training set size — use this for ALL size-based rules below.

Detected predict_type from sample_submission: "{predict_type}"
  - "class" means the submission expects integer labels (0/1)
  - "proba" means the submission expects probabilities

Your task: decide the best modeling strategy for this competition.

SIZE-BASED RULES (apply strictly based on n_rows_total):

SMALL dataset (n_rows_total < 2000):
  - algorithm: "diverse"  ← MANDATORY. Blends LightGBM + RandomForest + LogisticRegression.
  - n_trials: 10-15
  - cv_folds: 10
  - max_num_leaves: 15-25
  - min_child_samples_min: max(30, n_rows_total // 25)
  - n_ensemble_models: 3
  - sample_size: null
  - time_budget_minutes: 30

MEDIUM dataset (2000 ≤ n_rows_total < 20000):
  - algorithm: "diverse"  ← preferred.
  - n_trials: 20-40
  - cv_folds: 10 if n_rows_total < 10k, else 5
  - max_num_leaves: 31-80
  - min_child_samples_min: 20-50
  - n_ensemble_models: 3-4
  - sample_size: null
  - time_budget_minutes: 45

LARGE dataset (20000 ≤ n_rows_total < 200000):
  - algorithm: "lgbm"  ← LightGBM ensemble only. Do NOT use diverse — RF/LR are too slow on large data.
  - n_trials: 30-50
  - cv_folds: 5
  - max_num_leaves: 63-200
  - min_child_samples_min: 10-30
  - n_ensemble_models: 3-5
  - sample_size: null if n_rows_total ≤ 50k, else min(50000, n_rows_total // 4)
  - time_budget_minutes: 90

VERY LARGE dataset (n_rows_total ≥ 200000):
  - algorithm: "lgbm"
  - n_trials: 20-30 (stable CV with huge data means fewer trials needed)
  - cv_folds: 3
  - max_num_leaves: 63-300
  - min_child_samples_min: 10-20
  - n_ensemble_models: 3
  - sample_size: 50000
  - time_budget_minutes: 120

OTHER RULES:
- task_type: infer from target distribution. 2 unique values = binary_classification. >2 discrete = multiclass_classification. continuous = regression.
- metric: roc_auc for binary_classification, accuracy or logloss for multiclass, rmse or mae for regression.
- feature_selection_pct: fraction of features to keep (0.05 to 1.0).
  * n_numeric_features > 100 AND n_rows_total > 10x n_numeric_features: set 1.0
  * n_numeric_features > n_rows_total: set 0.1-0.3
  * Otherwise: 0.5-0.8.

TIME-SERIES RULES:
- is_timeseries: true if data_appears_sorted_by_time=true OR timeseries_candidate_cols is non-empty AND
  the column names strongly suggest temporal ordering.
- For time-series: prefer cv_folds=5, n_trials on the lower end.

Respond ONLY with a valid JSON object using exactly these keys:
{{
  "task_type": "...",
  "metric": "...",
  "algorithm": "lgbm" or "diverse",
  "n_trials": <int>,
  "cv_folds": <int>,
  "sample_size": <int or null>,
  "feature_selection_pct": <float>,
  "n_ensemble_models": <int>,
  "max_num_leaves": <int>,
  "min_child_samples_min": <int>,
  "is_timeseries": true or false,
  "time_budget_minutes": <int>,
  "feature_hints": ["...", "..."],
  "reasoning": "..."
}}"""

        print("[OrchestratorAgent] Calling Claude API...")
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        print("[OrchestratorAgent] Response received.")

        text = response.content[0].text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"[OrchestratorAgent] Could not parse JSON from response:\n{text}")
        decisions = json.loads(match.group())

        print(f"\n[OrchestratorAgent] {decisions.get('reasoning', '')}")
        print(
            f"[OrchestratorAgent] task={decisions['task_type']} | metric={decisions['metric']} "
            f"| trials={decisions['n_trials']} | folds={decisions['cv_folds']} | sample={decisions['sample_size']} "
            f"| budget={decisions.get('time_budget_minutes', 90)}min | timeseries={decisions.get('is_timeseries', False)}"
        )
        if decisions.get("feature_hints"):
            for hint in decisions["feature_hints"]:
                print(f"[OrchestratorAgent] Feature hint: {hint}")

        return decisions
