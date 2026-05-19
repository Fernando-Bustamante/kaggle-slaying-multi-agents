import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelingAgent:
    def __init__(self, config: dict):
        self.config = config
        self.n_trials = config["model"]["n_trials"]
        self.cv_folds = config["model"]["cv_folds"]
        self.best_model = None
        self.best_score = 0

    def _cross_validate(self, params, X, y) -> float:
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(n_estimators=5000, **params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(-1),
                ]
            )
            preds = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)
            scores.append(score)
            print(f"  fold {fold}/{self.cv_folds} AUC: {score:.4f}", flush=True)
        return np.mean(scores)

    def _objective(self, X, y):
        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": 2,
            }
            score = self._cross_validate(params, X, y)

            if not hasattr(self._pbar, 'best') or score > self._pbar.best:
                self._pbar.best = score

            self._pbar.set_postfix({"AUC": f"{score:.4f}", "best": f"{self._pbar.best:.4f}"})
            self._pbar.update(1)
            return score
        return objective

    def tune(self, X, y):
        print("[ModelingAgent] Tuning LightGBM with sample...")
        self._pbar = tqdm(total=self.n_trials, desc="Otimizando LightGBM")
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective(X, y), n_trials=self.n_trials)
        self._pbar.close()
        self.best_score = study.best_value
        print(f"[ModelingAgent] Best AUC on sample: {self.best_score:.4f}")
        best_params = {**study.best_params, "random_state": 42, "verbose": -1, "n_jobs": 2}
        return best_params

    def train_final(self, X, y, best_params):
        print("[ModelingAgent] Training final model on full dataset...")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        self.best_model = lgb.LGBMClassifier(n_estimators=5000, **best_params)
        self.best_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=True),
                lgb.log_evaluation(100),
            ]
        )
        print("[ModelingAgent] Final model trained.")
        return self.best_model
