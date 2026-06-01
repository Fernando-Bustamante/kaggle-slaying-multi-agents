import gc
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelingAgent:
    def __init__(self, config: dict):
        self.config = config
        self.n_trials = config["model"]["n_trials"]
        self.cv_folds = config["model"]["cv_folds"]
        self.task_type = config["competition"].get("task_type", "binary_classification")
        self.metric = config["competition"].get("metric", "roc_auc")
        self.algorithm = config["model"].get("algorithm", "lgbm")
        self.best_score = 0

    def _get_cv(self):
        is_ts = self.config["competition"].get("is_timeseries", False)
        if is_ts:
            return TimeSeriesSplit(n_splits=self.cv_folds)
        if self.task_type == "regression":
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

    def _score_preds(self, preds, y_val) -> float:
        if self.task_type == "regression":
            if self.metric == "mae":
                return -mean_absolute_error(y_val, preds)
            return -np.sqrt(mean_squared_error(y_val, preds))
        elif self.task_type == "multiclass_classification":
            return roc_auc_score(y_val, preds, multi_class="ovr", average="macro")
        else:
            if self.metric == "accuracy":
                return accuracy_score(y_val, (preds >= 0.5).astype(int))
            if self.metric == "logloss":
                return -log_loss(y_val, preds)
            return roc_auc_score(y_val, preds)

    def _score(self, model, X_val, y_val) -> float:
        return self._score_preds(self._predict(model, X_val), y_val)

    def _score_diverse_fold(self, params, X_tr, y_tr, X_val, y_val) -> float:
        preds = []
        for factory in [
            lambda: self._make_lgb(params, 42),
            lambda: self._make_rf(params, 42),
            lambda: self._make_lr(params, 42),
        ]:
            m = factory()
            self._fit_model(m, X_tr, y_tr, X_val, y_val)
            preds.append(self._predict(m, X_val))
            del m
            gc.collect()
        return self._score_preds(np.mean(preds, axis=0), y_val)

    def _predict(self, model, X):
        if self.task_type == "regression":
            return model.predict(X)
        elif self.task_type == "multiclass_classification":
            return model.predict_proba(X)
        return model.predict_proba(X)[:, 1]

    def _make_lgb(self, params, seed):
        p = {**params, "random_state": seed}
        if self.task_type == "regression":
            return lgb.LGBMRegressor(n_estimators=5000, **p)
        return lgb.LGBMClassifier(n_estimators=5000, **p)

    def _make_xgb(self, params, seed):
        lr = params.get("learning_rate", 0.05)
        depth = min(int(params.get("max_depth", 6)), 10)
        subsample = params.get("subsample", 0.8)
        colsample = params.get("colsample_bytree", 0.8)
        reg_alpha = params.get("reg_alpha", 0.1)
        reg_lambda = params.get("reg_lambda", 1.0)
        common = dict(
            n_estimators=5000, learning_rate=lr, max_depth=depth,
            subsample=subsample, colsample_bytree=colsample,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            random_state=seed, n_jobs=-1, verbosity=0,
            early_stopping_rounds=100,
        )
        if self.task_type == "regression":
            return xgb.XGBRegressor(**common)
        if self.task_type == "multiclass_classification":
            return xgb.XGBClassifier(objective="multi:softprob", **common)
        return xgb.XGBClassifier(**common)

    def _make_cat(self, params, seed):
        lr = params.get("learning_rate", 0.05)
        depth = min(int(params.get("max_depth", 6)), 10)
        l2 = params.get("reg_lambda", 1.0)
        common = dict(
            iterations=5000, learning_rate=lr, depth=depth,
            l2_leaf_reg=l2, random_seed=seed,
            verbose=False, early_stopping_rounds=100,
        )
        if self.task_type == "regression":
            return cb.CatBoostRegressor(**common)
        if self.task_type == "multiclass_classification":
            return cb.CatBoostClassifier(loss_function="MultiClass", **common)
        return cb.CatBoostClassifier(loss_function="Logloss", **common)

    def _make_rf(self, params, seed):
        max_depth = min(int(params.get("max_depth", 8)), 12)
        # reuse min_child_samples as min_samples_leaf hint, scaled down for RF
        min_leaf = max(5, int(params.get("min_child_samples", 20)) // 3)
        common = dict(n_estimators=500, max_depth=max_depth, min_samples_leaf=min_leaf,
                      random_state=seed, n_jobs=-1)
        if self.task_type == "regression":
            return RandomForestRegressor(**common)
        return RandomForestClassifier(**common)

    def _make_lr(self, params, seed):
        # Pipeline ensures StandardScaler is fit per fold — no leakage
        if self.task_type == "regression":
            return Pipeline([("sc", StandardScaler()), ("lr", Ridge(alpha=1.0))])
        return Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=seed, n_jobs=2, solver="lbfgs")),
        ])

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val):
        if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        elif HAS_XGB and isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif HAS_CAT and isinstance(model, (cb.CatBoostClassifier, cb.CatBoostRegressor)):
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        else:
            # sklearn estimators (LogisticRegression, Ridge, RandomForest, …)
            model.fit(X_tr, y_tr)
        return model

    def _cross_validate(self, params, X, y) -> float:
        cv = self._get_cv()
        scores = []
        split_iter = cv.split(X, y) if self.task_type != "regression" else cv.split(X)
        for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
            X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            num_cols = X_tr.select_dtypes(include=[np.number]).columns
            fold_medians = X_tr[num_cols].median()
            X_tr[num_cols] = X_tr[num_cols].fillna(fold_medians)
            X_val[num_cols] = X_val[num_cols].fillna(fold_medians)
            if self.algorithm == "diverse":
                score = self._score_diverse_fold(params, X_tr, y_tr, X_val, y_val)
            else:
                model = self._make_lgb(params, seed=42)
                self._fit_model(model, X_tr, y_tr, X_val, y_val)
                score = self._score(model, X_val, y_val)
                del model
                gc.collect()
            scores.append(score)
            print(f"  fold {fold}/{self.cv_folds} score: {score:.4f}", flush=True)
        return np.mean(scores)

    def _objective(self, X, y):
        max_leaves = self.config["model"].get("max_num_leaves", 300)
        min_child_min = self.config["model"].get("min_child_samples_min", 10)

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, max_leaves),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", min_child_min, max(min_child_min + 10, 100)),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
            score = self._cross_validate(params, X, y)
            if not hasattr(self._pbar, "best") or score > self._pbar.best:
                self._pbar.best = score
            self._pbar.set_postfix({"score": f"{score:.4f}", "best": f"{self._pbar.best:.4f}"})
            self._pbar.update(1)
            return score
        return objective

    def tune(self, X, y):
        budget_min = self.config["model"].get("time_budget_minutes", 90)
        # Use 45% of total budget for tuning; rest reserved for final training
        tune_timeout = budget_min * 60 * 0.45
        print(f"[ModelingAgent] Tuning LightGBM (task={self.task_type}, metric={self.metric}, "
              f"max_trials={self.n_trials}, timeout={tune_timeout/60:.0f}min)...")
        self._pbar = tqdm(total=self.n_trials, desc="Optimizing LightGBM")
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective(X, y), n_trials=self.n_trials, timeout=tune_timeout)
        self._pbar.close()
        self.best_score = study.best_value
        actual_trials = len(study.trials)
        print(f"[ModelingAgent] Best CV score (tuning sample): {self.best_score:.4f} after {actual_trials} trials")
        return {**study.best_params, "random_state": 42, "verbose": -1, "n_jobs": -1}

    def _score_oof(self, oof_preds, y) -> float:
        if self.task_type == "regression":
            if self.metric == "mae":
                return -mean_absolute_error(y, oof_preds)
            return -np.sqrt(mean_squared_error(y, oof_preds))
        elif self.task_type == "multiclass_classification":
            return roc_auc_score(y, oof_preds, multi_class="ovr", average="macro")
        else:
            if self.metric == "accuracy":
                return accuracy_score(y, (oof_preds >= 0.5).astype(int))
            if self.metric == "logloss":
                return -log_loss(y, oof_preds)
            return roc_auc_score(y, oof_preds)

    def _build_candidates(self, best_params, n_ensemble):
        """Return the candidate list based on the algorithm strategy."""
        if self.algorithm == "diverse":
            # Small datasets: LR + RF generalize better than boosting alone.
            # LightGBM is included but de-prioritized — only enters if n_ensemble_models > 2.
            pool = [
                ("LogisticReg",   lambda: self._make_lr(best_params, 42)),
                ("RandomForest",  lambda: self._make_rf(best_params, 42)),
                ("LightGBM-42",   lambda: self._make_lgb(best_params, 42)),
                ("RandomForest-2",lambda: self._make_rf(best_params, 123)),
                ("LightGBM-123",  lambda: self._make_lgb(best_params, 123)),
            ]
        else:
            # Large datasets: LightGBM ensemble + optional XGB/CatBoost
            pool = [
                ("LightGBM-42",  lambda: self._make_lgb(best_params, 42)),
                ("LightGBM-123", lambda: self._make_lgb(best_params, 123)),
                ("LightGBM-456", lambda: self._make_lgb(best_params, 456)),
            ]
            if HAS_XGB:
                pool.append(("XGBoost",  lambda: self._make_xgb(best_params, 42)))
            if HAS_CAT:
                pool.append(("CatBoost", lambda: self._make_cat(best_params, 42)))
        return pool[:n_ensemble]

    def train_final(self, X, y, best_params, test_features):
        n_ensemble = self.config["model"].get("n_ensemble_models", 5)
        candidates = self._build_candidates(best_params, n_ensemble)
        print(f"[ModelingAgent] Training {n_ensemble} model(s) × {self.cv_folds} folds "
              f"(OOF CV, algorithm={self.algorithm})...")

        is_multi = self.task_type == "multiclass_classification"
        n_classes = len(np.unique(y)) if is_multi else None

        if is_multi:
            test_accum = np.zeros((len(test_features), n_classes))
            oof_accum = np.zeros((len(X), n_classes))
        else:
            test_accum = np.zeros(len(test_features))
            oof_accum = np.zeros(len(X))

        cv = self._get_cv()
        n_models = 0
        for name, factory in candidates:
            print(f"  [{name}] training...", flush=True)
            try:
                model_test = np.zeros_like(test_accum)
                model_oof = np.zeros_like(oof_accum)
                split_iter = cv.split(X, y) if self.task_type != "regression" else cv.split(X)
                for fold, (tr_idx, val_idx) in enumerate(split_iter, 1):
                    X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
                    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
                    num_cols = X_tr.select_dtypes(include=[np.number]).columns
                    fold_medians = X_tr[num_cols].median()
                    X_tr[num_cols] = X_tr[num_cols].fillna(fold_medians)
                    X_val[num_cols] = X_val[num_cols].fillna(fold_medians)
                    model = factory()
                    self._fit_model(model, X_tr, y_tr, X_val, y_val)
                    model_oof[val_idx] = self._predict(model, X_val)
                    model_test += self._predict(model, test_features) / self.cv_folds
                    print(f"    fold {fold}/{self.cv_folds}", flush=True)
                    del model
                    gc.collect()
                test_accum += model_test
                oof_accum += model_oof
                n_models += 1
                print(f"  [{name}] done.")
            except Exception as e:
                print(f"  [{name}] skipped: {e}")

        oof_final = oof_accum / n_models
        self.best_score = self._score_oof(oof_final, y)
        print(f"[ModelingAgent] OOF score (honest): {self.best_score:.4f} | {n_models} model(s) × {self.cv_folds} folds")
        return test_accum / n_models
