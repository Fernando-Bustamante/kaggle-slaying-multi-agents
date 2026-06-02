import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold as _KFold


class FeatureAgent:
    def __init__(self, config: dict):
        self.config = config
        self.encoders = {}
        self._medians = None
        self._datetime_cols = []
        self._selected_features = []
        self._te_maps = {}
        self._split_cols = {}

    def _extract_title(self, series: pd.Series) -> pd.Series:
        """Extract token between ', ' and '.' for name-style columns (e.g. 'Braund, Mr. Owen' -> 'Mr')."""
        extracted = series.str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
        if extracted.notna().mean() > 0.5:
            return extracted
        # Fallback: first word
        return series.str.split().str[0]

    def add_delimiter_features(self, train: pd.DataFrame, test: pd.DataFrame, fit: bool = True) -> tuple:
        """Split structured string columns by consistent delimiters (e.g. 'B/0/P' → 3 parts)."""
        id_col = self.config["competition"]["id_column"]
        target_col = self.config["competition"]["target_column"]
        skip = {id_col, target_col}
        delimiters = ['/', '-', '_', '|', ':']

        if fit:
            self._split_cols = {}
            for col in train.select_dtypes(include=["object"]).columns:
                if col in skip:
                    continue
                # skip columns whose values are actually booleans stored as object
                if train[col].dropna().map(type).eq(bool).any():
                    continue
                best_delim, best_n_parts, best_score = None, 0, 0
                for delim in delimiters:
                    if not train[col].str.contains(delim, regex=False, na=False).mean() > 0.5:
                        continue
                    n_parts = train[col].dropna().str.split(delim).str.len()
                    mode_n = int(n_parts.mode()[0])
                    consistency = float((n_parts == mode_n).mean())
                    if consistency > 0.8 and mode_n >= 2 and consistency > best_score:
                        best_delim, best_n_parts, best_score = delim, mode_n, consistency

                if best_delim is None:
                    continue

                parts = train[col].str.split(best_delim, expand=True).iloc[:, :best_n_parts]
                if any(1 < parts[i].nunique() <= 200 for i in range(best_n_parts)):
                    self._split_cols[col] = (best_delim, best_n_parts)
                    print(f"[FeatureAgent] Detected delimiter '{best_delim}' in '{col}' → {best_n_parts} parts")

        for col, (delim, n_parts) in self._split_cols.items():
            for i in range(n_parts):
                new_col = f"{col}_p{i}"
                train[new_col] = train[col].str.split(delim).str[i].fillna("unknown")
                if col in test.columns:
                    test[new_col] = test[col].str.split(delim).str[i].fillna("unknown")

        return train, test

    def drop_high_cardinality(self, train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.5) -> tuple:
        id_col = self.config["competition"]["id_column"]
        target_col = self.config["competition"]["target_column"]
        exclude = [c for c in [id_col, target_col] if c in train.columns]
        dropped = []
        for col in train.select_dtypes(include=["object"]).columns:
            if col in exclude:
                continue
            if train[col].nunique() / len(train) > threshold:
                # Try to salvage a low-cardinality token before dropping
                extracted = self._extract_title(train[col])
                if extracted.nunique() <= 20:
                    new_col = f"{col}_title"
                    train[new_col] = extracted
                    test[new_col] = self._extract_title(test[col]) if col in test.columns else "unknown"
                    print(f"[FeatureAgent] Extracted '{new_col}' ({extracted.nunique()} unique) from high-cardinality '{col}'")
                dropped.append(col)
        if dropped:
            train = train.drop(columns=dropped)
            test = test.drop(columns=[c for c in dropped if c in test.columns])
            print(f"[FeatureAgent] Dropped {len(dropped)} high-cardinality columns: {dropped}")
        return train, test

    def clean(self, df: pd.DataFrame, fit: bool = True, impute_numeric: bool = True) -> pd.DataFrame:
        """
        fit=True  -> compute medians from this df (use for full train before final inference).
        impute_numeric=False -> skip numeric fillna so the CV loop can impute per-fold instead.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=["object"]).columns

        if fit:
            self._medians = df[numeric_cols].median()

        if impute_numeric:
            df[numeric_cols] = df[numeric_cols].fillna(self._medians)

        if len(object_cols) > 0:
            df[object_cols] = df[object_cols].fillna("unknown")

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        return df

    def add_datetime_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Auto-detect datetime string columns and extract time components.
        fit=True: detect which columns are datetime and store for reuse on test.
        fit=False: apply extraction only to previously detected columns.
        """
        id_col = self.config["competition"]["id_column"]
        target_col = self.config["competition"]["target_column"]
        exclude = [c for c in [id_col, target_col] if c in df.columns]

        if fit:
            self._datetime_cols = []
            for col in df.select_dtypes(include=["object"]).columns:
                if col in exclude:
                    continue
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    self._datetime_cols.append(col)

        if not self._datetime_cols:
            return df

        for col in self._datetime_cols:
            if col not in df.columns:
                continue
            parsed = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"]       = parsed.dt.year.astype(float).astype(np.float32)
            df[f"{col}_month"]      = parsed.dt.month.astype(float).astype(np.float32)
            df[f"{col}_day"]        = parsed.dt.day.astype(float).astype(np.float32)
            df[f"{col}_hour"]       = parsed.dt.hour.astype(float).astype(np.float32)
            df[f"{col}_dayofweek"]  = parsed.dt.dayofweek.astype(float).astype(np.float32)
            df[f"{col}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(np.float32)
            df = df.drop(columns=[col])

        print(f"[FeatureAgent] Extracted datetime features from: {self._datetime_cols}")
        return df

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.config["competition"]["id_column"]
        target_col = self.config["competition"]["target_column"]
        exclude = [c for c in [id_col, target_col] if c in df.columns]
        numeric_cols = df.drop(columns=exclude).select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 3:
            print(f"[FeatureAgent] Skipped statistical features ({len(numeric_cols)} numeric cols < 3)")
            return df
        arr = df[numeric_cols].values.astype(np.float32)
        new_cols = pd.DataFrame({
            'row_mean': np.nanmean(arr, axis=1),
            'row_std':  np.nanstd(arr, axis=1),
            'row_min':  np.nanmin(arr, axis=1),
            'row_max':  np.nanmax(arr, axis=1),
            'row_sum':  np.nansum(arr, axis=1),
        }, index=df.index, dtype=np.float32)
        df = pd.concat([df, new_cols], axis=1)
        del arr
        print(f"[FeatureAgent] Added 5 statistical features")
        return df

    def add_interaction_features(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        """Add pairwise ratio and product features.
        Only applied when there are <= 20 numeric feature columns (e.g. Titanic).
        Skipped for large/anonymous feature spaces (Santander, IEEE Fraud).
        """
        id_col = self.config["competition"]["id_column"]
        target_col = self.config["competition"]["target_column"]
        exclude = [c for c in [id_col, target_col] if c in train.columns]

        num_cols = train.drop(columns=exclude).select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 20:
            return train, test

        added = 0
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                ratio_col = f"{c1}_div_{c2}"
                prod_col  = f"{c1}_x_{c2}"
                train[ratio_col] = (train[c1] / train[c2].replace(0, np.nan)).astype(np.float32)
                test[ratio_col]  = (test[c1]  / test[c2].replace(0, np.nan)).astype(np.float32)
                train[prod_col]  = (train[c1] * train[c2]).astype(np.float32)
                test[prod_col]   = (test[c1]  * test[c2]).astype(np.float32)
                added += 2

        if added:
            print(f"[FeatureAgent] Added {added} interaction features ({len(num_cols)} numeric cols)")

        # Sum-of-members feature: columns whose names suggest family/group counts: columns whose names suggest family/group counts
        group_cols = [c for c in num_cols if any(k in c.lower() for k in ("sib", "parch", "family", "member", "group", "count"))]
        if len(group_cols) >= 2:
            train["group_size"] = train[group_cols].sum(axis=1) + 1
            test["group_size"]  = test[group_cols].sum(axis=1) + 1
            train["is_alone"]   = (train["group_size"] == 1).astype(np.float32)
            test["is_alone"]    = (test["group_size"] == 1).astype(np.float32)
            print(f"[FeatureAgent] Added group_size + is_alone from {group_cols}")

        return train, test

    def add_target_encoding(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        target_col = self.config["competition"]["target_column"]
        id_col = self.config["competition"]["id_column"]
        skip = {id_col, target_col}
        global_mean = float(train[target_col].mean())
        alpha = 10  # smoothing: pulls rare categories toward global mean

        cat_cols = [c for c in train.select_dtypes(include=["object"]).columns
                    if c not in skip and 2 < train[c].nunique() <= 200]
        if not cat_cols:
            return train, test

        self._te_maps = {}
        kf = _KFold(n_splits=5, shuffle=True, random_state=42)

        for col in cat_cols:
            stats = train.groupby(col)[target_col].agg(["mean", "count"])
            self._te_maps[col] = (
                (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
            ).to_dict()

            te_train = np.full(len(train), global_mean, dtype=np.float32)
            for tr_idx, val_idx in kf.split(train):
                fold_stats = train.iloc[tr_idx].groupby(col)[target_col].agg(["mean", "count"])
                fold_map = (
                    (fold_stats["count"] * fold_stats["mean"] + alpha * global_mean)
                    / (fold_stats["count"] + alpha)
                ).to_dict()
                te_train[val_idx] = train[col].iloc[val_idx].map(fold_map).fillna(global_mean).values

            train[f"{col}_te"] = te_train
            test[f"{col}_te"] = test[col].map(self._te_maps[col]).fillna(global_mean).astype(np.float32)

        print(f"[FeatureAgent] Target-encoded {len(cat_cols)} columns: {cat_cols}")
        return train, test

    def select_features(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        target_col = self.config["competition"]["target_column"]
        id_col = self.config["competition"]["id_column"]
        task_type = self.config["competition"].get("task_type", "binary_classification")
        drop_cols = [col for col in [target_col, id_col] if col in train.columns]

        X = train.drop(columns=drop_cols)
        y = train[target_col]
        n_cols = X.shape[1]

        pct = self.config.get("model", {}).get("feature_selection_pct")
        if pct is not None:
            k = max(int(n_cols * pct), 10)
        elif n_cols <= 20:
            k = n_cols
        elif n_cols <= 100:
            k = max(int(n_cols * 0.8), 10)
        else:
            k = max(int(n_cols * 0.5), 10)

        k = min(k, n_cols)

        X_for_sel = X.copy()
        if self._medians is not None:
            num_cols_sel = X_for_sel.select_dtypes(include=[np.number]).columns
            X_for_sel[num_cols_sel] = X_for_sel[num_cols_sel].fillna(self._medians)
        X_for_sel = X_for_sel.fillna(0)

        quick_params = dict(n_estimators=200, learning_rate=0.1, num_leaves=31,
                            random_state=42, verbose=-1, n_jobs=2)
        sel_model = lgb.LGBMRegressor(**quick_params) if task_type == "regression" else lgb.LGBMClassifier(**quick_params)
        sel_model.fit(X_for_sel, y)

        importances = pd.Series(
            sel_model.booster_.feature_importance(importance_type="gain"),
            index=X.columns,
        )
        self._selected_features = importances.nlargest(k).index.tolist()

        print(f"[FeatureAgent] Selected {k} of {n_cols} features via LightGBM gain importance")
        return self.apply_feature_selection(train, test)

    def apply_feature_selection(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        id_col = self.config["competition"]["id_column"]
        target_col = self.config["competition"]["target_column"]
        keep_train = self._selected_features + [c for c in [target_col, id_col] if c in train.columns]
        keep_test = self._selected_features + [id_col]
        train = train[[c for c in keep_train if c in train.columns]]
        test = test[[c for c in keep_test if c in test.columns]]
        print(f"[FeatureAgent] Applied saved selection: {len(self._selected_features)} features")
        return train, test

    def encode(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        id_col = self.config["competition"]["id_column"]
        target_col = self.config["competition"]["target_column"]
        skip = {id_col, target_col}
        for col in train.select_dtypes(include=["object"]).columns:
            if col in skip:
                continue
            le = LabelEncoder()
            le.fit(train[col].astype(str))
            train[col] = le.transform(train[col].astype(str))
            known = set(le.classes_)
            test[col] = test[col].astype(str).map(lambda v: v if v in known else le.classes_[0])
            test[col] = le.transform(test[col])
            self.encoders[col] = le
        print(f"[FeatureAgent] Encoded {len(self.encoders)} categorical columns")
        return train, test

    def split_features_target(self, train: pd.DataFrame) -> tuple:
        target = self.config["competition"]["target_column"]
        id_col = self.config["competition"]["id_column"]
        drop_cols = [col for col in [target, id_col] if col in train.columns]
        X = train.drop(columns=drop_cols)
        y = train[target]
        print(f"[FeatureAgent] Features: {X.shape[1]} columns | Target: {y.name}")
        return X, y
