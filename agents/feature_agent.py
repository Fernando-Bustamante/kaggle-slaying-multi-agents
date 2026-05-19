import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class FeatureAgent:
    def __init__(self, config: dict):
        self.config = config
        self.encoders = {}
        self._medians = None

    def clean(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=["object"]).columns

        if fit:
            self._medians = df[numeric_cols].median()

        df[numeric_cols] = df[numeric_cols].fillna(self._medians)

        if len(object_cols) > 0:
            df[object_cols] = df[object_cols].fillna("unknown")

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        return df

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.config["competition"]["id_column"]
        target_col = self.config["competition"]["target_column"]
        exclude = [c for c in [id_col, target_col] if c in df.columns]
        numeric_cols = df.drop(columns=exclude).select_dtypes(include=[np.number]).columns
        arr = df[numeric_cols].values.astype(np.float32)
        new_cols = pd.DataFrame({
            'row_mean': arr.mean(axis=1),
            'row_std':  arr.std(axis=1),
            'row_min':  arr.min(axis=1),
            'row_max':  arr.max(axis=1),
            'row_sum':  arr.sum(axis=1),
        }, index=df.index, dtype=np.float32)
        df = pd.concat([df, new_cols], axis=1)
        del arr
        print(f"[FeatureAgent] Added 5 statistical features")
        return df

    def encode(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        for col in train.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col]]).astype(str)
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
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
