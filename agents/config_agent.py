import os
import sys
import subprocess
import pandas as pd


class ConfigAgent:
    def __init__(self, competition_name: str, data_dir: str = "data/"):
        self.competition_name = competition_name
        self.data_dir = os.path.join(data_dir, competition_name) + "/"
        os.makedirs(self.data_dir, exist_ok=True)

    def _download_data(self):
        import zipfile
        train_path = os.path.join(self.data_dir, "train.csv")
        if os.path.exists(train_path):
            print(f"[ConfigAgent] Data already downloaded, skipping.")
            return
        kaggle_bin = os.path.join(os.path.dirname(sys.executable), "kaggle")
        subprocess.run([
            kaggle_bin, "competitions", "download",
            "-c", self.competition_name,
            "-p", self.data_dir,
        ], check=True)
        zip_path = os.path.join(self.data_dir, f"{self.competition_name}.zip")
        if os.path.exists(zip_path):
            print(f"[ConfigAgent] Extracting {self.competition_name}.zip...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.data_dir)

    def _find_submission_file(self) -> str:
        standard = os.path.join(self.data_dir, "sample_submission.csv")
        if os.path.exists(standard):
            return standard
        for f in os.listdir(self.data_dir):
            if "submission" in f.lower() and f.endswith(".csv"):
                return os.path.join(self.data_dir, f)
        raise FileNotFoundError("No submission file found in data directory.")

    def _detect_predict_type(self, sample_submission: pd.DataFrame, target_col: str) -> str:
        values = sample_submission[target_col].dropna().unique()
        if set(values).issubset({0, 1}):
            return "class"
        return "proba"

    def build_config(self) -> dict:
        print(f"[ConfigAgent] Detecting competition settings for: {self.competition_name}")
        print("[ConfigAgent] Downloading data...")
        self._download_data()
        print("[ConfigAgent] Download complete. Reading submission file...")

        sample_path = self._find_submission_file()
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")

        sample = pd.read_csv(sample_path)
        id_col = sample.columns[0]
        submission_col = sample.columns[1]
        predict_type = self._detect_predict_type(sample, submission_col)

        # target in train = columns in train but not in test
        train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
        test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
        extra = [c for c in train_cols if c not in test_cols and c != id_col]
        target_col = extra[0] if extra else submission_col

        print(f"[ConfigAgent] ID: {id_col} | Target: {target_col} | Submission col: {submission_col} | Predict: {predict_type}")
        # OrchestratorAgent will override n_trials, cv_folds, sample_size based on data profile.
        # These are just safe placeholder defaults.
        return {
            "competition": {
                "name": self.competition_name,
                "id_column": id_col,
                "target_column": target_col,
                "submission_column": submission_col,
                "task_type": "binary_classification",
                "metric": "accuracy" if predict_type == "class" else "roc_auc",
                "goal": "maximize",
                "predict_type": predict_type,
            },
            "model": {
                "algorithms": ["lightgbm"],
                "n_trials": 30,
                "cv_folds": 5,
                "sample_size": 50000,
            },
            "output": {
                "data_dir": self.data_dir,
                "models_dir": "models/",
                "submission_file": "submission.csv",
            }
        }
