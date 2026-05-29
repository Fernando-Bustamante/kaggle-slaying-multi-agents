import os
import sys
import subprocess
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class DataAgent:
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = config["output"]["data_dir"]
        os.makedirs(self.data_dir, exist_ok=True)

    def download_data(self) -> str:
        train_path = os.path.join(self.data_dir, "train.csv")
        if os.path.exists(train_path):
            print(f"[DataAgent] Data already exists, skipping download.")
            return self.data_dir
        competition = self.config["competition"]["name"]
        kaggle_bin = os.path.join(os.path.dirname(sys.executable), "kaggle")
        print(f"[DataAgent] Downloading data for: {competition}")
        subprocess.run([
            kaggle_bin, "competitions", "download",
            "-c", competition,
            "-p", self.data_dir,
        ], check=True)
        print(f"[DataAgent] Data saved to {self.data_dir}")
        return self.data_dir

    def load_data(self, sample: int = None) -> tuple:
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")

        print(f"[DataAgent] Reading train.csv (may take a few minutes for large files)...")
        train = pd.read_csv(train_path)
        print(f"[DataAgent] Reading test.csv...")
        test = pd.read_csv(test_path)

        if sample is not None and sample < len(train):
            train = train.sample(n=sample, random_state=42).reset_index(drop=True)

        print(f"[DataAgent] Train shape: {train.shape}")
        print(f"[DataAgent] Test shape: {test.shape}")
        return train, test

    def analyze(self, df: pd.DataFrame) -> dict:
        target = self.config["competition"]["target_column"]
        report = {
            "shape": df.shape,
            "missing_values": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "target_distribution": df[target].value_counts().to_dict() if target in df.columns else {}
        }
        print(f"[DataAgent] Shape: {report['shape']} | Missing cols: {len(report['missing_values'])}")
        return report
