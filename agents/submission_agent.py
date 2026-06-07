import os
import sys
import subprocess
import pandas as pd
from dotenv import load_dotenv
from agents.config_agent import _find_kaggle_bin

load_dotenv()


class SubmissionAgent:
    def __init__(self, config: dict):
        self.config = config
        self.submission_file = config["output"]["submission_file"]

    def create_submission(self, test_ids: pd.Series, predictions) -> str:
      id_col = self.config["competition"]["id_column"]
      submission_col = self.config["competition"].get("submission_column") or self.config["competition"]["target_column"]
      submission = pd.DataFrame({
          id_col: test_ids,
          submission_col: predictions
      })
      submission.to_csv(self.submission_file, index=False)
      print(f"[SubmissionAgent] Saved submission to {self.submission_file}")
      return self.submission_file

    def submit(self, message: str = "Automated submission by multi-agent system"):
        competition = self.config["competition"]["name"]
        kaggle_bin = _find_kaggle_bin()
        print(f"[SubmissionAgent] Submitting to {competition}...")
        
        
        subprocess.run([
            kaggle_bin, "competitions", "submit",
            "-c", competition,
            "-f", self.submission_file,
            "-m", message
        ], check=True)
        
        print("[SubmissionAgent] Submission complete.")

    def check_leaderboard(self):
        from agents.leaderboard_ui import show_leaderboard
        competition = self.config["competition"]["name"]
        kaggle_bin = _find_kaggle_bin()
        show_leaderboard(competition, kaggle_bin)