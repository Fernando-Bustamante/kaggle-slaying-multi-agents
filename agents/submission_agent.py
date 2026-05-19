import os
import sys
import subprocess
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class SubmissionAgent:
    def __init__(self, config: dict):
        self.config = config
        self.submission_file = config["output"]["submission_file"]

    def create_submission(self, test: pd.DataFrame, predictions) -> str:
      id_col = self.config["competition"]["id_column"]
      target_col = self.config["competition"]["target_column"]
      submission = pd.DataFrame({
          id_col: test[id_col],
          target_col: predictions
      })
      submission.to_csv(self.submission_file, index=False)
      print(f"[SubmissionAgent] Saved submission to {self.submission_file}")
      return self.submission_file

    def submit(self, message: str = "Automated submission by multi-agent system"):
        competition = self.config["competition"]["name"]
        kaggle_bin = os.path.join(os.path.dirname(sys.executable), "kaggle")
        print(f"[SubmissionAgent] Submitting to {competition}...")
        
        
        subprocess.run([
            kaggle_bin, "competitions", "submit",
            "-c", competition,
            "-f", self.submission_file,
            "-m", message
        ], check=True)
        
        print("[SubmissionAgent] Submission complete.")

    def check_leaderboard(self):
        competition = self.config["competition"]["name"]
        kaggle_bin = os.path.join(os.path.dirname(sys.executable), "kaggle")
        print("[SubmissionAgent] Checking leaderboard...")
        
        #status
        subprocess.run([
            kaggle_bin, "competitions", "leaderboard",
            "-c", competition,
            "--show"
        ], check=True)