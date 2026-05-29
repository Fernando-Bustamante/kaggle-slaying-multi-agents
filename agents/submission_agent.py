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

        # Show the user's own submissions (score + rank per submission)
        print("\n[SubmissionAgent] Fetching your submission scores...")
        result = subprocess.run(
            [kaggle_bin, "competitions", "submissions", "-c", competition],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            sep = "─" * 70
            print(f"\n{sep}")
            print(f"  {competition} — your submissions")
            print(sep)
            for line in lines[:6]:  # header + last 5
                print(f"  {line}")
            print(f"{sep}\n")
        else:
            # Fallback: show top of public leaderboard
            print("[SubmissionAgent] Could not fetch submissions, showing leaderboard top...")
            subprocess.run(
                [kaggle_bin, "competitions", "leaderboard", "-c", competition, "--show"],
                check=False,
            )