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
        sep = "─" * 70

        # 1. Fetch your best public score from recent submissions
        best_score = None
        subs_result = subprocess.run(
            [kaggle_bin, "competitions", "submissions", "-c", competition],
            capture_output=True, text=True,
        )
        if subs_result.returncode == 0 and subs_result.stdout.strip():
            lines = subs_result.stdout.strip().split("\n")
            print(f"\n{sep}")
            print(f"  {competition} — your submissions")
            print(sep)
            for line in lines[:6]:
                print(f"  {line}")
            print(sep)
            for line in lines[1:]:
                parts = line.split()
                for part in parts:
                    try:
                        score = float(part)
                        if 0 < score <= 1:
                            if best_score is None or score > best_score:
                                best_score = score
                    except ValueError:
                        continue

        # 2. Fetch public leaderboard to compute rank and percentile
        print("\n[SubmissionAgent] Fetching public leaderboard...")
        lb_result = subprocess.run(
            [kaggle_bin, "competitions", "leaderboard", "-c", competition,
             "--show", "--csv"],
            capture_output=True, text=True,
        )
        if lb_result.returncode == 0 and lb_result.stdout.strip():
            try:
                from io import StringIO
                lb_df = pd.read_csv(StringIO(lb_result.stdout))
                total = len(lb_df)
                score_col = [c for c in lb_df.columns if "score" in c.lower()]
                if best_score is not None and score_col:
                    sc = score_col[0]
                    position = int((lb_df[sc] >= best_score).sum()) + 1
                    top_pct = round(position / total * 100, 1)
                    medal = "✅ TOP 20%" if top_pct <= 20 else "❌ not top 20% yet"
                    print(f"\n{sep}")
                    print(f"  YOUR POSITION : {position} / {total}")
                    print(f"  YOUR SCORE    : {best_score:.5f}")
                    print(f"  PERCENTILE    : top {top_pct}%  {medal}")
                    print(f"{sep}\n")
                else:
                    print(f"  Total participants: {total}")
            except Exception as e:
                print(f"[SubmissionAgent] Could not parse leaderboard: {e}")
        else:
            print("[SubmissionAgent] Could not fetch public leaderboard (private competition or API limit).")