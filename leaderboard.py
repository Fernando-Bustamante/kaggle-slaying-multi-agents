import sys
import os
from agents.leaderboard_ui import show_leaderboard

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaderboard.py <competition-name>")
        sys.exit(1)
    competition = sys.argv[1]
    kaggle_bin = os.path.join(os.path.dirname(sys.executable), "kaggle")
    show_leaderboard(competition, kaggle_bin)
