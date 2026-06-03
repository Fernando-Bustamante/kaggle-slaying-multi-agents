import sys
import os
import shutil
from agents.leaderboard_ui import show_leaderboard


def _find_kaggle_bin() -> str:
    # Try venv-local first, then system PATH
    venv_path = os.path.join(os.path.dirname(sys.executable), "kaggle")
    if os.path.isfile(venv_path):
        return venv_path
    system_path = shutil.which("kaggle")
    if system_path:
        return system_path
    raise FileNotFoundError("kaggle CLI not found. Install with: pip install kaggle")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaderboard.py <competition-name>")
        sys.exit(1)
    competition = sys.argv[1]
    kaggle_bin = _find_kaggle_bin()
    show_leaderboard(competition, kaggle_bin)
