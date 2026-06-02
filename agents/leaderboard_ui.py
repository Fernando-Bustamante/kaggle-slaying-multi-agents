import subprocess
from io import StringIO
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def _fetch_submissions(competition: str, kaggle_bin: str) -> list[dict]:
    result = subprocess.run(
        [kaggle_bin, "competitions", "submissions", "-c", competition, "--csv"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        df = pd.read_csv(StringIO(result.stdout))
        rows = []
        for _, row in df.iterrows():
            score_val = None
            for col in df.columns:
                if "public" in col.lower() or "score" in col.lower():
                    try:
                        v = float(row[col])
                        if 0 < v <= 1:
                            score_val = v
                            break
                    except (ValueError, TypeError):
                        continue
            date_val = ""
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    date_val = str(row[col])[:16]
                    break
            status = ""
            for col in df.columns:
                if "status" in col.lower():
                    status = str(row[col]).replace("SubmissionStatus.", "")
                    break
            rows.append({"score": score_val, "date": date_val, "status": status})
        return rows
    except Exception:
        return []


def _fetch_leaderboard(competition: str, kaggle_bin: str) -> pd.DataFrame | None:
    result = subprocess.run(
        [kaggle_bin, "competitions", "leaderboard", "-c", competition, "--show", "--csv"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        # Skip "Next Page Token = ..." line that kaggle CLI prepends
        lines = [l for l in result.stdout.splitlines() if not l.startswith("Next Page Token")]
        return pd.read_csv(StringIO("\n".join(lines)))
    except Exception:
        return None


def _score_bar(score: float, max_score: float, width: int = 24) -> str:
    if max_score == 0:
        return "░" * width
    filled = int(score / max_score * width)
    return "█" * filled + "░" * (width - filled)


def show_leaderboard(competition: str, kaggle_bin: str) -> None:
    console.print(f"\n[bold cyan]Fetching leaderboard for [white]{competition}[/white]...[/bold cyan]")

    submissions = _fetch_submissions(competition, kaggle_bin)
    lb_df = _fetch_leaderboard(competition, kaggle_bin)

    best_score = None
    completed = [s for s in submissions if s["score"] is not None]
    if completed:
        best_score = max(s["score"] for s in completed)

    # ── Position table ────────────────────────────────────────────────
    if lb_df is not None and best_score is not None:
        total = len(lb_df)
        score_col = next((c for c in lb_df.columns if "score" in c.lower()), None)
        if score_col:
            position = int((lb_df[score_col] >= best_score).sum()) + 1
            top_pct = round(position / total * 100, 1)

            pos_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
            pos_table.add_column(style="bold yellow", no_wrap=True)
            pos_table.add_column(style="white")
            pos_table.add_row("Competition", competition)
            pos_table.add_row("Position", f"{position} / {total}")
            pos_table.add_row("Top %", f"{top_pct}%")
            pos_table.add_row("Best Score", f"{best_score:.5f}")
            console.print(pos_table)

    # ── Submission history ────────────────────────────────────────────
    if completed:
        max_s = max(s["score"] for s in completed)
        hist_table = Table(
            title="Recent Submissions",
            box=box.SIMPLE_HEAD,
            show_lines=False,
            padding=(0, 1),
        )
        hist_table.add_column("Score", style="bold green", justify="right", no_wrap=True)
        hist_table.add_column("", no_wrap=True)
        hist_table.add_column("Status", style="dim", no_wrap=True)
        hist_table.add_column("Date", style="dim", no_wrap=True)

        for s in completed[:5]:
            bar = _score_bar(s["score"], max_s)
            hist_table.add_row(
                f"{s['score']:.5f}",
                f"[cyan]{bar}[/cyan]",
                s["status"],
                s["date"],
            )
        console.print(hist_table)
    else:
        console.print("[yellow]No completed submissions found.[/yellow]")
