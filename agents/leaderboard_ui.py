import subprocess
import tempfile
import zipfile
import os
from io import StringIO

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def _parse_csv_safe(text: str) -> pd.DataFrame | None:
    lines = [l for l in text.splitlines() if not l.startswith("Next Page Token")]
    try:
        return pd.read_csv(StringIO("\n".join(lines)))
    except Exception:
        return None


def _fetch_submissions(competition: str, kaggle_bin: str) -> list[dict]:
    result = subprocess.run(
        [kaggle_bin, "competitions", "submissions", "-c", competition, "--csv"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    df = _parse_csv_safe(result.stdout)
    if df is None or df.empty:
        return []
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


def _fetch_leaderboard(competition: str, kaggle_bin: str) -> pd.DataFrame | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [kaggle_bin, "competitions", "leaderboard", "-c", competition,
             "--download", "-p", tmpdir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        try:
            zip_path = os.path.join(tmpdir, f"{competition}.zip")
            with zipfile.ZipFile(zip_path, "r") as z:
                csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
                with z.open(csv_name) as f:
                    content = f.read().decode("utf-8-sig")
            return pd.read_csv(StringIO(content))
        except Exception:
            return None


def _lower_is_better(lb_df: pd.DataFrame, score_col: str) -> bool:
    rank_col = next((c for c in lb_df.columns if c.lower() == "rank"), None)
    if rank_col is None:
        return False
    rank1_score = lb_df[lb_df[rank_col] == lb_df[rank_col].min()][score_col].values
    rankN_score = lb_df[lb_df[rank_col] == lb_df[rank_col].max()][score_col].values
    if len(rank1_score) == 0 or len(rankN_score) == 0:
        return False
    return float(rank1_score[0]) < float(rankN_score[0])


def _score_bar(score: float, ref: float, lower_better: bool, width: int = 24) -> str:
    if ref == 0:
        return "░" * width
    ratio = (ref / score) if lower_better else (score / ref)
    ratio = min(ratio, 1.0)
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def show_leaderboard(competition: str, kaggle_bin: str) -> None:
    console.print(f"\n[bold cyan]Fetching leaderboard for [white]{competition}[/white]...[/bold cyan]")

    submissions = _fetch_submissions(competition, kaggle_bin)
    lb_df = _fetch_leaderboard(competition, kaggle_bin)

    completed = [s for s in submissions if s["score"] is not None]

    # ── Detect metric direction from leaderboard ──────────────────────
    lower_better = False
    score_col = None
    if lb_df is not None:
        score_col = next((c for c in lb_df.columns if "score" in c.lower()), None)
        if score_col:
            lower_better = _lower_is_better(lb_df, score_col)

    # Best score respects metric direction
    best_score = None
    if completed:
        scores = [s["score"] for s in completed]
        best_score = min(scores) if lower_better else max(scores)

    # ── Position table ────────────────────────────────────────────────
    if lb_df is None:
        console.print("[yellow]Public leaderboard unavailable — competition may have ended or be private.[/yellow]")

    if lb_df is not None and best_score is not None and score_col:
        total = len(lb_df)
        if lower_better:
            position = int((lb_df[score_col] < best_score).sum()) + 1
        else:
            position = int((lb_df[score_col] > best_score).sum()) + 1
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
        ref = min(s["score"] for s in completed) if lower_better else max(s["score"] for s in completed)
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
            bar = _score_bar(s["score"], ref, lower_better)
            hist_table.add_row(
                f"{s['score']:.5f}",
                f"[cyan]{bar}[/cyan]",
                s["status"],
                s["date"],
            )
        console.print(hist_table)
    else:
        console.print("[yellow]No completed submissions found.[/yellow]")
