"""CLI smoke tests for documented run_pipeline.py flags."""

import subprocess
import sys
import os


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--help"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    assert result.returncode == 0
    assert "--demo-disagreement" in result.stdout
    assert "--force-disagreement" not in result.stdout
    assert "--no-llm" in result.stdout
    assert "--messiness" in result.stdout


def test_cli_no_llm_records_20(tmp_path, monkeypatch):
    repo = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env["AUDIT_DB_PATH"] = str(tmp_path / "audit.db")
    env["METRICS_PATH"] = str(tmp_path / "metrics.json")
    env["RESULTS_PATH"] = str(tmp_path / "results.json")
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--no-llm", "--records", "20", "--seed", "42"],
        capture_output=True, text=True, cwd=repo, env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_cli_demo_disagreement(tmp_path):
    repo = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env["AUDIT_DB_PATH"] = str(tmp_path / "audit.db")
    result = subprocess.run(
        [
            sys.executable, "run_pipeline.py",
            "--demo-disagreement", "--no-llm", "--records", "20", "--seed", "42",
        ],
        capture_output=True, text=True, cwd=repo, env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
