"""
Centralized version metadata for the repository.
"""

import subprocess

application_version = "1.0.0"
model_version = "v1.0.0"
build_date = "2026-07-16"


def get_git_commit() -> str:
    """Retrieve the current Git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception as e:
        import logging

        logging.getLogger("version").debug(f"Failed to get git commit: {e}")
        return "unknown"


git_commit = get_git_commit()
