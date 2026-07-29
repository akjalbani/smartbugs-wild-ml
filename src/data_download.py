"""
Download the two datasets we need:

  1. smartbugs-results  -> the 32 MB results_wild.json that gives us labels
  2. smartbugs-wild     -> the 47,398 .sol contract source files (features)

The results repo is small and cloned in full. The wild-contracts repo is large
(hundreds of MB), so by default we do a *shallow* clone. If you only want to
experiment with a subset, set download_contracts=False and the feature step
will fetch just the files it needs over HTTPS (slower per-file but tiny total).

Everything is idempotent: re-running skips work that is already done.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

from . import config

RESULTS_REPO = "https://github.com/smartbugs/smartbugs-results.git"
WILD_REPO = "https://github.com/smartbugs/smartbugs-wild.git"


def _run(cmd: list[str]) -> None:
    """Run a shell command, streaming output, and raise on failure."""
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _clone(repo_url: str, dest: Path, shallow: bool = True) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {dest} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [repo_url, str(dest)]
    _run(cmd)


def download(download_contracts: bool = True) -> None:
    """Fetch labels (always) and, optionally, the full contract corpus."""
    print("Downloading smartbugs-results (labels) ...")
    _clone(RESULTS_REPO, config.DATA_DIR / "smartbugs-results", shallow=True)

    if download_contracts:
        print("Downloading smartbugs-wild (contract source) ...")
        print("  This is a few hundred MB and may take several minutes.")
        _clone(WILD_REPO, config.DATA_DIR / "smartbugs-wild", shallow=True)
    else:
        print("Skipping full contract download "
              "(features will fetch files on demand).")

    # Sanity checks
    if not config.RESULTS_JSON.exists():
        sys.exit(f"ERROR: expected {config.RESULTS_JSON} but it is missing.")
    print("Done. Labels are at:", config.RESULTS_JSON)


if __name__ == "__main__":
    download()
