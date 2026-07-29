"""
Central configuration for the SmartBugs-Wild ML pipeline.

Everything that a user might reasonably want to change (paths, thresholds,
model hyper-parameters, the random seed) lives here so the rest of the code
never hard-codes a magic number.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# A single global seed is used everywhere (numpy, torch, sklearn) so that two
# runs on the same machine produce the same numbers. Change it if you want to
# measure run-to-run variance.
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# On Google Colab the default working directory is /content. Locally it is the
# repository root. We resolve paths relative to this file so the code works in
# both places without edits.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CONTRACTS_DIR = DATA_DIR / "smartbugs-wild" / "contracts"      # <address>.sol files
RESULTS_JSON = DATA_DIR / "smartbugs-results" / "metadata" / "results_wild.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"                     # outputs: labels, models, metrics

for _d in (DATA_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# The DASP-10 vulnerability categories used by SmartBugs.
# --------------------------------------------------------------------------- #
# NOTE: SmartBugs labels contracts with the DASP taxonomy, NOT the newer
# OWASP Smart Contract Top-10. They overlap heavily but are not identical.
# We keep the DASP names here because those are what actually appear in the
# data. docs/METHODOLOGY.md explains the mapping to OWASP.
#
# We deliberately DROP two categories:
#   * short_addresses  - not produced by any tool on this dataset
#   * bad_randomness    - produced for 0 contracts on this dataset (verified)
# Training a classifier for a class with zero positives is meaningless, so we
# exclude them rather than silently report a perfect-looking but empty score.
DASP_CATEGORIES = [
    "access_control",
    "arithmetic",
    "denial_service",
    "reentrancy",
    "unchecked_low_calls",
    "front_running",
    "time_manipulation",
]

# The nine analysis tools whose outputs form our weak labels.
TOOLS = [
    "mythril", "slither", "oyente", "osiris", "smartcheck",
    "manticore", "maian", "securify", "honeybadger",
]

# --------------------------------------------------------------------------- #
# Weak-labelling
# --------------------------------------------------------------------------- #
# A contract is labelled positive for a category when at least CONSENSUS_K of
# the nine tools flag that category. K=2 filters out a lot of single-tool
# noise; K=1 keeps everything (very noisy). See docs/METHODOLOGY.md.
CONSENSUS_K = 2

# --------------------------------------------------------------------------- #
# Sampling / splits
# --------------------------------------------------------------------------- #
# The full dataset is ~47k contracts. Start SMALL so a first run finishes in a
# few minutes on a free Colab CPU, then raise this (or set to None for all).
MAX_CONTRACTS = 6000
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # taken out of the training portion

# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
SEQ_MAX_LEN = 400        # token sequence length for the CNN
VOCAB_SIZE = 8000        # vocabulary cap for the CNN / GNN

# --------------------------------------------------------------------------- #
# Neural-net training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 1e-3
EMBED_DIM = 64
