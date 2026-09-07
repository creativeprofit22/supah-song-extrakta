"""Fetch the pinned inference code and public Cinematic Bandit v2 checkpoint."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

from .media import run

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / ".cache" / "msst"
REVISION = "0e5f1159fc5ea87fc13b957584e178b4977e5dd3"
WEIGHTS = ROOT / "models" / "checkpoint-multi_fixed.ckpt"
WEIGHTS_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/checkpoint-multi_fixed.ckpt"
WEIGHTS_SIZE = 149133378
# Observed HTTPS download hash; not an independently signed publisher digest.
WEIGHTS_SHA256 = "20bcd513dc7eb0541dd045909a4e7dff8dab474cc2efba4904101c76524aee85"


def validate_resources() -> None:
    revision = run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture=True).stdout.strip()
    changes = run(["git", "-C", str(CODE), "status", "--porcelain", "--untracked-files=no"], capture=True).stdout
    if revision != REVISION or changes:
        raise ValueError("Inference checkout differs from the pinned source; refusing to import it.")
    if not WEIGHTS.is_file() or WEIGHTS.stat().st_size != WEIGHTS_SIZE:
        raise ValueError("Missing or incomplete model. Run: python -m songtool setup-model")
    with WEIGHTS.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != WEIGHTS_SHA256:
            raise ValueError("Model checksum mismatch; refusing to load it.")


def setup() -> None:
    if not CODE.exists():
        CODE.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", "https://github.com/ZFTurbo/Music-Source-Separation-Training.git", str(CODE)])
        run(["git", "-C", str(CODE), "checkout", "--detach", REVISION])
    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        partial = WEIGHTS.with_suffix(".download")
        # A failed download remains visibly partial; never load it as a checkpoint.
        with urlopen(WEIGHTS_URL, timeout=60) as response, partial.open("wb") as output:
            remaining = WEIGHTS_SIZE
            while block := response.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(block)
                if remaining < 0:
                    raise ValueError("Checkpoint response exceeds its expected size.")
                output.write(block)
        if partial.stat().st_size != WEIGHTS_SIZE:
            raise ValueError("Incomplete checkpoint download.")
        partial.rename(WEIGHTS)
    validate_resources()
    with WEIGHTS.open("rb") as source:
        print("Model SHA256:", hashlib.file_digest(source, "sha256").hexdigest())
