"""Shared helpers for the scripts."""
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("ZRL_CACHE") or ROOT / ".cache")
RESULTS = ROOT / "docs" / "data"
FIGURES = ROOT / "docs"
ZIG = shlex.split(os.environ.get("ZIG", "zig"))


def zrl(optimize="ReleaseFast"):
    """Builds zrl and returns the path of the executable."""
    subprocess.run(ZIG + ["build", f"-Doptimize={optimize}"], cwd=ROOT, check=True)
    return str(ROOT / "zig-out" / "bin" / "zrl")


def save_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1) + "\n")
