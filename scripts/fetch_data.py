#!/usr/bin/env python3
"""Fetch the MotionSense dataset into ./data/motion-sense and unpack it.

The dataset is ~400 MB and is not committed. Run once:

    python scripts/fetch_data.py

The DeviceMotion CSVs ship inside `data/A_DeviceMotion_data.zip`; this
unpacks that archive in place, which is the layout `src.loader` expects.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "data" / "motion-sense"
URL = "https://github.com/mmalekzadeh/motion-sense.git"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not (DEST / ".git").is_dir():
        print(f"cloning {URL} -> {DEST}")
        subprocess.run(
            ["git", "clone", "--depth", "1", URL, str(DEST)], check=True
        )
    else:
        print(f"already cloned: {DEST}")

    archive = DEST / "data" / "A_DeviceMotion_data.zip"
    target = DEST / "data" / "A_DeviceMotion_data"
    if target.is_dir():
        print(f"already unpacked: {target}")
    else:
        print(f"unpacking {archive}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(DEST / "data")

    n = len(list(target.glob("*/sub_*.csv")))
    print(f"ok: {n} trial CSVs under {target}")
    print(f"\nexport MOTIONSENSE_ROOT={DEST / 'data'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
