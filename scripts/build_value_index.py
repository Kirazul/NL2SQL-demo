"""Build the value index from eicu.db.

Skips itself when the index is already there and populated: on Kaggle the
preparation notebook is re-run on every saved version, and rebuilding an index
that has not changed is minutes spent to produce the same file.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridsql.config import settings  # noqa: E402
from hybridsql.db.value_index import build  # noqa: E402


def already_built(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        n = cx.execute("SELECT COUNT(*) FROM values_fts").fetchone()[0]
        cx.close()
    except sqlite3.Error:
        return False
    return n > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    arguments = parser.parse_args()

    destination = settings().value_index_path
    if not arguments.force and already_built(destination):
        print(f"Index already built: {destination} "
              f"({destination.stat().st_size / 1e6:.1f} MB). --force to rebuild.")
    else:
        build()
