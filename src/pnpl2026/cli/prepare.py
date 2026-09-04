"""Download / cache LibriBrain data.

Run on the HPC *login node* -- compute nodes have no outbound internet.

    uv run python -m pnpl2026.cli.prepare
"""

from __future__ import annotations

from pnpl.datasets import LibriBrain100
from pnpl.tasks import WordClassification

from ..config import DATA_DIR, Config


def main() -> None:
    cfg = Config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LibriBrain100(
        data_path=DATA_DIR,
        task=WordClassification(tmin=cfg.tmin, tmax=cfg.tmax),
        partition="train",
    )
    print(f"data ready under {DATA_DIR}")


if __name__ == "__main__":
    main()
