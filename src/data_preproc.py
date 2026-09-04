import sys
from pathlib import Path

import torch
import pnpl
from pnpl.datasets import LibriBrain100
from pnpl.tasks import WordDetection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TMIN, TMAX, SFREQ = 0.0, 1.0, 250

DATA_PATH = Path("./data")

train_raw = LibriBrain100(
    data_path=DATA_PATH,
    task=WordDetection(tmin=TMIN, tmax=TMAX),
    partition="train",
)
