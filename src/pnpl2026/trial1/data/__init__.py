"""Trial-1 data access: the cached-window dataset and its DataModule.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED.
"""

from __future__ import annotations

from .cache_dataset import CachedWindows, crop_geometry, load_manifest
from .datamodule import Trial1DataModule, class_weights_from

__all__ = [
    "CachedWindows",
    "crop_geometry",
    "load_manifest",
    "Trial1DataModule",
    "class_weights_from",
]
