"""Read the cache: shard index, the crop to the scored window, jitter, session.

Copied from trial 1 and extended with one thing: every window also reports
which recording session it came from, because trial 4's session-adversarial
head needs that label. The crop arithmetic is the dangerous part and is
unchanged -- an off-by-one here trains happily and scores at chance.

The cache stores a WIDER window than the competition scores. `build_cache`
cuts (-0.32, 1.28) s at 250 Hz -- 400 samples -- while the model is scored on
(0.0, 1.0) s, which is 250 of them starting at sample 80. That margin exists
for exactly one reason: it makes time-jitter augmentation possible. Cropping a
250-sample window from a 250-sample cache would have nothing to slide over.

⚠️ Every offset here is DERIVED from the manifest, never hard-coded. If the
builder's bounds change, this follows; a literal 80 would not, and would fail
silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MANIFEST_NAME = "manifest.json"


def load_manifest(cache_dir):
    """Read the cache manifest written by `build_cache`.

    ARGS:
        cache_dir (Path or str): the directory holding the shards

    RETURNS:
        manifest (dict): as written by build_cache, including `shards`
    """
    path = Path(cache_dir) / MANIFEST_NAME

    #say what is missing and how to make it, rather than a bare FileNotFound
    if not path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {cache_dir}. Build the cache first:\n"
            f"    python -m pnpl2026.trial1.cli.build_cache"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def crop_geometry(manifest, tmin, tmax, jitter_seconds):
    """Work out where the scored window sits inside the cached one.

    ARGS:
        manifest (dict): the cache manifest, carrying `bounds` and sample rate
        tmin (float): start of the scored window, seconds from word onset
        tmax (float): end of the scored window, seconds from word onset
        jitter_seconds (float): requested one-sided jitter, in seconds

    RETURNS:
        n_model_samples (int): samples the model is given, T
        core_offset (int): index in the cached window where `tmin` sits
        max_jitter (int): samples the offset may move either way, clamped to
            what the cached margin can actually support
    """
    cache_start_s, cache_end_s = manifest["bounds"]
    sample_rate = manifest["sample_rate"]
    n_cached = manifest["n_samples_per_window"]

    #the scored window must lie inside what was cached, or there is nothing to cut
    if tmin < cache_start_s or tmax > cache_end_s:
        raise ValueError(
            f"Scored window ({tmin}, {tmax}) s falls outside the cached "
            f"({cache_start_s}, {cache_end_s}) s. Rebuild the cache with wider bounds."
        )

    n_model_samples = int(round((tmax - tmin) * sample_rate))

    #how far into the cached window word onset sits: 0.32 s -> 80 samples
    core_offset = int(round((tmin - cache_start_s) * sample_rate))

    #margin on each side, in samples, that the crop can slide into
    margin_before = core_offset
    margin_after = n_cached - core_offset - n_model_samples

    #a negative margin means the arithmetic above disagrees with the shard
    if margin_after < 0:
        raise ValueError(
            f"Crop of {n_model_samples} at offset {core_offset} overruns a "
            f"{n_cached}-sample cached window."
        )

    #never promise more jitter than the narrower margin can give
    requested = int(round(jitter_seconds * sample_rate))
    max_jitter = max(0, min(requested, margin_before, margin_after))

    return n_model_samples, core_offset, max_jitter


class CachedWindows(Dataset):
    """One split of the cache, cropped to the scored window on read.

    Shards stay on disk as memory maps: the full cache is ~18 GB at float16,
    which does not fit in RAM alongside a training process. Each __getitem__
    touches one window, so the OS page cache does the work.

    ⚠️ Memory maps are opened lazily, per worker. Opening them in __init__ and
    letting DataLoader fork the handles gives every worker the same file
    offsets and produces silently corrupted reads under some platforms.

    ATTRS:
        labels (np.ndarray): [n_windows] competition class ids for this split
        sessions (list): session id per window, parallel to `labels`
        n_model_samples (int): T, what the model is handed
        core_offset (int): where the scored window starts in the cached one
        max_jitter (int): one-sided jitter in samples, 0 outside training
    """

    def __init__(
        self,
        cache_dir,
        split,
        tmin,
        tmax,
        jitter_seconds=0.0,
        session_index=None,
    ):
        """Index one split of the cache without reading any window data.

        ARGS:
            cache_dir (Path or str): directory holding shards and the manifest
            split (str): "train" or "val", as recorded per shard by the builder
            tmin (float): start of the scored window in seconds
            tmax (float): end of the scored window in seconds
            jitter_seconds (float): one-sided jitter. Pass 0.0 for validation
            session_index (dict or None): session id -> adversary class index.
                A session missing from the mapping reports -1, which the
                training step reads as "no session label, skip the adversary"
        """
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.manifest = load_manifest(cache_dir)

        self.n_model_samples, self.core_offset, self.max_jitter = crop_geometry(
            self.manifest, tmin, tmax, jitter_seconds
        )

        #the split was decided by session at build time; honour it rather than
        #re-deriving it, or the two can disagree after a config edit
        shards = [s for s in self.manifest["shards"] if s["split"] == split]
        if not shards:
            raise ValueError(
                f"No shards marked split={split!r} in {cache_dir}. "
                f"Found: {sorted({s['split'] for s in self.manifest['shards']})}"
            )

        #labels are small enough to hold outright: 313k int16 is under a megabyte
        self.shard_keys = [s["key"] for s in shards]
        self._shard_of, self._row_of, labels, sessions = [], [], [], []
        for shard_index, shard in enumerate(shards):
            shard_labels = np.load(self.cache_dir / f"labels_{shard['key']}.npy")

            #the manifest count and the array must agree, or the index is wrong
            if len(shard_labels) != shard["n_windows"]:
                raise ValueError(
                    f"{shard['key']}: manifest says {shard['n_windows']} windows, "
                    f"labels file holds {len(shard_labels)}."
                )
            self._shard_of.append(np.full(len(shard_labels), shard_index, dtype=np.int32))
            self._row_of.append(np.arange(len(shard_labels), dtype=np.int32))
            labels.append(shard_labels)
            sessions.extend([shard["session"]] * len(shard_labels))

        self._shard_of = np.concatenate(self._shard_of)
        self._row_of = np.concatenate(self._row_of)
        self.labels = np.concatenate(labels).astype(np.int64)
        self.sessions = sessions

        #-1 for any session the mapping does not know. Validation sessions are
        #deliberately absent: the adversary is trained on the training split
        #only, and a val session forced into a train class index would be a lie
        mapping = session_index or {}
        self.session_ids = np.array(
            [mapping.get(s, -1) for s in sessions], dtype=np.int64
        )

        #opened on first read, in whichever process ends up doing the reading
        self._handles = None

    def __len__(self):
        return len(self.labels)

    def _windows_for(self, shard_index):
        """Return the memory-mapped window array for one shard, opening lazily.

        ARGS:
            shard_index (int): position in `self.shard_keys`

        RETURNS:
            windows : [n_windows, n_components, n_cached] float16 memmap
        """
        #one dict per process; workers each build their own on first access
        if self._handles is None:
            self._handles = {}
        if shard_index not in self._handles:
            key = self.shard_keys[shard_index]
            self._handles[shard_index] = np.load(
                self.cache_dir / f"windows_{key}.npy", mmap_mode="r"
            )
        return self._handles[shard_index]

    def __getitem__(self, index):
        """Read one window, cropped to the scored span.

        ARGS:
            index (int): position within this split

        RETURNS:
            window : [n_components, n_model_samples] float32
            label : scalar int64, a competition class id
            session : scalar int64, an adversary class index, or -1
        """
        windows = self._windows_for(int(self._shard_of[index]))

        #jitter shifts WHICH samples are cut, never pads -- the cached margin
        #is real recording, so a shifted crop is still real data
        offset = self.core_offset
        if self.max_jitter > 0:
            shift = int(torch.randint(-self.max_jitter, self.max_jitter + 1, (1,)).item())
            offset += shift

        #np.asarray forces the memmap slice into a real array before torch sees it
        window = np.asarray(
            windows[int(self._row_of[index]), :, offset : offset + self.n_model_samples]
        )

        #float16 out of the cache, float32 into the model: BatchNorm and the
        #whitened scale both want the headroom, and the copy is per-window cheap
        return (
            torch.from_numpy(window.astype(np.float32)),
            int(self.labels[index]),
            int(self.session_ids[index]),
        )
