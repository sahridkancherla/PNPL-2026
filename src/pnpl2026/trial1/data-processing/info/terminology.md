# Key Terminology & Shapes Reference

Quick reference for variables, tensor dimensions, and axis names used across data processing and modeling.

---

## Master Terminology Table

| Term | Typical Shape / Type | What it represents | Example / Notes |
|---|---|---|---|
| **`recording`** | `[n_channels, n_times]` | A full, continuous MEG recording file. | Raw continuous sensor session. |
| **`windows`** | `[n_windows, n_channels, n_samples]` | 3D array of epoched word trials cut from a continuous recording. | The core dataset array saved to disk and fed into models. |
| **`n_windows`** | `int` (Axis 0 of `windows`) | Total number of word trials in a recording or shard. | e.g. `500` trials. In PyTorch training batches, this axis is the `batch_size` (e.g. `32`). |
| **`n_channels`** | `int` (Axis 0 of `recording`, Axis 1 of `windows`) | Number of sensor channels or spatial features. | `306` for raw MEG sensors. |
| **`n_components`** | `int` (Replaces `n_channels` after whitening) | Number of spatial dimensions kept after PCA / whitening. | `73` components. Sits on the channel axis once whitened. |
| **`n_times`** | `int` (Axis 1 of `recording`) | Total discrete sample count of the **entire continuous session**. | e.g. `150,000` samples ($10\text{ min} \times 250\text{ Hz}$). **Integer sample count, not seconds.** |
| **`n_samples`** | `int` (Axis 2 of `windows`) | Number of time points in **one single word window**. | e.g. `400` samples ($1.6\text{s} \times 250\text{ Hz}$). |
| **`sample_rate`** | `float` (Hz) | Sampling frequency of the data. | `250.0` Hz ($1\text{ sample} = 4\text{ ms}$). |
| **`bounds`** | `tuple[float, float]` (seconds) | Window start and end relative to word onset. | `(-0.32, 1.28)` seconds ($1.6\text{s}$ total). |
| **`onsets_s`** | `np.ndarray [n_events]` (float) | Timestamps (in seconds) where words begin in the recording. | Taken from the `timeds` column of the events TSV. |
| **`window_start`** | `np.ndarray [n_events]` (int) | Exact sample index (column) in `recording` where each word window starts. | Calculated as: `(onsets_s + start_s) * sample_rate`. |
| **`labels`** | `np.ndarray [n_windows]` (int) | The target word IDs to predict. | Integer IDs from `0` to `49` (the 50-word competition vocabulary). |

---

## Quick Clarifications

* **`n_times` vs `n_samples`**:
  * **`n_times`** = length of the **whole continuous recording** (hundreds of thousands of points).
  * **`n_samples`** = length of **one word's window** (e.g. 400 points).

* **`n_channels` vs `n_components`**:
  * **`n_channels`** = raw sensor count (`306`).
  * **`n_components`** = spatial components after whitening (`73`). Both occupy Axis 1 of `windows`.

* **Model inputs (`forward`)**:
  * Model expects `windows` of shape `[n_windows, n_channels, n_samples]` (or `[batch_size, n_channels, n_samples]`).
