# emg-signalomics

EMG / LFP / EEG signalomics — detection of periodic bursts and related features
in single-channel neurophysiological signals.

## Contents

- `emg_signalomics.detect.atrain` — **A-train** (afterdischarge train)
  detection based on sliding-window autocorrelation in a configurable
  frequency band (default 100–200 Hz).

## Installation

From a local clone (editable install for development):

```bash
pip install -e .
```

Or directly from GitHub:

```bash
pip install git+https://github.com/DrMartinCerny/emg-signalomics.git
```

## Quick start

```python
import numpy as np
from emg_signalomics.detect.atrain import detect_atrains

# signal_uv: 1-D bandpassed signal in microvolts
# fs:        sampling rate in Hz
result = detect_atrains(signal_uv, fs)

atrain_mask = result["atrains"]    # (T,) bool — final A-train mask
silence     = result["silence"]    # (T,) bool — silence (low-RMS) mask
peak_ac     = result["peak_ac"]    # (T,) float — per-sample peak autocorrelation
```

See the docstring of `detect_atrains` for the full list of tunable
parameters and output arrays.

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20

## License

MIT — see `LICENSE`.
