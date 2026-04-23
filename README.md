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
import mne
from emg_signalomics.detect.atrain import detect_atrains

raw = mne.io.read_raw_fif("recording.fif", preload=True)

annotations = detect_atrains(raw)
raw.set_annotations(raw.annotations + annotations)

# Each annotation has description "ATRAIN_{channel_name}".
# Filter by channel:
fp1_atrains = [a for a in raw.annotations if a["description"] == "ATRAIN_Fp1"]
```

See the docstring of `detect_atrains` for the full list of tunable parameters.

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- MNE-Python ≥ 1.0

## License

MIT — see `LICENSE`.