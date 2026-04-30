# emg-signalomics

EMG / LFP / EEG signalomics — detection of periodic bursts and related features
in single-channel neurophysiological signals, plus loaders that bring vendor
CSV exports into MNE.

## Contents

- `emg_signalomics.detect.atrain` — **A-train** (afterdischarge train)
  detection based on sliding-window autocorrelation in a configurable
  frequency band (default 100–200 Hz).
- `emg_signalomics.load.{vendor}.{machine}` — vendor- and machine-specific
  CSV loaders that return an `mne.io.Raw`. Currently shipped:
  - `load.cadwell.cascade.load_cascade` — Cadwell Cascade / IOMax CSV.
  - `load.nim.eclipse.load_eclipse` — Medtronic NIM Eclipse (placeholder,
    raises `NotImplementedError`).

## Installation

From a local clone (editable install for development):

```bash
pip install -e .
```

Or directly from GitHub:

```bash
pip install git+https://github.com/DrMartinCerny/emg-signalomics.git
```

## Quick start — A-train detection

```python
import mne
from emg_signalomics.detect.atrain import detect_atrains

raw = mne.io.read_raw_fif("recording.fif", preload=True)

annotations = detect_atrains(raw)
raw.set_annotations(raw.annotations + annotations)

# Each annotation has description "ATRAIN_{channel_name}".
fp1_atrains = [a for a in raw.annotations if a["description"] == "ATRAIN_Fp1"]
```

## Quick start — loading vendor CSV

Loaders follow the pattern `emg_signalomics.load.{vendor}.{machine}` and each
exposes a `load_{machine}` function that returns a vanilla `mne.io.Raw`,
ready to be passed to the detection routines above or to any MNE workflow.

### Cadwell Cascade / IOMax

```python
from emg_signalomics.load.cadwell.cascade import load_cascade
from emg_signalomics.detect.atrain import detect_atrains

raw = load_cascade("Raw EMG Raw Data.csv")
raw.set_annotations(raw.annotations + detect_atrains(raw))
```

You can restrict which channels are loaded and cap how much data is read:

```python
raw = load_cascade(
    "Raw EMG Raw Data.csv",
    channels_to_keep=["EMG_CH01", "EMG_CH02"],
    max_samples=200_000,
)
```

Sample-index discontinuities in the source file are surfaced as standard MNE
`BAD_gap` annotations on the returned Raw.

### Medtronic NIM Eclipse

The Eclipse loader is a placeholder until the export format is wired up:

```python
from emg_signalomics.load.nim.eclipse import load_eclipse

raw = load_eclipse("eclipse_export.csv")  # raises NotImplementedError
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- pandas ≥ 1.3
- MNE-Python ≥ 1.0

## License

MIT — see `LICENSE`.
