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
  - `load.nim.eclipse.load_eclipse` — Medtronic NIM Eclipse "Raw EMG" CSV.

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

```python
from emg_signalomics.load.nim.eclipse import load_eclipse

raw = load_eclipse("18.csv")
```

The Eclipse "Raw EMG" export is much less tidy than the Cadwell one — its
header is free-form text, the column header wraps across multiple physical
lines, and each per-second per-channel "trace" is serialised as a single
quoted CSV cell whose contents are themselves a CSV record (with embedded
escaped quotes and a 32 KiB hard-wrap to extra physical lines).  The loader
re-glues those pieces, recovers each channel from the `Tr Name` field,
infers `sfreq` from the per-epoch sample count and the spacing between
consecutive `Date Time` stamps, and returns a regular `mne.io.Raw`.  Filter
cutoffs are taken from `LFF` / `HFF`; non-contiguous epochs become standard
`BAD_gap` annotations.

The Eclipse format does not declare units; samples are assumed to be
microvolts (the typical Eclipse default for Raw EMG) and the returned Raw
is in Volts as MNE expects.

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- pandas ≥ 1.3
- MNE-Python ≥ 1.0

## License

MIT — see `LICENSE`.
