# emg_signalomics/load/nim/eclipse.py
"""
Medtronic NIM Eclipse CSV -> MNE Raw loader.

Placeholder module — the export format for the NIM Eclipse has not been wired
up yet.  The public entry point :func:`load_eclipse` raises
``NotImplementedError`` so callers fail loudly until support is added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import mne


def load_eclipse(
    path: Union[str, Path],
    *,
    channels_to_keep: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
) -> "mne.io.Raw":
    """
    Load a Medtronic NIM Eclipse CSV export into an MNE Raw object.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the CSV file.
    channels_to_keep : sequence of str, optional
        Restrict loading to these channel names.
    max_samples : int, optional
        If given, only the first ``max_samples`` waveform rows are loaded.

    Returns
    -------
    mne.io.Raw
        Preloaded Raw object with EMG channels in Volts.

    Raises
    ------
    NotImplementedError
        Always — the NIM Eclipse export format is not yet supported.  See
        :func:`emg_signalomics.load.cadwell.cascade.load_cascade` for the
        intended call signature and behaviour.
    """
    raise NotImplementedError(
        "load_eclipse is not implemented yet. "
        "Contributions welcome — see load_cascade in "
        "emg_signalomics.load.cadwell.cascade for the reference implementation."
    )
