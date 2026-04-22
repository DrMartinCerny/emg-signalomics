"""
emg_signalomics.detect — burst- and event-detection routines.

Modules
-------
atrain : A-train (sustained HFO burst) detection via sliding-window autocorrelation.
"""

from emg_signalomics.detect.atrain import detect_atrains

__all__ = ["detect_atrains"]
