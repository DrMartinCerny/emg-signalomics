"""
emg_signalomics.load.nim — Medtronic NIM device loaders.

Modules
-------
eclipse : Medtronic NIM Eclipse Raw EMG CSV export loader.
"""

from emg_signalomics.load.nim.eclipse import head_eclipse, load_eclipse

__all__ = ["head_eclipse", "load_eclipse"]
