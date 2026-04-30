"""
emg_signalomics.load.nim — Medtronic NIM device loaders.

Modules
-------
eclipse : Medtronic NIM Eclipse loader (not yet implemented).
"""

from emg_signalomics.load.nim.eclipse import load_eclipse

__all__ = ["load_eclipse"]
