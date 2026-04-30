"""
emg_signalomics.load — vendor- and machine-specific loaders for multichannel
EMG recordings.

Layout
------
The package is organised as ``emg_signalomics.load.{vendor}.{machine}`` so
adding a new device only adds another leaf module.

Currently shipped
-----------------
load.cadwell.cascade   : Cadwell Cascade / IOMax CSV exports.
load.nim.eclipse       : Medtronic NIM Eclipse — placeholder, not yet implemented.

Each leaf module exposes a ``load_{machine}`` function that returns an
``mne.io.Raw`` object so downstream code (e.g.
``emg_signalomics.detect.atrain.detect_atrains``) can consume the result
uniformly regardless of vendor.
"""
