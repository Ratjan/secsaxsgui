# SEC-SAXS
Utilities for manipulating SEC-SAXS data

## SEC-SAXS GUI (`sec_saxs_gui.py`)

Desktop program for buffer subtraction of SEC-SAXS frames.

```
pip install -r requirements.txt
python sec_saxs_gui.py
```

1. **Data**: choose the parent folder (e.g. `rebinned_output`) and a scan subfolder (e.g. `62750_rebinned`).
   Frames must be named `shot_XXXX<suffix>.dat` with columns `q  I  err` and one header line, all on the same q-grid.
2. **q-range(s)**: one or two q-ranges for the average intensity; press *Load data + plot chromatogram*.
3. **Select frames** on the chromatogram:
   - click + drag = **sample** range (orange)
   - Shift + click + drag = **buffer** range (blue)
   - or type start/end frames in the boxes
4. **Buffer scale factor** (optional), then *Calculate scattering pattern*:
   - I_sub = I_sample − s·I_buffer
   - σ_sub = √(σ_sample² + (s·σ_buffer)²)
   - merged frames: I = mean(I_i), σ = √(Σσ_i²)/N
5. **Save** chromatograms to `output_chromatogram/` and buffer / sample / subtracted curves to `SAXS_output/`.

Example data: `rebinned_output/62750_rebinned`.

## Notebook workflow

Make sure these libraries are installed:
os
sys
numpy
matplotlib
ipympl
pandas

To run, frames/shots in the format of .dat from CoSAXS should be in a folder labeled scan-xxxxx placed in the Data folder. (The placeholder.txt can be removed)
E.g. Data/scan-62739/shot_0001.dat

Otherwise, change the data directory as needed.
