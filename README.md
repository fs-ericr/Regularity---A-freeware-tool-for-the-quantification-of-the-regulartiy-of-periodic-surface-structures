# ReguΛarity

## A freeware tool for the objective quantification of the regularity of periodic surface structures

ReguΛarity is a graphical analysis application for scientists and researchers.
It quantifies the regularity of periodic surface structures using image-based
methods such as FFT, P3S, Gini-based period statistics, Fourier orientation
isotropy and DLOA.

## Key Features

- Compact dashboard GUI with run-aware overview plots.
- Separate Data Browser for detailed plot inspection.
- Separate Results Table with run filtering and Excel export.
- Shared ROI workflow for P3S and DLOA, including apply-to-all ROI support.
- Project saving/loading with run parameters, image metadata and ROI definitions.
- Structured export of results, metadata, plots and optional raw plot data.
- PSD sampling options for fine HSFL structures.
- Revised PSD width evaluation using R68 envelope support for broad HSFL peaks.
- Origin-friendly raw data export for PSD, FFT, DLOA and segment-level plots.

## Installation

Download the executable release package and start `ReguΛarity.exe` to start the installer.
No separate Python installation is required for the executable version.

## How to Use

See `MANUAL.txt` for detailed instructions.

Basic workflow:

1. Open or select an input/output project directory.
2. Enter physical image dimensions via `Info`.
3. Define ROI settings if needed.
4. Run `P3S`.
5. Optionally run `DLOA`.
6. Inspect batch comparisons in the main window.
7. Inspect individual plots in the Data Browser.
8. Export results, metadata, plots and optional raw data.

## Licensing

This software is distributed under a freeware license:

- Free to use for non-commercial purposes.
- Redistribution of the unmodified software is allowed.
- Modification, reverse engineering, commercial use and sale are prohibited
  unless explicitly permitted by the author.

See `LICENSE.txt` for complete details.

## Citation

If you use ReguΛarity for research or publications, please cite:

```
Eric Rahner, Tobias Thiele, Heike Voss, Frank A. Müller, Jörn Bonse, Stephan Gräf,
Objective, high-throughput regularity quantification of laser-induced periodic
surface structures (LIPSS), Applied Surface Science, 2026, 165919,
ISSN 0169-4332, https://doi.org/10.1016/j.apsusc.2026.165919.
```

## Funding

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German
Research Foundation) - Project Number 530345255.

## Contact

Eric Rahner
Otto Schott Institute for Materials Research (OSIM)
Friedrich Schiller University Jena

eric.rahner@uni-jena.de
