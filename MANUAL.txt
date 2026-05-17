# ReguΛarity - User Manual

## Purpose

ReguΛarity is a graphical analysis tool for the objective quantification of the
regularity of periodic surface structures, especially laser-induced periodic
surface structures (LIPSS). The software combines image-based Fourier analysis,
the P3S method (Perpendicular, Period and Phase Scanning), Gini-based period
statistics, Fourier orientation isotropy and DLOA (Dispersion of the LIPSS
Orientation Angle).

The current version keeps the established calculation workflow, but introduces a
more compact dashboard interface, project/run handling, improved ROI handling,
structured exports and a revised PSD width evaluation for HSFL-type spectra.

## Installation and Start

1. Download the installation file from the release package.
2. Use the file to install the ReguΛarity executable
3. Start the program by double-clicking `ReguΛarity.exe`.

No separate Python installation is required for the executable release.

## Main Window

The main window is organized as a compact scientific dashboard.

### Top Command Bar

- `Open`: open an existing ReguΛarity project.
- `Save`: save the current project with settings, runs, ROI definitions and
  available results.
- `Save Run`: save only one selected run as a separate project.
- `Browser`: open the Data Browser for detailed plot inspection.
- `Results`: open the full Results Table.
- Directory button: edit input and output directories.
- `Info`: enter or check physical image dimensions.
- `ROI`: define regions of interest before analysis.
- `P3S`: start P3S, period, phase, PSD and Fourier-isotropy analysis.
- `DLOA`: start DLOA analysis.
- `Export`: open the structured export dialog.
- `Reset`: clear the current session.

The Data Browser and Results Table open as independent windows. They can be
minimized separately from the main window.

### Overview Workspace

The central workspace shows overview plots for comparing result metrics across
images and runs. It is intended for batch-level comparison, not for detailed
inspection of individual plots.

Available controls:

- `Run`: show all runs or only one selected run.
- `Metric`: choose the displayed comparison metric.
- `Images`: include or exclude individual images from the overview plot.

Detailed plots such as processed image, period maps, phase maps, PSD, FFT,
Fourier isotropy and DLOA are inspected in the Data Browser.

## Recommended Analysis Workflow

1. Select the input and output directories.
2. Click `Info` and enter the physical image dimensions in µm.
3. Click `ROI` if only a selected image region should be analyzed.
4. Optionally apply the selected ROI to all remaining images.
5. Click `P3S` and select the P3S parameters.
6. Optionally click `DLOA`; the previously defined ROI set is reused.
7. Review batch-level comparisons in the Overview workspace.
8. Open `Browser` for individual image plots.
9. Open `Results` for numerical results and run filtering.
10. Use `Export` to save tables, metadata, plots and optional raw plot data.

## ROI Handling

ROI definitions are now handled before the analysis and are stored with the
project. P3S and DLOA use the same ROI definitions, which avoids accidental
comparison of different image regions.

ROI modes:

- Whole image: the complete image is analyzed.
- Manual ROI: a rectangular region is selected.
- Apply ROI to all remaining images: useful for image series with the same
  field of view.

ROI definitions are saved in the project and included in the export metadata.

## P3S Parameters

### Preprocessing

- Mean subtraction + Hann window: recommended default for robust FFT/P3S
  analysis.
- Mean subtraction: removes the DC component while keeping edge information.
- Hann window: reduces edge artifacts in the Fourier transform.
- No preprocessing: available for special cases.

### Segment Width

The segment width controls the lateral sampling of the local period and phase
analysis.

- `1 px`: highest local resolution.
- Larger values: faster processing, lower local resolution.

### Frequency Range

The frequency range is entered in `1/µm`.

- Minimum frequency excludes large-scale background variations.
- Maximum frequency defines the peak search range.

For PSD width evaluation, ReguΛarity now calculates the radial PSD up to the
physical limit allowed by the image calibration and the Nyquist criterion. The
entered maximum frequency controls where the dominant peak is searched, but it
does not artificially clip the PSD width calculation.

Nyquist frequency:

```
f_Nyquist = 1 / (2 * pixel_size)
```

Example:

```
pixel_size = 0.00390625 µm/px
f_Nyquist = 128 1/µm
```

If the entered maximum frequency is above the Nyquist frequency, the analysis is
internally limited to the physically valid frequency range.

### PSD Sampling

The PSD sampling setting controls the density of the radial PSD bins.

- Standard: fastest.
- Fine: denser radial PSD binning.
- Very fine: highest radial PSD bin density, useful for fine HSFL structures.

This option increases frequency-axis resolution without applying additional
smoothing.

### Decimals

Controls numerical rounding in displayed and exported result tables.

### Correction Factor

Adjusts the physical scale if the image was geometrically corrected before
analysis.

### Rotation

Optional automatic rotation is available as an experimental feature. Use it only
when the structure orientation should be normalized before analysis.

## DLOA Parameters

DLOA quantifies the dispersion of local structure orientation.

Available gradient modes:

- Riesz filters: recommended default.
- Gaussian.
- Finite difference.
- Splines.

Important settings:

- Local window sigma in px.
- Baseline percentile.
- Prominence-based peak evaluation.
- Optional rotation.

DLOA results are merged into the same session table as P3S results when the
images and runs match.

## Result Metrics

The normal result views focus on six key metrics:

- `RΛ,2D`: 2D regularity derived from PSD period width.
- `RΛ`: regularity of the y-period.
- `G`: Gini coefficient of the y-period.
- `δθ/DLOA`: orientation-angle dispersion.
- Mean Δ-y-phase.
- NEW: Fourier orientation isotropy `Hθ`. See also Georges et al., ASS, 741 (2026) 167063 10.1016/j.apsusc.2026.167063

The `All` result view additionally includes period and phase values that are
relevant for scientific interpretation. Developer diagnostics are kept in the
internal data structures but are intentionally hidden from the normal table.

## Results Table

The Results Table can be opened from the command bar. It supports:

- filtering by run,
- key-metric view,
- all-metric view,
- copy of selected values,
- Excel export.

The Results Table no longer contains a batch summary panel.

## Data Browser

The Data Browser is used for detailed inspection of individual plots. It shows:

- project/run tree,
- images and available plots,
- selected plot,
- structured details for image information and analysis parameters.

Available plot types depend on the completed analysis and may include:

- processed image,
- period plots,
- phase and Δ-phase plots,
- PSD,
- FFT,
- Fourier isotropy,
- DLOA.

Figure options are available for plot display/export styling, including font
family, font size and grid display.

## Export

The structured export dialog allows selection of export content:

- Results table as `.xlsx`.
- User settings, image information and ROI metadata as `.xlsx`.
- Plots as `.png`.
- Optional raw plot data as `.csv`.
- Optional grouping of plots/data by image.
- Plot font and font size for saved figures.

Raw plot data export is intended for external plotting software such as Origin.
Depending on available payloads, exported raw data may include:

- radial PSD data and selected width markers,
- FFT magnitude and log-magnitude matrices,
- processed image matrices,
- period and phase segment data,
- Fourier angular distributions,
- DLOA histograms.

## Project and Run Handling

Projects store:

- input and output directories,
- image metadata,
- ROI definitions,
- run parameters,
- result tables,
- plot references where available.

Multiple runs can be kept in one session. Overview plots and the Results Table
can be filtered by run. A selected run can be saved as a separate project.

## Supported Image Formats

Supported image formats include:

- `.tif`
- `.tiff`
- `.png`
- `.jpg`
- `.jpeg`

TIFF files with vendor-specific metadata tags are supported. Non-standard TIFF
metadata warnings are suppressed during normal GUI operation if the image data
can be read correctly.

## Troubleshooting

### No results are shown

Check that:

- input directory contains supported image files,
- image dimensions were entered via `Info`,
- ROI selection was completed,
- output directory is writable.

### PSD width looks too small or too large

Check:

- physical image calibration,
- pixel size and Nyquist frequency,
- selected frequency range,
- PSD sampling setting,
- whether the ROI contains enough periods.

For fine HSFL structures, use `Fine` or `Very fine` PSD sampling and ensure that
the maximum frequency is high enough to include the relevant HSFL peak.

### Plots look crowded

Open the plot in the Data Browser and use `Figure Options` to adjust font size,
font family and grid display before saving/exporting.

### Long batch processing

Use consistent ROI and image metadata before starting a batch. Very fine PSD
sampling may increase processing time.

## Citation

Please cite ReguΛarity as:

```
Eric Rahner, Tobias Thiele, Heike Voss, Frank A. Müller, Jörn Bonse, Stephan Gräf,
Objective, high-throughput regularity quantification of laser-induced periodic
surface structures (LIPSS), Applied Surface Science, 2026, 165919,
ISSN 0169-4332, https://doi.org/10.1016/j.apsusc.2026.165919.
```

## Contact

Eric Rahner
Otto Schott Institute of Materials Research
Friedrich Schiller University Jena

Email: eric.rahner@uni-jena.de
