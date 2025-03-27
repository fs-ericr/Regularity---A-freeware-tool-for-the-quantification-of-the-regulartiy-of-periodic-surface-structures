# ReguΛarity – Comprehensive User Guide
![LoadingScreen](https://github.com/user-attachments/assets/3015dcb9-6bd4-42f2-9537-d34be70a5c8a)


## Introduction

ReguΛarity software provides researchers and engineers with a powerful graphical tool for the detailed and automated analysis and quantification of the regularity of laser-induced periodic surface structures (LIPSS). ReguΛarity integrates advanced mathematical methods such as Fourier transformation, the Perpendicular, Period, and Phase Scanning (P³S) method, Gini coefficient analysis, and the calculation of the Dispersion of the LIPSS Orientation Angle (DLOA). This enables precise, reproducible, and efficient examination of surface structures.

## Installation and Startup

Follow these simple steps to install the software:

1. Download the latest version of the executable file (`ReguΛarity.exe`) from the [GitHub Releases section](#).
2. Save the file to an appropriate folder on your computer.
3. Double-click the file to launch the software.

## User Interface and General Usage

The user interface is intuitive and includes the following key elements:

- **Input Directory:** Folder selection for your image data.
- **Output Directory:** Destination folder for analysis results and exported data.
  **File Infos:** Insert the correct physical dimensions of your images

Ensure the correct paths are set before beginning analysis to avoid errors.

## Parameter Settings

### Segment & Compute

The P³S method systematically analyzes period and phase data of LIPSS structures. You can adjust the following parameters:

- **Filter Options:**
  - **Hann Window:** Reduces artifacts in Fourier transformation by applying smooth transitions at image edges, enhancing frequency spectrum quality.
  - **Mean Subtraction:** Enhances image contrast by subtracting the mean grayscale value, aiding in accurate period detection.
  - **Combined:** Optimal combination of both filters for highest accuracy and stability.

- **Segment Width (px):**
  - Default is `1 px`, providing maximum spatial resolution and accuracy for period and phase determination.
  - Larger segment widths accelerate analysis but reduce local resolution.

- **Notch Filter (1/µm):**
  - Acts as a circular aperture in frequency space, restricting analysis to relevant structural sizes.
  - Default value is `0.1 1/µm`. Adjusting the notch filter allows targeted analysis of High Spatial Frequency LIPSS (HSFL) or Low Spatial Frequency LIPSS (LSFL).

- **Decimals:**
  - Sets the number of decimal places for result display and storage. 
- **Correction Factor:**
  - Default is `0.99`. Reduces edge effects caused by the Hann Window, preventing distortion of the mean period.

- **Rotation:**
  - Automatically aligns structures to the x-axis using Fourier transformation. This ensures consistent structural orientation, improves analysis precision, and minimizes artifacts.

### DLOA Calculation

The DLOA calculation quantifies structure orientation, offering the following methods:

- **Gradient Methods:**
  - **Riesz Filters (recommended):** Robust, isotropic gradient calculation with high stability and quality, ideal for fine structures and sensitive analyses.
  - **Gaussian:** Effective results for noisier images.
  - **Finite Difference:** Faster but less accurate method, suitable for preliminary analyses.
  - **Splines:** Highest precision for exceptionally smooth surface structures.

- **Local Window σ (px):**
  - Determines how strongly adjacent pixels influence the structure tensor calculation.
  - Default value `σ=1 px` offers maximum spatial accuracy but increases noise sensitivity.
  - Larger values enhance noise stability but reduce spatial detail resolution.

- **Rotation:**
  - Automatically aligns structures to the x-axis using Fourier transformation. This ensures consistent structural orientation, improves analysis precision, and minimizes artifacts.

## Conducting an Analysis

1. Ensure input and output directories are correctly set.
2. Select the desired parameters (P³S or DLOA).
3. Start the analysis by clicking the "Start Calculation" button.
4. Optionally define a Region of Interest (ROI) to exclude artifacts and focus analysis on specific areas.
5. Confirm your selection to start the analysis automatically.

## Results Display and Visualization

### Data Browser

Upon analysis completion, the Data Browser opens automatically, interactively displaying:
- Processed grayscale image after filtering.
- 2D Fourier spectrum of raw data.
- Distributions of periods (Λ), phases (φ), and phase changes (Δφ).

These interactive visualizations enable direct examination and saving of results.

### Results Table

Summarizes all quantitative results compactly:
- regularity of period (RΛ)
- Phase (φ) and phase changes (Δφ)
- Gini coefficient (G)
- DLOA (Dispersion of the LIPSS Orientation Angle)
- ...

Exportable as a CSV file for further analysis.


## Data Export and Further Processing

Saved data includes:
- (optional) Processed images
- (optional) Interactive plots
- Results in Excel-compatible CSV files

## Troubleshooting and Recommendations

- Supported image formats: `.jpg`, `.png`, `.tif`.
- Utilize optimal parameter and ROI settings.
- If unexplained errors occur, restart ReguΛarity and verify your parameter settings.

## Citation

Please cite ReguΛarity as follows:

```
Rahner et al., ReguΛarity Software, Version X.X, Year, URL
```

## Contact and Support

For questions or technical support, contact:

**Eric Rahner**, Otto Schott Institute of Materials Research, Friedrich Schiller University Jena  
*Email: eric.rahner@uni-jena.de*
