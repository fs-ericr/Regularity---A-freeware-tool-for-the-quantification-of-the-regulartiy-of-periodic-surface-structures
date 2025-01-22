# **ReguΛarity - Quantification of Surface Structure Regularity**

## **📌 Project Overview**
The **Regularity Quantification Tool** is designed to analyze surface structures in images and quantify their regularity using **Fourier Transformation, P³S Method, and Gini Coefficients**. The software provides 1D and 2D Fourier analysis, visualizations, and statistical measures to assess structural periodicity and uniformity.

---

## **🚀 Key Features**
✔ **Graphical User Interface (GUI)** for intuitive operation  
✔ **Custom segmentation width** (1 px or more for accuracy adjustment)  
✔ **Fourier Transform Analysis (FFT, P³S Method, Gini Coefficients)**  
✔ **Computation of key statistical measures**:
  - Mean and standard deviation of period & phase
  - Delta-phase analysis
  - Regularity Paremeter for for the regularity of period and the phase change
  - Gini Coefficient for quantifying distribution uniformity
    
✔ **Visualization tools**:
  - Interactive plots for frequency, phase, and period distribution
  - Spectral analysis
    
✔ **Export results** to CSV for further analysis  

---

## **📥 Installation & Setup**

###  Download & Run the Application**
#### Use the Prebuilt `.exe` File**
- Download the latest **executable file** from [GitHub Releases](https://github.com/fs-ericr/Regularity---Quantification-of-the-Regularity-of-surface-structures/releases).
- Extract the downloaded ZIP file and **run `Regularity.exe`**.

## **🖼 How to Use the GUI**

### **1️⃣ Select Image Directory**
1. Click **"Browse"** under **Input Directory** to select the folder containing images.
2. Click **"Browse"** under **Output Directory** to choose where to save results.

### **2️⃣ Configure Processing Parameters**
1. **Choose a Preprocessing Filter:**
   - None
   - Substract Mean + Hanning-Window
   - Substract Mean
   - Hanning-Window
2. **Select Segment Width** to adjust analysis accuracy.
3. **Set Decimal Precision** for numerical outputs.

### **3️⃣ Start Processing**
- Click **"Set Image Info per File"** to input image properties.
- Enable **Rotation** if needed.
- Click **"Run Processing"** to execute the analysis.

### **4️⃣ View & Export Results**
- Click **"Show Results Table"** for an overview.
- Click **"Export CSV"** to save results.

---

## **📊 Understanding the Output**

### **Image Data**
- `image_name`: Processed image name.
- `magnification`: Microscope magnification.
- `width_px`, `height_px`: Image dimensions in pixels.
- `width_um`, `height_um`: Image dimensions in micrometers.

### **Period Analysis**
- `Mean Period X/Y`: Average period in X/Y direction.
- `SD Period X/Y`: Standard deviation of period.
- `mf_px/py`: Most frequent period.
- `mf_fx/fy`: Most frequent frequency.
- `cv_x/y`: Regularity of Period.
- `gini_period_x/y`: Gini coefficient of period distribution.

### **Phase Analysis**
- `Mean Phase X/Y`: Mean phase value.
- `SD Phase X/Y`: Standard deviation of phase.
- `Mean Delta-Phase X/Y`: Mean delta-phase.
- `SD Delta-Phase X/Y`: Standard deviation of delta-phase.
- `Gini Delta-Phase X/Y`: Gini coefficient of delta-phase distribution.

---

## **📊 Visualization Features**
- **FFT Spectrum**: Frequency domain representation.
- **Period Distribution**: Plots period values along the segment.
- **Phase Distribution**: Displays phase values and delta-phase variations.

---

## **📬 Contact & Support**
For any issues or feature requests, please **open an Issue on GitHub** or contact the developer directly.

📌 **Latest Release:** [GitHub Releases](https://github.com/fs-ericr/Regularity---Quantification-of-the-Regularity-of-surface-structures/releases).

