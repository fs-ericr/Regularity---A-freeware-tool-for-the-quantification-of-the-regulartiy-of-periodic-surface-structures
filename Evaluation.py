import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fftfreq, fft2


def plot_spectrum(freqs, magnitude, phases, title="Spectrum"):
    """Generate a figure with amplitude and phase plots with improved spacing and font sizes."""
    fig = plt.Figure(figsize=(8, 8))  # Set a more compact figure size

    # Amplitude Plot
    ax_ampl = fig.add_subplot(2, 1, 1)
    ax_ampl.plot(freqs, magnitude, 'b-', markersize=5)
    ax_ampl.set_title(f"{title} - Amplitude", fontsize=10)
    ax_ampl.set_xlabel('Frequency (1/µm)', fontsize=9)
    ax_ampl.set_ylabel('Amplitude', fontsize=9)
    ax_ampl.grid(True)

    # Phase Plot
    ax_phase = fig.add_subplot(2, 1, 2)
    ax_phase.plot(freqs, phases, 'g-', markersize=5, label='Phase (radians)')
    ax_phase.set_title(f"{title} - Phase", fontsize=10)
    ax_phase.set_xlabel('Frequency (1/µm)', fontsize=9)
    ax_phase.set_ylabel('Phase (radians)', fontsize=9)
    ax_phase.grid(True)

    ax2 = ax_phase.twinx()
    ax2.plot(freqs, np.degrees(phases), 'r--', markersize=5, label='Phase (degrees)')
    ax2.set_ylabel('Phase (degrees)', color='red', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='red')

    # Adjust layout to avoid overlap
    fig.tight_layout()

    return fig


def analyze_and_visualize_results(df, df_c, image_data, selected_region):
    """
    Analyzes the data and visualizes results as matplotlib.Figure objects.
    Returns a list of figures with improved layout and font size.
    """
    
    def micrometer_to_pixel(x):
         return x / step_x
    def pixel_to_micrometer(px):
         return px * step_x
    
    figs = []
    
    # Collect necessary data
    image_name = df_c.loc[0, 'image_name']
    step_x = df_c.loc[0, 'step_x']
    segment = df['segment'].tolist()
    period_y = df['period_y'].tolist()
    phase_y = df['phase_y'].tolist()
    delta_phase_y = df['delta_phase_y'].tolist()
    
    # Statistical calculations
    mean_period_y = np.mean(period_y)
    std_period_y = np.std(period_y)
    mean_delta_phase_y = np.mean(delta_phase_y)
    std_delta_phase_y = np.std(delta_phase_y)
    mean_phase_y = np.mean(phase_y)
    std_phase_y = np.std(phase_y)
    
    # Store results in df_c
    df_c.loc[0, 'mean_period_y'] = round(mean_period_y,4)
    df_c.loc[0, 'std_period_y'] = round(std_period_y,4)
    df_c.loc[0, 'mean_phase_y'] = round(mean_phase_y,4)
    df_c.loc[0, 'std_phase_y'] = round(std_phase_y,4)
    df_c.loc[0, 'mean_delta_phase_y'] = round(mean_delta_phase_y,4)
    df_c.loc[0, 'std_delta_phase_y'] = round(std_delta_phase_y,4)
    
    # Log results to console
    print("\nResults")
    print(f"Mean period (Y): {mean_period_y:.5f} µm (std: {std_period_y:.5f} µm)")
    print(f"Mean phase (Y): {mean_phase_y:.5f} rad (std: {std_phase_y:.5f} rad)")
    print(f"Mean Δ phase (Y): {mean_delta_phase_y:.5f} rad (std: {std_delta_phase_y:.5f})")

    # ---- Figure 1: analyzed region ----
    fig_region = plt.Figure(figsize=(5, 4))  # Set figure size to be more compact
    ax_reg = fig_region.add_subplot(111)
    cax = ax_reg.imshow(selected_region, cmap='gray')
    ax_reg.set_title(f"Analyzed Region in {image_name}", fontsize=10)
    ax_reg.set_xlabel("X (pixels)", fontsize=9)
    ax_reg.set_ylabel("Y (pixels)", fontsize=9)
    fig_region.colorbar(cax, label="Intensity")
    figs.append(fig_region)
    
    # ---- Figure 2: Period with pixel scale ----
    fig1 = plt.Figure(figsize=(5, 4))
    ax1 = fig1.add_subplot(111)
    ax1.plot(segment, period_y, linestyle='-', color='r', label='Y-Period (µm)')
    ax1.axhline(mean_period_y, color='r', linestyle='--', label=f'Mean Y-Period: {mean_period_y:.5f} µm')
    ax1.fill_between(segment, mean_period_y - std_period_y, mean_period_y + std_period_y,
                     color='r', alpha=0.2, label=f'SD Y-Period ±{std_period_y:.5f} µm')
    ax1.set_xlabel('x (µm)', fontsize=9)
    ax1.set_ylabel('Y-Period (µm)', color='r', fontsize=9)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True)
    
    secax = ax1.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
    secax.set_xlabel('x (pixels)', fontsize=9)
    
    fig1.tight_layout()
    figs.append(fig1)

        # ---- Figure 3: Phase and delta-phase ----
    fig2 = plt.Figure(figsize=(5, 4))
    axs = fig2.add_gridspec(2, 1, height_ratios=[1, 1])
    ax_phase = fig2.add_subplot(axs[0])
    ax_delta_phase = fig2.add_subplot(axs[1], sharex=ax_phase)

    ax_phase.plot(segment, phase_y, color='orange', label='Y-Phase (rad)', markersize=4)
    ax_phase.axhline(mean_phase_y, color='orange', linestyle='--', label=f'Mean Y-Phase: {mean_phase_y:.5f} rad')
    ax_phase.fill_between(segment, mean_phase_y - std_phase_y, mean_phase_y + std_phase_y, color='orange', alpha=0.2,
                          label=f'SD Y-Phase ±{std_phase_y:.5f} rad')
    ax_phase.set_ylabel('Y-Phase (rad)', color='orange', fontsize=9)
    ax_phase.legend(loc='upper right', fontsize=8)
    ax_phase.grid(True)
    
    secax = ax_phase.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
    secax.set_xlabel('x (pixels)', fontsize=9)

    ax_delta_phase.plot(segment, delta_phase_y, color='b', label='Y-Δ-Phase (rad)', markersize=4)
    ax_delta_phase.axhline(mean_delta_phase_y, color='b', linestyle='--', label=f'Mean Y-Δ-Phase: {mean_delta_phase_y:.5f} rad')
    ax_delta_phase.fill_between(segment, mean_delta_phase_y - std_delta_phase_y, mean_delta_phase_y + std_delta_phase_y,
                                    color='b', alpha=0.2, label=f'SD Δ-Phase ±{std_delta_phase_y:.5f} rad')
    ax_delta_phase.set_xlabel('x (µm)', fontsize=9)
    ax_delta_phase.set_ylabel('Y-Δ-Phase (rad)', color='b', fontsize=9)
    ax_delta_phase.legend(loc='upper right', fontsize=8)
    ax_delta_phase.grid(True)
    secax = ax_delta_phase.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
    secax.set_xlabel('x (pixels)', fontsize=9)
    
    fig2.tight_layout()
    figs.append(fig2)

    return figs  # Return list of figure objects



def analyze_and_visualize_results_2d(df, df_c, image_data, selected_region):
    """
    Analyzes the data and visualizes results as matplotlib.Figure objects.
    Returns a list of figures with improved layout and font size.
    """

    def micrometer_to_pixel(x):
         return x / step_x
    def pixel_to_micrometer(px):
         return px * step_x
       
    figs = []
    
    image_name = df_c.loc[0, 'image_name']
    step_x = df_c.loc[0, 'step_x']
    
    segment = df['segment'].tolist()
    
    # Berechnung der Mittelwerte und Standardabweichungen
    mean_period_x = df['period_x'].mean()
    std_period_x = df['period_x'].std()
    mean_period_y = df['period_y'].mean()
    std_period_y = df['period_y'].std()
    mean_phase_x = df['phase_x'].mean()
    std_phase_x = df['phase_x'].std()
    mean_phase_y = df['phase_y'].mean()
    std_phase_y = df['phase_y'].std()
    
    # Ausgabe der berechneten Werte
    print(f"Mean Period (X): {mean_period_x:.5f} µm (Std: {std_period_x:.5f} µm)")
    print(f"Mean Period (Y): {mean_period_y:.5f} µm (Std: {std_period_y:.5f} µm)")
    print(f"Mean Phase (X): {mean_phase_x:.5f} rad (Std: {std_phase_x:.5f} rad)")
    print(f"Mean Phase (Y): {mean_phase_y:.5f} rad (Std: {std_phase_y:.5f} rad)")
    print(f"Coefficient of variation y: {df_c.loc[0, 'cv_y']}")
    print(f"Coefficient of variation x: {df_c.loc[0, 'cv_x']}")
    
    # Displaying the selected area
    plt.figure(figsize=(6, 5))
    plt.imshow(selected_region, cmap='gray')
    plt.xlabel("X (pixels)", fontsize=9)
    plt.ylabel("Y (pixels)", fontsize=9)
    plt.colorbar(label="Intensity")
    plt.show()
    
        # Plot: Periods X and Y direction below each other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    
    # X-Period
    ax1.plot(segment, df['period_x'], linestyle='-', color='b', label='Period X (µm)')
    ax1.axhline(mean_period_x, color='b', linestyle='--', label=f'Mean Period X: {mean_period_x:.5f} µm')
    ax1.fill_between(segment, mean_period_x - std_period_x, mean_period_x + std_period_x, color='b', alpha=0.2, label=f'SD X-Period ±{std_period_x:.5f} µm')
    ax1.set_ylabel('X-Period (µm)', fontsize=9)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle = '--', alpha = 0.7)
    
    # Secondary x-axis for pixel scale
    img_width_pixels = image_data.shape[1]  # Assumes image_data is an array with shape [height, width]

    
    secax = ax1.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
    secax.set_xlabel('x (pixels)', fontsize=9)
        
    # Set finer ticks for both x-axes
    num_ticks_pixels = 10  # Finer granularity for pixel scale
    pixel_ticks = np.linspace(0, img_width_pixels, num_ticks_pixels)
    secax.set_xticks(pixel_ticks)
        
    num_ticks_micrometers = 10  # Finer granularity for micrometer scale
    micrometer_ticks = np.linspace(segment[0], segment[-1], num_ticks_micrometers)
    ax1.set_xticks(micrometer_ticks)
    
    ax1.set_title(f'Period X/Y - {image_name}', pad=10, fontsize=12)
    
    # Y-Period
    ax2.plot(segment, df['period_y'], linestyle='-', color='g', label='Period Y (µm)')
    ax2.axhline(mean_period_y, color='g', linestyle='--', label=f'Mean Y-Period: {mean_period_y:.5f} µm')
    ax2.fill_between(segment, mean_period_y - std_period_y, mean_period_y + std_period_y, color='g', alpha=0.2, label=f'SD Y-Period ±{std_period_y:.5f} µm')
    ax2.set_xlabel('x (µm)', fontsize=9)
    ax2.set_ylabel('Y-Period (µm)', fontsize=9)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True)
    #ax2.set_title(f'Y-Period - {image_name}', pad=10)

    # Shared x-ticks for micrometer scale
    ax2.set_xticks(micrometer_ticks)
    secax = ax2.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
    secax.set_xlabel('x (pixels)', fontsize=9)
    plt.tight_layout(pad=3)
    figs.append(fig)
        
    # Handling segment width > 1 (multiple segments)
    if len(segment) > 1 and df['segment'].nunique() > 1:
        # Plot Phase X and Δ-Phase X (If multiple segments exist)
        fig_x = plt.Figure(figsize=(6, 5), dpi=100)
        ax_phase_x = fig_x.add_subplot(211)
        ax_delta_x = fig_x.add_subplot(212, sharex=ax_phase_x)
        
        # Phase X plot
        ax_phase_x.set_title(f'X-Phase/Δ-Phase - {image_name}', pad = 10)
        ax_phase_x.plot(segment, df['phase_x'], color='orange', linestyle='-', label='X-Phase (rad)', markersize=4)
        ax_phase_x.axhline(mean_phase_x, color='orange', linestyle='--', label=f'Mean X-Phase: {mean_phase_x:.5f} rad')
        ax_phase_x.fill_between(segment, mean_phase_x - std_phase_x, mean_phase_x + std_phase_x, color='orange', alpha=0.2, label=f'SD X-Phase ±{std_phase_x:.5f} rad')
        ax_phase_x.set_ylabel('X-Phase (rad)', fontsize=9)
        ax_phase_x.legend(loc='upper right', fontsize=8)
        ax_phase_x.grid(True)
        secax = ax_phase_x.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
        secax.set_xlabel('x (pixels)', fontsize=9)
        
        # Δ-Phase X plot
        delta_phase_x = np.abs(df['phase_x'].diff().fillna(0))  # Calculate Δ-Phase X
        ax_delta_x.plot(segment, delta_phase_x, color='black', linestyle='-', label='Δ-Phase X (rad)', markersize=4)
        ax_delta_x.axhline(delta_phase_x.mean(), color='black', linestyle='--', label=f'Mean ΔPhase X: {delta_phase_x.mean():.5f} rad')
        ax_delta_x.fill_between(segment, delta_phase_x.mean() - delta_phase_x.std(), delta_phase_x.mean() + delta_phase_x.std(), color='black', alpha=0.2, label=f'SD ΔPhase X ±{delta_phase_x.std():.5f} rad')
        ax_delta_x.set_xlabel('x (µm)', fontsize=9)
        ax_delta_x.set_ylabel('X-Δ-Phase (rad)', fontsize=9)
        ax_delta_x.legend(loc='upper right', fontsize=8)
        ax_delta_x.grid(True)
        secax = ax_delta_x.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
        secax.set_xlabel('x (pixels)', fontsize=9)
        figs.append(fig_x)
        
        # Plot Phase Y and Δ-Phase Y (If multiple segments exist)
        fig_y = plt.Figure(figsize=(6, 5), dpi=100)
        ax_phase_y = fig_y.add_subplot(211)
        ax_delta_y = fig_y.add_subplot(212, sharex=ax_phase_y)
        
        # Phase Y plot
        ax_phase_y.plot(segment, df['phase_y'], color='purple', linestyle='-', label='Y-Phase (rad)', markersize=4)
        ax_phase_y.axhline(mean_phase_y, color='purple', linestyle='--', label=f'Mean Y-Phase: {mean_phase_y:.5f} rad')
        ax_phase_y.fill_between(segment, mean_phase_y - std_phase_y, mean_phase_y + std_phase_y, color='purple', alpha=0.2, label=f'SD Y-Phase ±{std_phase_y:.5f} rad')
        ax_phase_y.set_ylabel('Y-Phase (rad)', fontsize=9)
        ax_phase_y.legend(loc='upper right', fontsize=8)
        ax_phase_y.grid(True)
        ax_phase_y.set_title(f'Y-Phase/Δ-Phase - {image_name}', pad=10)
        secax = ax_phase_y.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
        secax.set_xlabel('x (pixels)', fontsize=9)
        # Δ-Phase Y plot
        delta_phase_y = np.abs(df['phase_y'].diff().fillna(0))  # Calculate Δ-Phase Y
        ax_delta_y.plot(segment, delta_phase_y, color='cyan', linestyle='-', label='Δ-Phase Y (rad)', markersize=4)
        ax_delta_y.axhline(delta_phase_y.mean(), color='cyan', linestyle='--', label=f'Mean ΔPhase Y: {delta_phase_y.mean():.5f} rad')
        ax_delta_y.fill_between(segment, delta_phase_y.mean() - delta_phase_y.std(), delta_phase_y.mean() + delta_phase_y.std(), color='cyan', alpha=0.2, label=f'SD ΔPhase Y ±{delta_phase_y.std():.5f} rad')
        ax_delta_y.set_xlabel('x (µm)', fontsize=9)
        ax_delta_y.set_ylabel('Y-Δ-Phase  (rad)', fontsize=9)
        ax_delta_y.legend(loc='upper right', fontsize=8)
        ax_delta_y.grid(True)
        secax = ax_delta_y.secondary_xaxis('top', functions=(micrometer_to_pixel, pixel_to_micrometer))
        secax.set_xlabel('x (pixels)', fontsize=9)
        figs.append(fig_y)
    
    return figs  # Liste von Figures zurückgeben




def plot_fft(image_data, df_c, adjust_colormap=True):
    """Generate a figure showing the FFT plot of the image."""
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    # Perform the FFT calculation
    fft_spectrum = np.abs(fftshift(fft2(image_data)))
    height, width = image_data.shape

    # Compute frequency axes
    freqs_x = fftshift(fftfreq(width, d=df_c.loc[0, 'step_x']))
    freqs_y = fftshift(fftfreq(height, d=df_c.loc[0, 'step_y']))

    # Apply logarithmic scaling to the FFT magnitude
    log_fft_spectrum = np.log1p(fft_spectrum)

    if adjust_colormap:
        vmin = np.mean(log_fft_spectrum)
        vmax = np.max(log_fft_spectrum)
    else:
        vmin = np.min(log_fft_spectrum)
        vmax = np.max(log_fft_spectrum)

    im = ax.imshow(log_fft_spectrum, cmap='inferno', extent=(freqs_x[0], freqs_x[-1], freqs_y[0], freqs_y[-1]), aspect='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label='Log of FFT Magnitude')
    ax.set_title(f'Fourier Transform of {df_c.loc[0, "image_name"]}')
    ax.set_xlabel('Frequency (1/µm)')
    ax.set_ylabel('Frequency (1/µm)')
    ax.grid(False)

    fig.tight_layout()
    return fig


def get_single_metadata_value(df, column_name):
    """Retrieve the first non-null value from a specified column in a dataframe."""
    if column_name in df and not df[column_name].dropna().empty:
        return df[column_name].dropna().iloc[0]
    return None


def extract_and_save_data(output_dir, df_c, notch_Filter, segment_width, df):
    """Extract data, create metadata, and save it as a CSV file."""
    
    # Erstelle den Pfad, unter dem die Ergebnisse gespeichert werden sollen
    output_path = os.path.join(output_dir, f"{df_c.loc[0, 'image_name']}.csv")

    # Extrahiere Metadaten
    metadata = {
        "Image Name": df_c.loc[0, 'image_name'],
        "Magnification": df_c.loc[0, 'magnification'],
        "Height (px)": df_c.loc[0, 'height_px'],
        "Width (px)": df_c.loc[0, 'width_px'],
        "Height (µm)": df_c.loc[0, 'height_um'],
        "Width (µm)": df_c.loc[0, 'width_um'],
        "Step_x (µm/px)": df_c.loc[0, 'step_x'],
        "Step_y (µm/px)": df_c.loc[0, 'step_y'],
        "Notch_Filter (1/µm)": notch_Filter,
        "Segment Width (px)": segment_width,
    }

    # Daten zur x-Achse vorbereiten
    x_dependent_data = {
        "x (µm)": [i * df_c.loc[0, 'step_x'] for i in range(len(df['period_y'].tolist()))],
        "x (pixels)": range(len(df['period_y'].tolist())),
    }

    # Optionale Metadaten-Spalten
    optional_metadata_columns = {
        "most frequent period y (µm)": get_single_metadata_value(df_c, 'mf_py'),
        "most frequent frequency y (1/µm)": get_single_metadata_value(df_c, 'mf_fy'),
        "Coefficient of variation y": get_single_metadata_value(df_c, 'cv_y'),
        "most frequent period x (µm)": get_single_metadata_value(df_c, 'mf_px'),
        "most frequent frequency x (1/µm)": get_single_metadata_value(df_c, 'mf_fx'),
        "Coefficient of variation x": get_single_metadata_value(df_c, 'cv_x')
    }

    # Füge optionale Metadaten zu den Metadaten hinzu
    metadata.update({key: value for key, value in optional_metadata_columns.items() if value is not None})

    # Optional Spalten-Daten aus df hinzufügen
    optional_columns = {
        "period_y (µm)": df.get('period_y', pd.Series(dtype=float)).dropna().tolist(),
        "phase_y (rad)": df.get('phase_y', pd.Series(dtype=float)).dropna().tolist(),
        "ΔPhase_y (rad)": df.get('delta_phase_y', pd.Series(dtype=float)).dropna().tolist(),
        "period_x (µm)": df.get('period_x', pd.Series(dtype=float)).dropna().tolist(),
        "phase_x (rad)": df.get('phase_x', pd.Series(dtype=float)).dropna().tolist(),
        "ΔPhase_x (rad)": df.get('delta_phase_x', pd.Series(dtype=float)).dropna().tolist()
    }

    # Füge optionale Spalten-Daten zur x_dependent_data hinzu, falls sie existieren
    for key, value in optional_columns.items():
        if value:
            x_dependent_data[key] = value

    # Erstelle DataFrames für Metadaten und x-abhängige Daten
    metadata_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])
    x_dependent_df = pd.DataFrame(x_dependent_data)

    # Speichere die Daten als CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        metadata_df.to_csv(f, index=False, header=False)
        f.write("\n")  # Leere Zeile
        x_dependent_df.to_csv(f, index=False)

    print(f"Results for '{df_c.loc[0, 'image_name']}' saved as CSV in: {output_path}")
    
    return x_dependent_df

