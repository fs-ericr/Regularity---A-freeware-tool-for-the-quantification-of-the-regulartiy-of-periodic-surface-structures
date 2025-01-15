import numpy as np
from scipy.fftpack import fft, fftfreq, fft2, fftshift
import pandas as pd

from Evaluation import plot_spectrum
from Calculations import get_most_frequent_period_and_frequency

def seg_proc(image_data, df_c, segment_width, notch_Filter, coordinates):
    try:
        
        width_px = df_c.loc[0,'width_px']
        height_px = df_c.loc[0,'height_px']
        step_x = df_c.loc[0,'step_x']
        step_y = df_c.loc[0,'step_y']
        
        # Check required columns in df_c
        if 'width_px' not in df_c.columns or 'height_px' not in df_c.columns:
            raise ValueError(f"Missing required columns in df_c: {df_c.columns}")
            
            # Koordinaten validieren
        start_x, end_x = (coordinates[0] or 0), (coordinates[1] or width_px)
        start_y, end_y = (coordinates[2] or 0), (coordinates[3] or height_px)
    
        # Werte runden
        start_x, end_x = int(start_x), int(end_x)
        start_y, end_y = int(start_y), int(end_y)
            
        if start_x < 0 or end_x > width_px or start_y < 0 or end_y > height_px:
            start_x = max(0, start_x)
            end_x = min(width_px, end_x)
            start_y = max(0, start_y)
            end_y = min(height_px, end_y)

        selected_region = image_data[start_y:end_y, start_x:end_x]
        if selected_region.size == 0:
            raise ValueError("Selected region is empty.")


    
        # Cut out the area of interest from the image data
        selected_region = image_data[start_y:end_y, start_x:end_x]
        # 1D processing
        if (segment_width == 1):
            # Initialization of lists for results
            results = []
            # Iteration over horizontal segments of the selected area
            for seg_start_x in range(0, end_x - start_x, segment_width):           
                # Determine the end coordinate of the current segment
                seg_end_x = min(seg_start_x + segment_width, end_x - start_x)            
                # Cutting the current segment
                region = selected_region[:, seg_start_x:seg_end_x]             
                fft_result, freqs, magnitude    = compute_fft (region, step_x)
                period_y, phase_y, phase_list       = sort_and_compute (freqs, magnitude, fft_result, notch_Filter)            
                segment_micrometers = (start_x + seg_start_x + seg_end_x) / 2 * step_x           
                results.append({
                    "period_y": period_y,
                    "phase_y": phase_y,
                    "segment": segment_micrometers
                })           
                # Visualization of the spectrum
                #plot_spectrum(freqs, magnitude, phase_list)
            # collect all periods from the results
            periods_y = [value["period_y"] for value in results]
            # calculation of most frequent period / frequent
            mf_py, mf_fy, cv_y = get_most_frequent_period_and_frequency(periods_y)
            df_c['mf_py'] = mf_py
            df_c['mf_fy'] = mf_fy
            df_c['cv_y'] = cv_y     # Regularity parameter   
            # Extract phases from resuts
            phases_y = [result['phase_y'] for result in results]
            # Compute delta phases
            delta_phase_y = np.abs(np.diff(phases_y, prepend=phases_y[0]))       
            # Add delta_phase to each dictionary in results
            for i, result in enumerate(results):
                result["delta_phase_y"] = delta_phase_y[i]
        # 2D processing
        else:
            # Initialization of a list to collect the results
            results = []        
            # Iteration over horizontal segments of the selected area
            for seg_start_x in range(0, end_x - start_x, segment_width):            
                # Determine the end coordinate of the current segment
                seg_end_x = min(seg_start_x + segment_width, end_x - start_x)            
                # Cutting the current segment
                region = selected_region[:, seg_start_x:seg_end_x]          
                result, df_c = compute_2D (df_c, region, step_x, step_y)           
                segment_micrometers = (start_x + seg_start_x + seg_end_x) / 2 * step_x            
                # Saving results with additional information
                results.append({
                    "frequency_x": result["dominant_freq_x"],
                    "frequency_y": result["dominant_freq_y"],
                    "segment_start_x": seg_start_x,
                    "segment_end_x": seg_end_x,
                    "period_x": result["period_x"],
                    "period_y": result["period_y"],
                    "phase_x": result["phase_x"],
                    "phase_y": result["phase_y"],
                    "magnitude_x": result["magnitude_x"],
                    "magnitude_y": result["magnitude_y"],
                    "segment": segment_micrometers
                })
                #segment.append(segment_micrometers)
            # Extract phases from resuts
            periods_x = [value['period_x'] for value in results]
            periods_y = [value['period_y'] for value in results]
            mf_py, mf_fy, cv_y = get_most_frequent_period_and_frequency(periods_y)
            mf_px, mf_fx, cv_x = get_most_frequent_period_and_frequency(periods_x)
            df_c['mf_py'] = mf_py
            df_c['mf_fy'] = mf_fy
            df_c['cv_y'] = cv_y     # Regularity parameter   
            df_c['mf_px'] = mf_px
            df_c['mf_fx'] = mf_fx
            df_c['cv_x'] = cv_x     # Regularity parameter           
            # Extract phases from resuts
            phases_x = [value['phase_x'] for value in results]
            phases_y = [value['phase_y'] for value in results]
            # Compute delta phases
            delta_phase_y = np.abs(np.diff(phases_y, prepend=phases_y[0]))       
            delta_phase_x = np.abs(np.diff(phases_x, prepend=phases_x[0]))
            # Add delta_phase to each dictionary in results
            for i, result in enumerate(results):
                result["delta_phase_y"] = delta_phase_y[i]
                result["delta_phase_x"] = delta_phase_x[i]
                
        # Conversion of the results into a DataFrame
        return pd.DataFrame(results), df_c, selected_region #, segment
    except Exception as e:
        print(f"Error in seg_proc: {e}")
        raise
def compute_fft (data, step_x):    
    fft_result = fft (data.flatten())
    freqs = fftfreq(len(data), d = step_x)
    magnitude = np.abs(fft_result)    
    return fft_result, freqs, magnitude

def sort_and_compute (freqs, magnitude, fft_result, notch_Filter):    
    # Determine dominant frequency
    dominant_freq = freqs[np.argmax(magnitude)]        
    # Blocking range for dominant frequency
    if notch_Filter is not None:
        ignore_range = (np.abs(freqs - dominant_freq) < notch_Filter)
        filtered_freqs = freqs[~ignore_range]
        filtered_magnitude = magnitude[~ignore_range]
    else:
        filtered_freqs = freqs
        filtered_magnitude = magnitude
    # Find the indices of the two highest frequencies
    peak_indices = np.argsort(filtered_magnitude)[-2:]
    peak_indices.sort()
    # Calculation of the frequency spacing
    if len(peak_indices) == 2:
        peak_distance = np.abs(filtered_freqs[peak_indices[1]] - filtered_freqs[peak_indices[0]])
        period = 2 / peak_distance if peak_distance != 0 else np.nan
    else:
        period = np.nan
    # Dominant phase and all phases
    dominant_phase = np.angle(fft_result[peak_indices[0]], deg=False)
    phase_list = np.angle(fft_result, deg=False)
    return period, dominant_phase, phase_list            

def compute_2D(df_c, region, step_x, step_y):
    fft2_result = fft2(region)
    fft_shifted = fftshift(fft2_result)   
    # Calculation of magnitude and phase
    fft_magnitude = np.abs(fft_shifted)
    fft_phase = np.angle(fft_shifted)  
    # Calculation of frequencies
    nrows, ncols = region.shape
    freq_x = fftshift(fftfreq(ncols, d=step_x))
    freq_y = fftshift(fftfreq(nrows, d=step_y))
    # Projections of the magnitude onto the axes
    magnitude_x = np.sum(fft_magnitude, axis=0)  # Sum over the rows (y)
    magnitude_y = np.sum(fft_magnitude, axis=1)  # Sum over the columns (x)  
    # Indices of the maximum frequencies in x and y direction
    idx_x_max = np.argmax(magnitude_x)  # Dominant frequency in x-direction
    idx_y_max = np.argmax(magnitude_y)  # Dominant frequency in y-direction  
    # Dominant frequency
    dominant_freq_x = freq_x[idx_x_max]
    dominant_freq_y = freq_y[idx_y_max]
    # Phases at dominant frequencies
    phase_x = fft_phase[nrows // 2, idx_x_max]  # Phase in x-direction (row center, idx_x_max)
    phase_y = fft_phase[idx_y_max, ncols // 2]  # Phase in y-direction (column center, idx_y_max)
    # Calculate periods
    period_x = 1 / abs(dominant_freq_x) if dominant_freq_x != 0 else np.inf
    period_y = 1 / abs(dominant_freq_y) if dominant_freq_y != 0 else np.inf
    # Output of results
    # print(f"Dominant frequency: fx = {dominant_freq_x}, fy = {dominant_freq_y}")
    # print(f"Period: Tx = {period_x}, Ty = {period_y}")
    # print(f"Dominant phase phi_x: {phase_x}")
    # print(f"Dominant phase phi_y: {phase_y}")
    # Return of results
    return {
        "dominant_freq_x": dominant_freq_x,
        "dominant_freq_y": dominant_freq_y,
        "period_x": period_x,
        "period_y": period_y,
        "phase_x": phase_x,
        "phase_y": phase_y,
        "magnitude_x": magnitude_x,
        "magnitude_y": magnitude_y
        }, df_c


def gini_coefficient(df, df_c, l):
    
    if l == 1:
       
        df_c['gini_period_y']=gini_coefficient_calc(df['period_y'])
        df_c['gini_phase_y']=gini_coefficient_calc(df['phase_y'])
        df_c["gini_delta_phase_y"] = gini_coefficient_calc(df['delta_phase_y'])
        
        return df_c
        
    if l == 2: 
        df_c['gini_period_y']=gini_coefficient_calc(df['period_y'])
        df_c['gini_phase_y']=gini_coefficient_calc(df['phase_y'])
        df_c["gini_delta_phase_y"] = gini_coefficient_calc(df['delta_phase_y'])
        df_c['gini_period_x']=gini_coefficient_calc(df['period_x'])
        df_c['gini_phase_x']=gini_coefficient_calc(df['phase_x'])
        df_c["gini_delta_phase_x"] = gini_coefficient_calc(df['delta_phase_x'])

        return df_c

def gini_coefficient_calc(values):
    """Calculates the Gini-Coefficient for a given list of values."""
    values = np.array(values)
    n = len(values)
    if n == 0: 
        return np.nan
    sorted_values = np.sort(values)
    cumulative_sum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_values)) 
          - (n + 1) * cumulative_sum[-1]) / (n * cumulative_sum[-1])
    
    return gini

