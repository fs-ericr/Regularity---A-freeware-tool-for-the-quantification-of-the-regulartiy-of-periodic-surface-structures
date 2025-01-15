"""Module providing all calculation funcations."""

from collections import Counter
import numpy as np
import pandas as pd

def get_most_frequent_period_and_frequency(periods):
    """ calculats: most frequent period / frequency, coeffient of variation """

    # Periods are rounded to 3 decimal places to group similar values
    rounded_periods = np.round(periods, decimals=3)
    # Find most frequent period
    period_counter = Counter(rounded_periods)
    mf_p, _ = period_counter.most_common(1)[0]
    # Calculate frequency from the most frequent period
    mf_f = 1 / mf_p if mf_p != 0 else np.nan
    # Regularity parameter: Standard deviation and coefficient of variation
    std_dev = np.std(periods)
    mean_period = np.mean(periods)
    cv = std_dev / mean_period if mean_period != 0 else np.nan

    return mf_p, mf_f, cv

def apply_size_correction(df_c, correction_factors):
    factor = correction_factors
    try:
        df_c['height_px'] = (df_c['height_px']*factor).round().astype(int)
        df_c['width_px'] = (df_c['width_px']*factor).round().astype(int)
        print(f"New height after correction: {df_c['height_px'].values}")
        print(f"New width after correction: {df_c['width_px'].values}")
        return df_c
    except:
        print(f"Correction error with factor {factor}")
        raise ValueError("Error:Image could not be scaled correctly.")

def round_dataframe(df, decimals):
    """
    Rounds all numerical values in a DataFrame to the specified number of decimal places.
    """
    return df.apply(lambda col: col.map(lambda x: round(x, decimals) if isinstance(x, (int, float)) else x))

def round_nested(data, decimals):
    """
    Rounds numerical values in nested lists, dictionaries or other structures.
    """
    if isinstance(data, list):
        return [round_nested(item, decimals) for item in data]
    if isinstance(data, dict):
        return {key: round_nested(value, decimals) for key, value in data.items()}
    if isinstance(data, pd.DataFrame):
        return round_dataframe(data, decimals)
    if isinstance(data, (int, float)):
        return round(data, decimals)
    return data  # Return unchanged if no numerical value

def round_values(decimals, *args):
    """
    Rounds numeric values in any arguments, including DataFrames, lists and dictionaries.
    """
    return [round_nested(arg, decimals) for arg in args]
