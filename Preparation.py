import subprocess
import sys
from PIL import Image
import numpy as np
import os
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt

# Function for installing the required packages
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} is being installed...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} was sucessfully installed.")

def load_and_prepare_image(image_path, Filter):
    # Define image-specific parameters as a dictionary
    image_params = {
        "ES_Rechteck_02_0°": {"magnification": 1000, "width_um": 114.2, "height_um": 77.5},
        "ES_Rechteck_03_0°": {"magnification": 2000, "width_um": 57.1, "height_um": 38.68},
        "ES_Rechteck_04_0°": {"magnification": 5000, "width_um": 22.84, "height_um": 15.5},
        "sin_09µm": {"magnification": 1, "width_um": 10, "height_um": 10},
        "sinusphase": {"magnification": 1, "width_um": 10, "height_um": 10},
        "sin": {"magnification": 1, "width_um": 10, "height_um": 10},
        "sinusgedreht": {"magnification": 1, "width_um": 10, "height_um": 10},
        "ES_01_0_grad04": {"magnification": 2000, "width_um": 57.18, "height_um": 38.59},
        # Add more images here
    }
    # load Image
    image = Image.open(image_path).convert('L')
    image_data = np.array(image)
    height_px, width_px = image_data.shape
    # Determine image name and retrieve specific parameters
    image_name = os.path.basename(image_path).split('.')[0]  # Without extension
    
    if image_name in image_params:
        magn = image_params[image_name]["magnification"]
        width_um = image_params[image_name]["width_um"]
        height_um = image_params[image_name]["height_um"]
    else:
        print("Image name not found in parameter table. Please enter values manually.")
        try:
            magn = float(input(f"Enter the magnification for {image_name}: "))
            width_um = float(input(f"  Enter the width [µm] of {image_name}: "))
            height_um = float(input(f"  Enter the height [µm] of {image_name}: "))
        except ValueError:
            print("Invalid input. Please enter a numerical value!!")
            return None
    # Calculate physical size (in µm)
    step_x = width_um/width_px
    step_y = height_um/height_px

    print("\nApplied Options For Filtering:")

    image_data = apply_hanning_and_subtract_mean(image_data, Filter)
    print("\nImage Data:")
    print(f"Image size [px]: {width_px} px (width) x {height_px} px (height)")
    print(f"Image height: {height_um} µm \nImage width: {width_um} µm")
    print(f"Value of step_x: {step_x:.5f} \nValue of step_y: {step_y:.5f}")
    print("\n")
    constants = {
        "image_name": image_name,
        "magnification": magn,
        "width_px": width_px,
        "height_px": height_px,
        "width_um": width_um,
        "height_um": height_um,
        "step_x": step_x,
        "step_y": step_y
        }
    df_c = pd.DataFrame([constants])
    return image_data, df_c

def apply_hanning_and_subtract_mean(image_data, filter_option):
    # If no filter is required
    if filter_option not in [1, 2, 3]:
        print("No filter applied.")
        return image_data
    # Subtract the mean value if necessary
    if filter_option in [1, 2]:
        image_data = image_data - np.mean(image_data)
        print("Mean value was substracted beforehand.")
    # Apply Hanning window if required
    if filter_option in [1, 3]:
        hanning_window = np.outer(np.hanning(image_data.shape[0]), np.hanning(image_data.shape[1]))
        image_data = image_data * hanning_window
        print("Hanning window was applied.")
    return image_data

# Rotate image and crop to largest possible rectangle
def rot_img(image_data):
    plt.imshow(image_data, cmap='gray')
    plt.axis("off")
    plt.title("original")
    plt.show()
    def calculate_rotation_angle(image):
        """
        Function to calculate the angly by which the image need to be rotated
        to align the structures
        
        INPUT: Image Data
        OUTPUT: Rotated Image 

        """
        fshift = np.fft.fftshift(np.fft.fft2(image))
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        cy, cx = np.array(magnitude_spectrum.shape) // 2
        magnitude_spectrum_no_dc = magnitude_spectrum.copy()
        magnitude_spectrum_no_dc[magnitude_spectrum_no_dc == 0] = np.nan
        
        # Apply bandpass filter
       # filtered_spectrum = apply_bandpass_filter(magnitude_spectrum_no_dc, low_cutoff=0, high_cutoff=0.3)
       # Apply Gaussian filter to smoothen
       # filtered_spectrum = apply_gaussian_filter(magnitude_spectrum_no_dc, sigma=2)
       # Apply threshold filter to retain only strong frequencies
       # filtered_spectrum = apply_threshold_filter(magnitude_spectrum_no_dc, threshold=10000)
       
       # max_pos = np.unravel_index(np.argmax(filtered_spectrum), filtered_spectrum.shape)
        max_pos = np.unravel_index(np.argmax(magnitude_spectrum_no_dc), magnitude_spectrum_no_dc.shape)
        angle_rad = np.arctan2(max_pos[0] - cy, max_pos[1] - cx)
        angle_deg = np.degrees(angle_rad)
        rotation_angle = 180 - (90 - angle_deg)
        return rotation_angle + (180 if rotation_angle < 0 else 0)

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, image.shape[::-1], flags=cv2.INTER_LINEAR)

    def largest_rotated_rect(w, h, angle):
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        alpha = angle if quadrant in [0, 2] else math.pi - angle
        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)
        gamma = math.atan2(bb_w, bb_w) if w < h else math.atan2(bb_w, bb_w)
        delta = math.pi - alpha - gamma
        d = h * math.cos(alpha) if w < h else w * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)
        return bb_w - 2 * a * math.cos(gamma), bb_h - 2 * a * math.sin(gamma)

    def crop_around_center(image, width, height):
        image_center = tuple(np.array(image.shape[::-1]) // 2)
        x1, x2 = int(image_center[0] - width // 2), int(image_center[0] + width // 2)
        y1, y2 = int(image_center[1] - height // 2), int(image_center[1] + height // 2)
        return image[y1:y2, x1:x2]

    rotation_angle = calculate_rotation_angle(image_data)
    print(f"Calculated rotation angle: {rotation_angle:.2f} degrees")

    rotated_image = rotate_image(image_data, rotation_angle)
    new_width, new_height = largest_rotated_rect(rotated_image.shape[1], rotated_image.shape[0], math.radians(rotation_angle))
    cropped_image = crop_around_center(rotated_image, int(new_width), int(new_height))
    
    plt.imshow(cropped_image, cmap='gray')
    plt.axis("off")
    plt.title("cropped and rotated image")
    plt.show()
    
    return cropped_image

def apply_bandpass_filter(spectrum, low_cutoff, high_cutoff):
    """
    Apply a bandpass filter to the magnitude spectrum.

    Parameters:
        spectrum (2D array): The magnitude spectrum of the Fourier Transform.
        low_cutoff (float): Minimum distance from the center to retain frequencies.
        high_cutoff (float): Maximum distance from the center to retain frequencies.

    Returns:
        filtered_spectrum (2D array): Spectrum with frequencies outside the range suppressed.
    """
    cy, cx = np.array(spectrum.shape) // 2
    rows, cols = spectrum.shape
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create a mask for the bandpass filter
    mask = (distance_from_center >= low_cutoff) & (distance_from_center <= high_cutoff)
    filtered_spectrum = np.copy(spectrum)
    filtered_spectrum[~mask] = np.nan  # Suppress frequencies outside the desired range
    
    return filtered_spectrum

from scipy.ndimage import gaussian_filter

def apply_gaussian_filter(spectrum, sigma):
    """
    Apply a Gaussian filter to smooth the magnitude spectrum.

    Parameters:
        spectrum (2D array): The magnitude spectrum of the Fourier Transform.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        filtered_spectrum (2D array): Smoothed spectrum.
    """
    return gaussian_filter(spectrum, sigma=sigma)


def apply_threshold_filter(spectrum, threshold):
    """
    Apply a threshold to suppress low-magnitude frequencies.

    Parameters:
        spectrum (2D array): The magnitude spectrum of the Fourier Transform.
        threshold (float): Minimum magnitude to retain.

    Returns:
        filtered_spectrum (2D array): Spectrum with low magnitudes suppressed.
    """
    filtered_spectrum = np.copy(spectrum)
    filtered_spectrum[filtered_spectrum < threshold] = np.nan  # Suppress low magnitudes
    return filtered_spectrum
