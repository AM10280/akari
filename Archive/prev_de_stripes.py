import sys
import os
import numpy as np
# import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel, interpolate_replace_nans
# from multiprocessing import Pool
import time


def read_fits_list(fits_list_path):
    """Read list of FITS files from a text file."""
    with open(fits_list_path, 'r') as file:
        return file.read().splitlines()

def read_fits(file):
    """Read data and header from a FITS file."""
    with fits.open(file) as hdul:
        return hdul[0].data, hdul[0].header


def load_fits_images(fits_files):
    """Load all FITS images into a list of data arrays."""
    return [read_fits(file)[0] for file in fits_files]

'''
def load_fits_images(fits_files):
    """Load data arrays from all FITS files using parallel processing."""
    with Pool() as pool:
        return pool.map(lambda file: read_fits(file)[0], fits_files)
'''

def despike(data, threshold=5.0):
    """Despike data by replacing spikes with the median.
    replace values greater than 'threshold' times the std with median"""
    median_val = np.median(data)
    std_dev = np.std(data)
    mask = np.abs(data - median_val) > threshold * std_dev
    data[mask] = median_val
    return data


# gauss_fill - Fill NaN values with Gaussian-weighted mean of surroundings
def gauss_fill(image, sigma=5.):
    # Find the indices of NaN values in the image
    mask = np.where(np.isnan(image))
    rows, cols = image.shape
    cj, ci  = np.meshgrid(range(cols), range(rows))
    # Iterate over each NaN value
    for i, j in zip(mask[0], mask[1]):
        gg = np.exp(-(((ci-i)/sigma)**2+((cj-j)/sigma)**2)/2)
        gg[i, j] = np.nan
        weighted_sum = np.nansum(image * gg)
        weight_total = np.nansum(gg)
        # Replace NaN with the weighted mean
        image[i, j] = weighted_sum / weight_total if weight_total != 0 else np.nan

    return image



def gauss_fill3(image, sigma=5.0):
    """Fill NaN values using my own Gaussian filter.
     gauss_fill - Fill NaN values with Gaussian-weighted mean of surroundings"""
    mask = np.isnan(image)
    rows, cols = image.shape
    cj, ci  = np.meshgrid(range(cols), range(rows)) #誤差が少ない
    # ci, cj = np.meshgrid(np.arange(cols), np.arange(rows)) #高速とされる
    
    for i, j in zip(*np.where(mask)):
        # gg = np.exp(-(((ci-i)/sigma)**2+((cj-j)/sigma)**2)/2)      
        gg = np.exp(-(((ci - i) ** 2 + (cj - j) ** 2) / (2 * sigma ** 2)))
        #gg = np.exp(-(((ci - j) ** 2 + (cj - i) ** 2) / (2 * sigma ** 2)))
        gg[mask] = 0  # Set weights for NaN positions to 0
        weighted_sum = np.nansum(image * gg)
        weight_total = np.nansum(gg)
        image[i, j] = weighted_sum / weight_total if weight_total != 0 else np.nan
    return image



def gauss_fill2(image, sigma=5.0):
    """Fill NaN values using my own Gaussian filter.
     gauss_fill - Fill NaN values with Gaussian-weighted mean of surroundings"""
    mask = np.isnan(image)
    rows, cols = image.shape
    cj, ci  = np.meshgrid(range(cols), range(rows)) #誤差が少ない
    # ci, cj = np.meshgrid(np.arange(cols), np.arange(rows)) #高速とされる
    
    for i, j in zip(*np.where(mask)):
        gg = np.exp(-(((ci - j) ** 2 + (cj - i) ** 2) / (2 * sigma ** 2)))
        gg[mask] = 0  # Set weights for NaN positions to 0
        weighted_sum = np.nansum(image * gg)
        weight_total = np.nansum(gg)
        image[i, j] = weighted_sum / weight_total if weight_total != 0 else np.nan
    return image




def replace_nans(image, stddev):
    # Replace NaN values with interpolated values
        stddev = 1.0
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            kernel = Gaussian2DKernel(stddev, stddev)
            # astropy_conv = convolve(image, kernel)
            image = interpolate_replace_nans(image, kernel)

            if np.isnan(image).sum() == 0:
                break
            stddev += 1
            iteration += 1
            image = image
        if iteration < max_iterations:
            print(f"Interpolated by a Gaussian 2D kernel with the stddev of {stddev}")
        return image



def despiker(image, sigma=5.0, threshold=5.0, max_iterations=10):
    """Enhanced despiking with iterative replacement and Gaussian smoothing."""
    imw = np.copy(image)
    iterations = 0
    while iterations < max_iterations:
        ave = np.nanmean(imw)
        sgm = np.nanstd(imw)
        mask = np.abs(imw - ave) > threshold * sgm
        if not np.any(mask):
            break
        imw[mask] = np.nan
        iterations += 1
    return gauss_fill(imw, sigma)


def despiker3(image, sigma=5):
    """Enhanced despiking using iterative replacement and Gaussian smoothing."""
    imw = np.copy(image)
    while True:
        ave = np.nanmean(imw)
        sgm = np.nanstd(imw)
        mask = np.abs(imw - ave) > sigma * sgm
        if not np.any(mask):
            break
        imw[mask] = np.nan

#    return replace_nans(imw, sigma)
    return gauss_fill(imw, sigma)


def write_fits(filename, data, header, outdir):
    """Write FITS file to specified directory."""
    output_path = os.path.join(outdir, f"{filename}.fits")
    fits.writeto(output_path, data, header, overwrite=True)

def rm_stripes(file, pattern_image, outdir='./rmstripes'):
    """Remove stripes and save corrected FITS images."""
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(file).replace('.fits', '')
    data, header = read_fits(file)
    data = data.astype(np.float64)
    # Extract NAXIS1 and NAXIS2 from header
    # nx = hd0['NAXIS1']
    # ny = hd0['NAXIS2']

    # offy = 2
    # ny,nx = (len(data)-offy, len(data[0]))


    # Subtract pattern image from original FITS data
    corrected_data = data - pattern_image
    write_fits(filename, corrected_data, header, outdir)


def create_pattern_image(fits_files):
    """Compute the pattern image for all FITS files."""
    # Load all FITS images into a list of data arrays
    data_list = load_fits_images(fits_files)
    # data_list = [read_fits(file)[0] for file in fits_files]

    # Create a stacked image by despiking each data array  
    # despiked_data = [despike(data) for data in data_list]
    despiked_data = [despiker(data) for data in data_list]
    despiked_data_stack = np.array(despiked_data)
    stacked_image = np.mean(despiked_data_stack, axis=0)
    # stacked_image = np.median(despiked_data_stack, axis=0)  # ← discretized by 24
    
    stacked_image_cropped = stacked_image[:-3, :]  # Exclude the last 3 rows

    # Create a (mean or median) profile along the X-direction
    # profile_x = np.mean(stacked_image_cropped, axis=0)
    profile_x = np.median(stacked_image_cropped, axis=0)

    # Compute the difference in specified X ranges and normalize
    x_range1, x_range2 = (6, 63), (69, 126)
    mean_val1 = np.mean(profile_x[x_range1[0]:x_range1[1]+1])
    mean_val2 = np.mean(profile_x[x_range2[0]:x_range2[1]+1])
    profile_diff = np.zeros_like(profile_x)
    profile_diff[x_range1[0]:x_range1[1]+1] = profile_x[x_range1[0]:x_range1[1]+1] - mean_val1
    profile_diff[x_range2[0]:x_range2[1]+1] = profile_x[x_range2[0]:x_range2[1]+1] - mean_val2

    # Create a 2D pattern image based on the 1D profile difference
    return np.tile(profile_diff, (stacked_image.shape[0], 1))

def rm_stripes_list(fits_list_path, for_pattern):
    """Main function to process all files."""
    pattern_files = read_fits_list(for_pattern)
    pattern_image = create_pattern_image(pattern_files)
    fits_files = read_fits_list(fits_list_path)
    for file in fits_files:
        rm_stripes(file, pattern_image)
    # with Pool() as pool:
    #    pool.starmap(rm_stripes, [(file, pattern_image) for file in fits_files])

if __name__ == "__main__":
    fits_list_path = sys.argv[1]
    for_pattern = sys.argv[2]
    rm_stripes_list(fits_list_path, for_pattern)
