import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import csv
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Gaussian2DKernel, interpolate_replace_nans
# from multiprocessing import Pool
import time


#mpl.style.use('seaborn-darkgrid')
# plt.style.use("seaborn-v0_8")
# plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn-whitegrid')
# sns.set(font='IPAexGothic')
# sns.set_style("whitegrid")
# sns.set_style("darkgrid")


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


def save_fits(file, filename, outdir='./profile_image'):
    os.makedirs(outdir, exist_ok=True)
    hdu = fits.PrimaryHDU(file)
    hdu.writeto(os.path.join(outdir, f"{filename}.fits"), overwrite=True)
    # hdu.writeto(f"./profile_image/{filename}.fits", overwrite=True)



def write_fits(filename, data, header, outdir):
    """Write FITS file to specified directory."""
    output_path = os.path.join(outdir, f"{filename}.fits")
    fits.writeto(output_path, data, header, overwrite=True)


'''
def gauss_fill(image, sigma=5.0):
    """
    Fill NaN values with a Gaussian-weighted mean of surroundings.
    """
    kernel = Gaussian2DKernel(x_stddev=sigma)
    return interpolate_replace_nans(image, kernel)
'''


# gauss_fill - Fill NaN values with Gaussian-weighted mean of surroundings
def gauss_fill(image, sigma=5.):
    # Find the indices of NaN values in the image
    mask = np.where(np.isnan(image))
    rows, cols = image.shape
    cj, ci = np.meshgrid(range(cols), range(rows))
    # Iterate over each NaN value
    for i, j in zip(mask[0], mask[1]):
        gg = np.exp(-(((ci - i) / sigma) ** 2 + ((cj - j) / sigma) ** 2) / 2)
        gg[i, j] = np.nan
        weighted_sum = np.nansum(image * gg)
        weight_total = np.nansum(gg)
        # Replace NaN with the weighted mean
        image[i, j] = weighted_sum / weight_total if weight_total != 0 else np.nan

    return image






def sigma_clipping2(images, sigma=5.0, threshold=5.0, max_iterations=10):

    # optional iterate
    iterations = 3
    for i in range(iterations):
        # Compute the mean and standard deviation across the stack for each pixel
        mean_image = np.mean(images, axis=0)
        std_image = np.std(images, axis=0)
        # Define a clipping threshold
        sigma = 3
        # Create a mask for the pixels that are not outliers
        mask = np.abs(images - mean_image) <= sigma * std_image
        # Apply the mask: This will create a new image with outliers removed
        clipped_images = images * mask

    # Recompute the mean image after clipping
    mean_clipped_image = np.mean(clipped_images, axis=0)
    return mean_clipped_image


def save_profile(profile_x, season, outdir='./profile_ave_csv'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(season).replace('.txt', '')

    profile_filename = os.path.join('profile_ave_csv', f'{filename}.csv')
    np.savetxt(profile_filename, profile_x)
    # np.savetxt('./profile_ave_csv/1st_season.csv', profile_x, delimiter=',')



def save_profile_diff(profile_x, season, outdir='./profile_ave_csv'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(season).replace('.txt', '_d')

    profile_filename = os.path.join('profile_ave_csv', f'{filename}.csv')
    np.savetxt(profile_filename, profile_x)
    # np.savetxt('./profile_ave_csv/1st_season.csv', profile_x, delimiter=',')



def plot_profile(profile_x, season, outdir='./profile_ave'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(season).replace('.txt', '')

    plt.figure(figsize=(12, 6))
    plt.plot(profile_x, label=filename, color='C0', alpha=0.7)
    # plt.plot(median_profile, label='Median Profile', color='r', alpha=0.7)
    # plt.ylim(-16.2, 7)
    # plt.ylim(0, 225)
    plt.ylim(0, 250)
    # plt.ylim(-34, 11) # average, each 2 rows
    plt.xlabel('Pixel (X-direction)')
    plt.ylabel('Power')
    plt.title('Average Profiles')
    # plt.title('Median Profiles of FITS Image (Y-Integrated)')
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(outdir, f'{filename}.png')
    plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()



def plot_profile_diff(profile_x, season, outdir='./profile_ave_diff'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(season).replace('.txt', '')

    plt.figure(figsize=(12, 6))
    plt.plot(profile_x, label=filename, color='C0', alpha=0.7)
    # plt.plot(median_profile, label='Median Profile', color='r', alpha=0.7)
    # plt.ylim(-16.2, 7)
    # plt.ylim(0, 225)
    plt.ylim(-34, 11) # average, each 2 rows
    plt.xlabel('Pixel (X-direction)')
    plt.ylabel('Power')
    plt.title('Average Profiles')
    # plt.title('Median Profiles of FITS Image (Y-Integrated)')
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(outdir, f'{filename}.png')
    plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()




def de_stripes(file, pattern_image, outdir='./destripes'):
    """Remove stripes and save corrected FITS images."""
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(file).replace('.fits', '')
    data, header = read_fits(file)
    data = data.astype(np.float64)

    # Apply sigma clipping to remove outliers before subtracting the pattern
    # data = apply_sigma_clipping(data)

    # Subtract pattern image from original FITS data
    corrected_data = data - pattern_image
    # save_fits(corrected_data, filename, outdir)
    write_fits(filename, corrected_data, header, outdir)


def de_stripes_outer(fits_files, season):
    """Compute the pattern image for all FITS files."""
    # Load all FITS images into a list of data arrays
    # Load all FITS images into a 3D stack (n_images, height, width)
    data_list = load_fits_images(fits_files)
    # data_list = [read_fits(file)[0] for file in fits_files]
    data_stack = np.array(data_list)

    # despike each data array and create a stacked image
    ## despiked_data = [despiker(data) for data in data_list]
    ## despiked_data_stack = np.array(despiked_data)
    # stacked_image = np.mean(despiked_data_stack, axis=0)
    # stacked_image = np.median(despiked_data_stack, axis=0)  # ← discretized by 24
    
    # Apply sigma clipping to exclude outliers across the stack
    # sigma_clipped_stack = apply_sigma_clipping_group(data_stack)

    # Apply sigma clipping along the image stack (axis=0: across images at the same location)
    # Apply sigma-clipping along the first axis (images) for each (x, y)
    # clipped_stack = sigma_clip(data_stack, sigma=3, maxiters=5, axis=0)
    ## clipped_stack = sigma_clip(despiked_data_stack, sigma=3, maxiters=5, axis=0)
    clipped_stack = sigma_clip(data_stack, sigma=3, maxiters=5, axis=0)
    # print('clipped_stack: ', clipped_stack)
    # clipped_stack = sigma_clip(despiked_data_stack, sigma=3, maxiters=5, axis=0, masked=False)
    
    # Replace outliers with NaN for further processing
    ## sigma_clipped_stack = np.where(clipped_stack.mask, np.nan, despiked_data_stack)
    sigma_clipped_stack = np.where(clipped_stack.mask, np.nan, data_stack)
    # sigma_clipped_stack = clipped_stack.fill(np.nan)
    # sigma_clipped_stack = clipped_stack.fill(np.nanmean(clipped_stack))
    # print('sigma_clipped_stack: ', sigma_clipped_stack)

    # Compute the mean image after outlier exclusion
    # Compute the mean of unclipped (valid) values
    # mean_image = np.mean(clipped_stack, axis=0)
    mean_image = np.nanmean(sigma_clipped_stack, axis=0)

    # original code sigma_clipping
    # mean_image = sigma_clipping2(despiked_data_stack, sigma=3.0, threshold=5.0, max_iterations=5)

    # save_fits(mean_image, 'mean_image')

    # Create a profile image for pattern removal
    mean_image_cropped = mean_image[:-2, :]  # Exclude the last 3 rows
    # save_fits(mean_image_cropped, 'mean_image_cropped')

    # Create a (mean or median) profile along the X-direction
    profile_x = np.mean(mean_image_cropped, axis=0)

    # save_profile(profile_x, season)
    # plot_profile(profile_x, season, outdir='./profile_ave')


    
    # Normalize and prepare the pattern image
    x_range1, x_range2 = (6, 63), (69, 126)
    mean_val1 = np.mean(profile_x[x_range1[0]:x_range1[1] + 1])
    mean_val2 = np.mean(profile_x[x_range2[0]:x_range2[1] + 1])
    profile_diff = np.zeros_like(profile_x)
    profile_diff[x_range1[0]:x_range1[1] + 1] = profile_x[x_range1[0]:x_range1[1] + 1] - mean_val1
    profile_diff[x_range2[0]:x_range2[1] + 1] = profile_x[x_range2[0]:x_range2[1] + 1] - mean_val2

    # save_profile_diff(profile_diff, season)
    # plot_profile_diff(profile_diff, season, outdir='./profile_ave_diff')

    # Create a 2D pattern image based on the 1D profile difference
    # pattern_image_d = np.tile(profile_diff, (mean_image.shape[0], 1))
    # save_fits(pattern_image_d, 'pattern_image_diff')
    
    """
    # Compute the difference in specified X ranges and normalize
    x_range1, x_range2 = (6, 63), (69, 126)
    mean_val1 = np.mean(profile_x[x_range1[0]:x_range1[1] + 1])
    mean_val2 = np.mean(profile_x[x_range2[0]:x_range2[1] + 1])
    profile_section = np.zeros_like(profile_x)
    profile_section[x_range1[0]:x_range1[1] + 1] = profile_x[x_range1[0]:x_range1[1] + 1]
    profile_section[x_range2[0]:x_range2[1] + 1] = profile_x[x_range2[0]:x_range2[1] + 1]



    # Compute the difference in specified X ranges and normalize
    x_range1, x_range2 = (6, 63), (69, 126)
    profile_section = np.zeros_like(profile_x)
    profile_section[x_range1[0]:x_range1[1]+1] = profile_x[x_range1[0]:x_range1[1]+1]
    profile_section[x_range2[0]:x_range2[1]+1] = profile_x[x_range2[0]:x_range2[1]+1]
    """

    # Create a 2D pattern image based on the 1D profile difference
    # pattern_image = np.tile(profile_diff, (stacked_image.shape[0], 1))
    # pattern_image = np.tile(profile_x, (stacked_image.shape[0], 1))
    # pattern_image = np.tile(profile_section, (mean_image.shape[0], 1))
    # pattern_image = np.tile(profile_x, (mean_image.shape[0], 1))
    pattern_image = np.tile(profile_diff, (mean_image.shape[0], 1))

    # save_fits(pattern_image, 'pattern_image')
    
    return pattern_image


def de_stripes_list(fits_list_path):
    """Main function to process all files."""
    fits_files = read_fits_list(fits_list_path)
    pattern_image = de_stripes_outer(fits_files, fits_list_path)
    for file in fits_files:
        de_stripes(file, pattern_image)


if __name__ == "__main__":
    fits_list_path = sys.argv[1]
    de_stripes_list(fits_list_path)
