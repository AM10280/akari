import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os



# Read the list of FITS files
def read_fits_list(fits_list_path):
    with open(fits_list_path, 'r') as file:
        file_paths = file.read().splitlines()
#        fitsname = os.path.basename(file_paths)
    return file_paths



def read_fits(file):
    with fits.open(file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header



def load_fits_images(fits_files):
    data_list = []
    for file in fits_files:
        with fits.open(file) as hdul:
            data = hdul[0].data
            data_list.append(data)
    return data_list



def despike(data, threshold=5.0):
    # Example despike process: replace values greater than 'threshold' times the std with median
    median_val = np.median(data)
    std_dev = np.std(data)
    data[np.abs(data - median_val) > threshold * std_dev] = median_val
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




# Gaussian 
def despiker3(image, sigma=5):
    imw = np.copy(image)
    
    while True:
        ave = np.nanmean(imw)
        sgm = np.nanstd(imw)

        # replace spikes (outliers) with NaN
        cond = np.abs(imw - ave) > sigma * sgm
        cnt = np.count_nonzero(cond)
        # cnt = np.sum(cond)
        if cnt == 0:
            break
        # Mark spikes in the mask and set them to NaN in the data
        imw[cond] = np.nan

    # Fill NaNs using Gaussian smoothing
    imw_filled = gauss_fill(imw, sigma)
    image_despiked = imw_filled
    # image_spikes = image - image_despiked

    return image_despiked


def create_stack_image(data_list):
    despiked_data = [despike(data) for data in data_list]
    stacked_image = np.mean(despiked_data, axis=0)
    # stacked_image_med = np.median(despiked_data, axis=0)
    return stacked_image

def create_profile_x_direction(stack_image):
    # profile_x_ave = np.mean(stack_image, axis=0)
    profile_x = np.median(stack_image, axis=0)
    return profile_x

def compute_difference(profile_x, x_range1=(6, 63), x_range2=(69, 126)):
    profile_diff = np.zeros_like(profile_x)
    mean_val = np.mean(profile_x)
    profile_diff[x_range1[0]:x_range1[1]+1] = profile_x[x_range1[0]:x_range1[1]+1] - mean_val
    profile_diff[x_range2[0]:x_range2[1]+1] = profile_x[x_range2[0]:x_range2[1]+1] - mean_val
    return profile_diff

def create_corrected_pattern_image(profile_diff, img_shape):
    pattern_image = np.tile(profile_diff, (img_shape[0], 1))
    return pattern_image

def subtract_pattern_from_original(fits_data, pattern_image):
    corrected_data = fits_data - pattern_image
    return corrected_data


def write_fits(filename, data, header, outdir):
    """Writes data to a FITS file with the provided header."""
    hdu = fits.ImageHDU(data, header=header)
#    hdr = fits.Header()
    primary_hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary_hdu, hdu])
    output_path = os.path.join(outdir, filename + '.fits')
    hdul.writeto(output_path, overwrite=True)

    # hdu = fits.PrimaryHDU(data)
    # output_path = f"{outdir}/{filename}.fits"
    # hdu.writeto(output_path, overwrite=True)





def rm_stripes(file, pattern_image):

    if 'outdir' not in locals():
        outdir='./rmstripes'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    filename = os.path.basename(file).replace('.fits', '')

    data, header = read_fits(file)
    data = data.astype(np.float64)

    # Extract NAXIS1 and NAXIS2 from header
    # nx = hd0['NAXIS1']
    # ny = hd0['NAXIS2']

    # offy = 2
    # ny,nx = (len(data)-offy, len(data[0]))



    # Step 5: Subtract the pattern image from each FITS data and save result
    stripes_removed_data = data - pattern_image

    write_fits(filename, stripes_removed_data, header, outdir)




def rm_stripes_outer(fits_files):

    # Step 1: Load and create a stack image
    data_list = load_fits_images(fits_files)
    stacked_image = create_stack_image(data_list)

    # Step 2: Profile in X direction
    profile_x = create_profile_x_direction(stacked_image)


    # Step 3: Compute difference in specified X range and set edges to 0
    profile_diff = compute_difference(profile_x)

    # Step 4: Create a corrected pattern image with the same shape as original data
    img_shape = stacked_image.shape
    pattern_image = create_corrected_pattern_image(profile_diff, img_shape)



    return pattern_image



def rm_stripes_list(fits_list_path):
    input_files = read_fits_list(fits_list_path)
    pattern_image = rm_stripes_outer(input_files)
    for file in input_files:
        # file = f + '.fits'
        # filename = os.path.basename(file).replace('.fits', '')
        rm_stripes(file, pattern_image)



fits_list_path = sys.argv[1]
rm_stripes_list(fits_list_path)




