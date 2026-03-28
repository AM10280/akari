import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
# import logging
# import time

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# FITS I/O Utilities

def read_fits_list(fits_list_path):
    """Read FITS file paths from a text file."""
    with open(fits_list_path, 'r') as f:
        return list(map(lambda x: x.strip() + '.fits', f.readlines()))


def read_fits(file):
    """Read data and header from a FITS file."""
    with fits.open(file) as hdul:
        return hdul[0].data, hdul[0].header



def load_fits_stack(file_list, dtype=np.float32):
    """Load FITS files into a preallocated 3D array (N, H, W)."""
    first, _ = read_fits(file_list[0])
    stack = np.empty((len(file_list), *first.shape), dtype=dtype)

    stack[0] = first
    for i, f in enumerate(file_list[1:], start=1):
        stack[i], _ = read_fits(f)

    return stack



def save_fits(file, filename, outdir='./profile_image'):
    os.makedirs(outdir, exist_ok=True)
    hdu = fits.PrimaryHDU(file)
    hdu.writeto(os.path.join(outdir, f"{filename}.fits"), overwrite=True)



def write_fits(filename, data, header=None, outdir="./destripes"):
    """Write FITS file."""
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, f"{filename}.fits")
    fits.writeto(output_path, data, header, overwrite=True)


# Optional: Profile Plotting


def save_profile(profile_x, season, outdir='./profile_ave_csv'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(season).replace('.txt', '')
    np.savetxt(os.path.join(outdir, f"{filename}.csv"), profile_x)

def save_profile_diff(profile_x, season, outdir='./profile_ave_csv'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(season).replace('.txt', '_d')

    profile_filename = os.path.join('profile_ave_csv', f'{filename}.csv')
    np.savetxt(profile_filename, profile_x)



def plot_profile(profile_x, season, outdir='./profile_ave'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(season).replace('.txt', '')

    plt.figure(figsize=(12, 6))
    plt.plot(profile_x, label=filename, color='C0', alpha=0.7)
    # plt.plot(median_profile, label='Median Profile', color='r', alpha=0.7)
    # plt.ylim(-16.2, 7)
    # plt.ylim(0, 225)
    # plt.ylim(320, 410)
    # plt.ylim(390, 410)
    # plt.ylim(0, 250)
    # plt.ylim(-34, 11) # average, each 2 rows
    plt.xlabel('Pixel (X-direction)')
    plt.ylabel('Power')
    plt.title(filename)
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(outdir, f'{filename}.png')
    plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()



def plot_profile_diff(profile_x, season, outdir='./profile_ave_diff'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(season).replace('.txt', '')

    plt.figure(figsize=(12, 6))
    plt.plot(profile_x, label=filename, color='C0', alpha=0.7)
    # plt.ylim(-16.2, 7)
    # plt.ylim(0, 225)
    # plt.ylim(-34, 11) # average, each 2 rows
    plt.xlabel('Pixel (X-direction)')
    plt.ylabel('Power')
    plt.title('Average Profiles')
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(outdir, f'{filename}.png')
    plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# DATA PROCESSING


def get_band_ranges(band):
    """Return X ranges based on observation band."""
    band = band.upper()

    if band == "S":       # 9 µm
        return [(6, 63), (69, 126)]
    elif band == "L":     # 18 µm
        return [(1, 58), (64, 122)]
    else:
        raise ValueError(f"Unknown band: {band}")


def de_stripes(file, pattern_image, outdir='./destripes'):
    """Apply stripe correction to a single FITS file."""
    """Remove stripes and save corrected FITS images."""
    
    # start_time = time.process_time()

    data, header = read_fits(file)
    # Ensure consistent dtype (avoid repeated casting later)
    data = data.astype(np.float32, copy=False)

    # Apply sigma clipping to remove outliers before subtracting the pattern
    # data = sigma_clipping(data)

    # Subtract pattern image from original FITS data
    corrected_data = data - pattern_image

    filename = os.path.basename(file).replace(".fits", "")
    write_fits(filename, corrected_data, header, outdir)

    # logger.info("経過時間 1枚： %s", time.process_time() - start_time)
    return


def de_stripes_outer(fits_files, season, band, sigma=3, maxiters=5):
    """Compute the pattern image for all FITS files."""
    # Load all FITS images into a 3D stack (n_images, height, width)

    # start_time = time.process_time()

    data_stack = load_fits_stack(fits_files)

    # Apply sigma clipping to exclude outliers across the stack
    # Apply sigma clipping along the image stack (axis=0: across images at the same location)
    # Apply sigma-clipping along the first axis (images) for each (x, y)
    clipped_stack = sigma_clip(data_stack, sigma=sigma, maxiters=maxiters, axis=0)
    
    # Replace outliers with NaN for further processing
    # Compute the mean image after outlier exclusion
    # Compute the mean of unclipped (valid) values
    mean_image = np.nanmean(clipped_stack.filled(np.nan), axis=0)

    # Create a profile image for pattern removal
    mean_image_cropped = mean_image[:-2, :]  # Exclude the last 2 rows

    # Create a mean profile along the X-direction  # mean or median ?
    # profile_x = np.mean(mean_image_cropped, axis=0)
    profile_x = mean_image[:-2].mean(axis=0)

    # save_profile(profile_x, season)  # for debugging
    # plot_profile(profile_x, season, outdir='./profile_ave')  # for debugging

    
    # Normalize and prepare the pattern image
    # Normalize profile by subtracting mean in given ranges.
    # Compute the difference in specified X ranges and normalize
    x_ranges = get_band_ranges(band)
    profile_diff = np.zeros_like(profile_x)
    for start, end in x_ranges:
        segment = profile_x[start:end + 1]
        profile_diff[start:end + 1] = segment - segment.mean()   # the difference from the average

    # save_profile_diff(profile_diff, season)  # for debugging
    # plot_profile_diff(profile_diff, season, outdir='./profile_ave_diff')  # for debugging

    # Create a 2D pattern image based on the 1D profile difference
    pattern_image = np.tile(profile_diff, (mean_image.shape[0], 1))
    # pattern_image = np.broadcast_to(profile_diff, (mean_image.shape[0], profile_diff.size)).copy()
    # save_fits(pattern_image, 'pattern_image')  # for debugging
    
    # logger.info("経過時間 補正用画像： %s", time.process_time() - start_time)

    return pattern_image


def de_stripes_list(fits_list_path, band):
    """Main function to process all files."""
    fits_files = read_fits_list(fits_list_path)
    pattern_image = de_stripes_outer(fits_files, fits_list_path, band)
    for file in fits_files:
        de_stripes(file, pattern_image)


# Entry Point

if __name__ == "__main__":
    fits_list_path = sys.argv[1]
    band = sys.argv[2]
    
    de_stripes_list(fits_list_path, band)
