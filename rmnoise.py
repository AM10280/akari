import os
import sys
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pyfftw.interfaces.scipy_fft as fft
from scipy.ndimage import uniform_filter, generic_filter, gaussian_filter
from scipy import stats
from astropy.io import fits
# from astropy.stats import sigma_clip
from astropy.convolution import convolve, Gaussian2DKernel, Box2DKernel, interpolate_replace_nans
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from typing import Callable, Tuple, Dict
import time


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Configuration

@dataclass
class NoiseReductionConfig:
    # Geometry
    mode: int = 2
    xlim: int = 35
    ylim: int = 200

    # Filtering
    kernel_size: float = 2.0  # 3→2
    hpf_ksize: Tuple[float, ...] = (2.0, 1.0)
    hpf_siglim: float = 3.0

    # Despike
    despike_sigma: float = 3.0  # 5→3
    # max_iter: int = 10

    # FFT
    use_fftshift: bool = True

    # Misc
    seed: int | None = None


@dataclass
class IOConfig:
    basename: str | None = None
    leftonly: bool = False
    rightonly: bool = False
    outdir: str = "./after_rmnoise"
    verbose: bool = False
    no_hpf: bool = False
    no_despike: bool = False
    raw: bool = False



# FITS I/O

def read_fits(file: str):
    """Read a FITS file and return the image data and header."""
    with fits.open(file) as hdul:
        return hdul[0].data, hdul[0].header


def write_fits(file: str, data, header):
    """Writes data to a FITS file with the provided header."""
    primary_hdu = fits.PrimaryHDU(header=header)
    hdu = fits.ImageHDU(data)
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(file, overwrite=True)


def save_fits(filepath, data):
    """
    Helper function to save a numpy array as a FITS file
    Save data to a FITS file.

    - 2D arrays → saved in PrimaryHDU
    - 3D arrays or lists → saved in ImageHDU
    """
    # logger.info("filepath: %s, length: %d", filepath, len(data))
    if isinstance(data, list) or (isinstance(data, np.ndarray) and data.ndim == 3):
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data)])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU(data)])
    hdul.writeto(filepath, overwrite=True)




def read_fits_list(fits_list_path):
    """Read the list of FITS files
    Read a text file containing FITS file paths (one per line)."""
    with open(fits_list_path, 'r') as file:
        file_paths = file.read().splitlines()
        # file_paths = [line.strip() for line in file if line.strip()]
    return file_paths


# Utilities

def replace_nans(image, max_iterations: int = 10):
    """
    Replace NaN values with interpolated values
    Replace NaN values using progressively broader Gaussian interpolation.
    Stops early if all NaNs are replaced.
    """
    stddev = 1.0

    for iteration in range(max_iterations):
        kernel = Gaussian2DKernel(stddev)
        image = interpolate_replace_nans(image, kernel)

        if not np.isnan(image).any():
            logger.info("NaNs fully interpolated by a Gaussian 2D kernel with the stddev of %s", stddev)
            break

        stddev += 1.0
        
    return image



def nan_uniform_filter(image, size):
    """Apply a NaN-aware uniform (mean) filter."""
    return generic_filter(image, np.nanmean, size=size, mode='constant', cval=np.nan)


def nan_box2_filter(image, width):
    """Apply a NaN-aware box (uniform) filter using convolution."""
    kernel = Box2DKernel(width=width)
    smoothed_data = convolve(image, kernel, boundary='extend', nan_treatment='interpolate')
    return smoothed_data


def nan_gaussian_filter(image, sigma):
    """Apply a NaN-aware Gaussian filter using convolution."""
    kernel = Gaussian2DKernel(x_stddev=sigma)
    smoothed_data = convolve(image, kernel, boundary='extend', nan_treatment='interpolate')
    return smoothed_data



def nan_gaussian_filter2(image, sigma):
    """
    Gaussian smoothing that ignores NaNs and renormalizes locally.
    Fill NaN values with Gaussian-weighted mean of surroundings

    Parameters:
    image (2D array): Input image with NaN values
    sigma (float): Standard deviation of Gaussian kernel
    """
    valid = np.isfinite(image).astype(float)
    image_filled = np.nan_to_num(image, nan=0.0)

    smooth_image = gaussian_filter(image_filled, sigma=sigma, mode='nearest')
    smooth_valid = gaussian_filter(valid, sigma=sigma, mode='nearest')
    
    # Compute the weighted mean of surrounding pixels
    with np.errstate(invalid='ignore', divide='ignore'):
        result = smooth_image / smooth_valid
    
    # Replace NaN with the weighted mean
    result[smooth_valid == 0] = np.nan
    return result






# Preprocessing


# High-pass filter (Gaussian-based)
def hpfilter(image, ksize=NoiseReductionConfig.kernel_size, siglim=NoiseReductionConfig.hpf_siglim):
    # kernel size = 3→2
    # Create a working copy of the input image
    imw = image.astype(np.float32, copy=True)
    
    # Initial threshold and mask bright spots
    med = np.median(imw)
    sig = np.nanstd(imw - med)
    mask = np.abs(imw - med) > siglim * sig
    count = np.count_nonzero(mask)
    cnt_k = count
    # logger.info("High-pass filter: %d pixels masked.", count)
    imw[mask] = np.nan
    ims_prev = None

    while count > 0:
        ims = nan_gaussian_filter(imw, sigma=ksize)

        diff = imw - ims
        sig = np.nanstd(diff)
        mask = (np.abs(diff) > siglim * sig) & (~np.isnan(imw))

        if not np.any(mask):
            break

        imw[mask] = np.nan
        ims_prev = ims

        count = np.count_nonzero(mask)
        # logger.info("High-pass filter: %d pixels masked.", count)
        cnt_k += count

    logger.info("\nHigh-pass filter: %d pixels masked in total.", cnt_k)
    
    ## hpfilter 無視する場合
#    ims = np.zeros_like(im)

    ims = ims if ims_prev is None else ims_prev
    imh = image - ims
    
    return imh, ims




# High-pass filter (Gaussian-based)
def hpfilter2(image, siglim=NoiseReductionConfig.hpf_siglim, ksize=NoiseReductionConfig.hpf_ksize):
    """
    Iterative Gaussian-based high-pass filter.

    Parameters
    ----------
    image: input 2D image
    siglim : float; Sigma threshold for clipping
    ksize : iterable of float; Gaussian kernel sigmas (applied iteratively)
    
    Returns
    -------
    imh: high-frequency component
    ims: smooth/background component
    image = imh + ims

    loop with changing kernel size
    """
    
    # Create a working copy of the input image
    imw = image.astype(np.float32, copy=True)

    # Despike
    spikes = despiker5(imw)
    spikes_cnt = np.count_nonzero(spikes)
    if np.any(spikes):
        imw[spikes] = np.nan

    # Initial threshold and mask bright spots
    # --- Sigma clipping ---
    med = np.median(imw)
    sig = np.std(imw[~np.isnan(imw)])
    mask = np.abs(imw - med) > siglim * sig
    # count = np.count_nonzero(mask)
    # logger.info("High-pass filter: %d pixels masked.", count)
    
    # Apply masks
    imw[mask] = np.nan

    # Iterative smoothing
    # Accumulator for smooth component
    ims = np.zeros_like(imw)

    # Iterative smoothing
    for sigma in ksize:
        ims_iter = nan_gaussian_filter(imw, sigma=sigma)
        imw -= ims_iter  # progressively removes low-frequency components
        ims += ims_iter  # sum of removed smooth components

    ## hpfilter 無視
    # ims = np.zeros_like(im)

    imh = image - ims
    
    return imh, ims








# despike - Spike removal
# A simple despiking function that removes spikes from an image.
def despiker(image, sigma=NoiseReductionConfig.despike_sigma): # sigma 5→3
    """
    A simple despiking function that removes spikes from an image.

    Parameters:
    image : 2D numpy array
        The input image to be despiked.
    sigma : float, optional
        The Gaussian width in pixels for filling the spikes. Default is 5.

    Returns:
    image_despiked : 2D numpy array
        The image after spikes have been removed and filled.
    image_spikes : 2D numpy array
        The image containing only the spikes that were removed.
    """
    imw = image.copy()
    mask = ~np.isnan(imw)
    
    while True:
        ave = np.mean(imw[mask])
        sgm = np.std(imw[mask])

        # replace spikes (outliers) with NaN
        outliers = np.abs(imw[mask] - ave) > sigma * sgm

        if not outliers.any():
            break

        # Mark spikes in the mask and set them to NaN in the data
        # Get indices of valid pixels
        valid_idx = np.flatnonzero(mask)
        # Map outliers back to flat indices
        outlier_idx = valid_idx[outliers]
        # Set NaNs using flat indexing (fast)
        imw.flat[outlier_idx] = np.nan
        # Update valid mask incrementally
        mask.flat[outlier_idx] = False

    # Fill NaNs using Gaussian smoothing
    image_despiked = replace_nans(imw)

    # The spikes image is the difference between the original and the despiked image
    image_spikes = image - image_despiked

    return image_despiked, image_spikes







# despike - Spike removal # Type 5
# A simple despiking function that removes spikes from an image.
def despiker5(image, sigma=NoiseReductionConfig.despike_sigma): # sigma 5→3
    """
    A simple despiking function that removes spikes from an image.

    Parameters:
    image : 2D numpy array
        The input image to be despiked.
    sigma : float, optional
        The Gaussian width in pixels for filling the spikes. Default is 3.
        Threshold multiplier (default = 3).

    Returns:
    spikes : 2D numpy array (int)
        Binary spike mask (1 = spike, 0 = normal).
    """
    # imw = image.copy()
    imw = image.astype(np.float32, copy=False)

    # Initialize output mask
    # spikes = np.zeros(imw.shape, dtype=np.int8)

    # Convolution
    imc = nan_gaussian_filter(imw, sigma=sigma)

    # Only mask pixels that are significantly brighter than their surroundings.
    # Compute statistics (ignore NaNs)
    valid = ~np.isnan(imc)
    valid_image = imc[valid]
    ave = valid_image.mean()
    sgm = valid_image.std()

    # Detect positive spikes
    # detect only positive pixels
    # Create spike map
    spikes = ((imc - ave) > (sigma * sgm)).astype(np.uint8)

    # For debugging
    # save_fits('spikes_imw.fits', imw)
    # save_fits('spikes_imc.fits', imc)
    # save_fits('spikes_spk.fits', spikes)

    return spikes


# FFT utilities

def mirror(im_target, shape):
    """Mirror-pad image for FFT symmetry."""
    # image size
    h, w = shape
    # Mirror and copy the data
    # Create a new image that is 4 times the size by folding the original image with 1-pixel overlap (to obtain zero imaginary data when FFT)    
    # Mirror and copy the data (the way to obtain zero imaginary data when FFT)
    im4 = np.zeros((h * 2, w * 2))
    
    im4[1:h+1, 1:w+1] = im_target
    im4[h:2*h, 1:w+1] = im_target[::-1, :] # Vertically mirror
    im4[1:h+1, w:2*w] = im_target[:, ::-1] # Horizontally mirror
    im4[h:2*h, w:2*w] = im_target[::-1, ::-1]  # Horizontal and vertical mirror

    # logger.info("shape of im4: %s", im4.shape)
    # logger.info("finite of im4: %d", np.isfinite(im4).sum())

    return im4


def mirror_reconstruct(fft_masked, shape):
    """Reconstruct full FFT from quadrant."""
    # image size
    h, w = shape
    # Refold the masked Fourier image (with 1-pixel overlap)
    # Mirror updated far to fa
    fa4 = np.zeros((h * 2, w * 2))
    
    fa4[:h+1, :w+1] = fft_masked
    fa4[h:2*h, :w+1] = fft_masked[h+1:0:-1, :]  # Vertically mirror
    fa4[:h+1, w:2*w] = fft_masked[:, w:0:-1]  # Horizontally mirror (with 1-pixel overlap)
    fa4[h:2*h, w:2*w] = fft_masked[h+1:0:-1, w:0:-1]

    return fa4


# ================================
# Noise reduction 
# ================================


def field_peri_noise_reduction(image, config: NoiseReductionConfig):
    """
    Noise reduction using masking + sigma clipping + suppression.
    Suppress noise by scaling.

    mode 
        0: y >= ylim
        1: rhombus rectangle, x >= xlim & y >= ylim
        2: ellipse, x >= xlim & y >= ylim

    seed  : int or None (random seed)
    """

    # Setup

    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input image must be a 2D numpy array.")

    dat = image.astype(float, copy=True)
    # Dimensions of the image  # note reversed order
    ny, nx = dat.shape

    rng = np.random.default_rng(config.seed)
    mode, xlim, ylim = config.mode, config.xlim, config.ylim


    # --------------------------------------------------
    # 1. Define protected area (pct)
    # --------------------------------------------------
    # Protect specific regions based on mode
    pct = np.zeros((ny, nx), dtype=bool)

    if mode == 0:
        logger.info("Mode = 0: Protecting y >= ylim.")
        pct[ylim:, :] = True
    elif mode == 1:
        logger.info("Mode = 1: Protecting rectangular region.")
        pct[ylim:, xlim:] = True
    elif mode == 2:
        logger.info("Mode = 2: Protecting elliptical region.")
        ex, ey = float(nx - xlim), float(ny - ylim) # semi-major axis and semi-minor axis
        # Create a grid of coordinates
        y = (ny - 1) - np.arange(ny)
        x = (nx - 1) - np.arange(nx)
        xx, yy = np.meshgrid(x, y)
        # Create the elliptical mask (True: inside the ellipse, False: outside)
        pct = (xx / ex) ** 2 + (yy / ey) ** 2 <= 1.0
        # pcnt = np.count_nonzero(pct)  # the number of True

    else:
        raise ValueError("Mode must be 0, 1, or 2.")

    # --------------------------------------------------
    # 2. Prepare mask and threshold for untouched pixels
    # --------------------------------------------------

    # Pre-mask known noise regions
    base = dat[pct] if pct.any() else dat

    # STDDEV of protected area
    sgm = np.nanstd(base)

    # set the threshold for untouch pixels.
    pmax = 3.0 * sgm

    # untouched pixels
    # pixels small enough to be untouched
    nidx = np.abs(dat) <= pmax
    # ncnt = np.count_nonzero(nidx) # the number of True

    # protect region
    msk = pct.copy()
    
    # --------------------------------------------------
    # 3. Hard-coded noise stripes
    # --------------------------------------------------    
    # Set the mask flag for the specific noisy columns
    bad_x_ranges = [
        (0, 0),
        (11, 14),
        (32, 38),
        (57, 58),
    ]

    for x0, x1 in bad_x_ranges:
        msk[:, x0:x1 + 1] = True

    # Replace pixels where msk is True with NaN
    dat[msk] = np.nan


    # --------------------------------------------------
    # 4. Iterative sigma-clipping
    # --------------------------------------------------
    while True:
        # Calculate mean and standard deviation excluding NaN values
        valid = ~np.isnan(dat)
        if not valid.any():
            break

        vals = dat[valid]
        ave = vals.mean()
        sgm = vals.std()

        if sgm == 0 or not np.isfinite(sgm):
            break

        cond = (np.abs(dat) > pmax) & (np.abs(dat - ave) > 3.0 * sgm)
        if not cond.any():
            break

        # Replace outliers with NaN and update the mask
        # Mark noisy pixels in the mask and set them to NaN in the data
        dat[cond] = np.nan
        # msk[cond] = True
        msk |= cond

    # number of NaNs
    nan_n = np.count_nonzero(np.isnan(dat))
    if nan_n == 0:
        return image, np.zeros_like(image, dtype=int)
    # if not np.isnan(dat).any():
    #     return image, np.zeros_like(image, dtype=int)

    logger.info('FieldPeriNoiseReduction: %d pixels are processed.', nan_n)


    # --------------------------------------------------
    # 6. Suppress noise by scaling
    # --------------------------------------------------

    # indices of masked pixels
    fill_idx = msk
    healthy = ~msk

    # copy entire data to working data and identify pixels above the limit
    # Recalculate the data from the original image ←important
    # dat = image.copy()
    dat = image.astype(float, copy=True)

    # Calculate suppression factor
    if healthy.any() and fill_idx.any():
        sg_normal = dat[healthy].std()
        sg_noise = dat[fill_idx].std()

        if sg_noise != 0:  # Avoid division by zero
            # Suppress noise by scaling
            # Apply suppression factor to noisy pixels
            dat[fill_idx] *= (sg_normal / sg_noise)
            logger.info("Suppress factor = %f", sg_normal / sg_noise)


    # -----------------------------
    # 7. Restore protected + untouched region
    # -----------------------------
    # Restore protected region
    # Restore protected and untouched pixels

    # overwrite protected area by the original data
    if pct.any():
        dat[pct] = image[pct]

    # overwrite untouched area by the original data  # Data smaller than pmax should be restored.
    if nidx.any():
        dat[nidx] = image[nidx]

    # Update the image with processed data
    # write back to the data
    image = dat
    # logger.info("The number of NaNs : %d", np.count_nonzero(np.isnan(image)))

    return image, msk.astype(int)





def field_peri_noise_reduction_rev7(image, config: NoiseReductionConfig):
    """
    Fill with Gaussian random values

    mode 
        0: y >= ylim
        1: rhombus rectangle, x >= xlim & y >= ylim
        2: ellipse, x >= xlim & y >= ylim

    seed  : int or None (random seed)
    """

    # Setup

    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input image must be a 2D numpy array.")

    dat = image.astype(float, copy=True)
    # Dimensions of the image  # note reversed order
    ny, nx = dat.shape

    rng = np.random.default_rng(config.seed)
    mode, xlim, ylim = config.mode, config.xlim, config.ylim


    # --------------------------------------------------
    # 1. Define protected area (pct)
    # --------------------------------------------------
    # Protect specific regions based on mode
    pct = np.zeros((ny, nx), dtype=bool)

    if mode == 0:
        logger.info("Mode = 0: Protecting y >= ylim.")
        pct[ylim:, :] = True
    elif mode == 1:
        logger.info("Mode = 1: Protecting rectangular region.")
        pct[ylim:, xlim:] = True
    elif mode == 2:
        logger.info("Mode = 2: Protecting elliptical region.")
        ex, ey = float(nx - xlim), float(ny - ylim) # semi-major axis and semi-minor axis
        # Create a grid of coordinates
        y = (ny - 1) - np.arange(ny)
        x = (nx - 1) - np.arange(nx)
        xx, yy = np.meshgrid(x, y)
        # Create the elliptical mask (True: inside the ellipse, False: outside)
        pct = (xx / ex) ** 2 + (yy / ey) ** 2 <= 1.0
        # pcnt = np.count_nonzero(pct)  # the number of True

    else:
        raise ValueError("Mode must be 0, 1, or 2.")

    # --------------------------------------------------
    # 2. Prepare mask and threshold for untouched pixels
    # --------------------------------------------------

    # Pre-mask known noise regions
    base = dat[pct] if pct.any() else dat

    # STDDEV of protected area
    sgm = np.nanstd(base)

    # set the threshold for untouch pixels.
    pmax = 3.0 * sgm

    # untouched pixels
    # pixels small enough to be untouched
    nidx = np.abs(dat) <= pmax
    # ncnt = np.count_nonzero(nidx) # the number of True

    # protect region
    msk = pct.copy()
    
    # --------------------------------------------------
    # 3. Hard-coded noise stripes
    # --------------------------------------------------    
    # Set the mask flag for the specific noisy columns
    bad_x_ranges = [
        (0, 0),
        (11, 14),
        (32, 38),
        (57, 58),
    ]

    for x0, x1 in bad_x_ranges:
        msk[:, x0:x1 + 1] = True

    # Replace pixels where msk is True with NaN
    dat[msk] = np.nan


    # --------------------------------------------------
    # 4. Iterative sigma-clipping
    # --------------------------------------------------
    while True:
        # Calculate mean and standard deviation excluding NaN values
        valid = ~np.isnan(dat)
        if not valid.any():
            break

        vals = dat[valid]
        ave = vals.mean()
        sgm = vals.std()

        if sgm == 0 or not np.isfinite(sgm):
            break

        cond = (np.abs(dat) > pmax) & (np.abs(dat - ave) > 3.0 * sgm)

        if not cond.any():
            break

        # Replace outliers with NaN and update the mask
        # Mark noisy pixels in the mask and set them to NaN in the data
        dat[cond] = np.nan
        # msk[cond] = True
        msk |= cond

    # number of NaNs
    nan_n = np.count_nonzero(np.isnan(dat))
    if nan_n == 0:
        return image, np.zeros_like(image, dtype=int)
    # if not np.isnan(dat).any():
    #     return image, np.zeros_like(image, dtype=int)

    logger.info('FieldPeriNoiseReduction: %d pixels are processed.', nan_n)


    # --------------------------------------------------
    # 5. Y-dependent statistics (Row-wise statistics)
    # --------------------------------------------------
    # 5. Y-dependent mean and stddev
    # Y-dependent statistics from healthy pixels
    # (row-by-row: collapse X direction) # axis=1

    ave_y = np.nanmean(dat, axis=1)
    sgm_y = np.nanstd(dat, axis=1)

    # Prevent zero std
    # sgm_y = np.maximum(sgm_y, 1e-6)

    # indices of masked pixels
    # msk_int = msk.astype(int)  # bool → int
    fill_idx = np.where(msk)

    # --------------------------------------------------
    # 6. Fill masked pixels with Gaussian random values
    # --------------------------------------------------

    y_indices = fill_idx[0]
    # y_indices, x_indices = fill_idx
    noise = rng.standard_normal(len(y_indices))
    dat[fill_idx] = ave_y[y_indices] + noise * sgm_y[y_indices]

    # -----------------------------
    # 7. Restore protected + untouched region
    # -----------------------------
    # 7. Restore protected region
    # Restore protected and untouched pixels

    # dat[pct | nidx] = image[pct | nidx]
    
    # restore_mask = pct | nidx
    # dat[restore_mask] = image[restore_mask]

    # overwrite protected area by the original data
    if pct.any():
        dat[pct] = image[pct]

    # overwrite untouched area by the original data  # Data smaller than pmax should be restored.
    if nidx.any():
        dat[nidx] = image[nidx]

    # Update the image with processed data
    # write back to the data
    image = dat
    # logger.info("The number of NaNs : %d", np.count_nonzero(np.isnan(image)))

    return image, msk.astype(int)



# Region handling
# Dictionary
REGIONS = {
    "LEFT":  (slice(3, 303), slice(6, 64), "_L"),
    "RIGHT": (slice(3, 303), slice(69, 127), "_R"),
}

def extract_region(image: np.ndarray, side: str):
    """Extract LEFT and RIGHT detector regions."""
    """Return region view and metadata."""
    y, x, tag = REGIONS[side]
    return image[y, x], y, x, tag


# ======================================
# Core pipeline
# ======================================

def tanzaku_noise_reduction(image, side, basename, config, io):
    """Process a single region (LEFT or RIGHT region) in-place. 
    Process the first and second lines of the tanzaku separately.
    Perform noise reduction on a tanzaku image."""
    start_time = time.process_time()

    # Create a config instance
    # config = NoiseReductionConfig()
    # io = IOConfig()

    os.makedirs(io.outdir, exist_ok=True)
    verbose = io.verbose and basename is not None

    # Process LEFT (row1) or RIGHT (row2) region in-place.
    # Check whether side is "LEFT" or "RIGHT".
    assert side in REGIONS

    # Region selection
    # Extract target region
    region, y, x, lr = extract_region(image, side)
    im_target = region.copy()  # copy once (we modify)


    # ------------------------------
    # High-pass filter
    # ------------------------------
    # High-pass filtering
    if not io.no_hpf:
        # im_high, im_smth = hpfilter(im_target)
        im_target, im_smth = hpfilter2(im_target)
        if verbose:
            save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), [im_target, im_smth])
            # save_fits(os.path.join(io.outdir, basename + '_hpf' + lr + '.fits'), np.hstack([im_target, im_smth]))
    else:
        im_smth = np.zeros_like(im_target)
    
    # ------------------------------
    # Despike
    # ------------------------------
    # Despike
    if not io.no_despike:
        im_target, im_spk = despiker(im_target)
        if verbose:
            save_fits(os.path.join(outdir, basename + '_dsp' + lr + '.fits'), [im_target, im_spk])
            # save_fits(os.path.join(io.outdir, basename + '_dsp' + lr + '.fits'), np.hstack([im_target, im_spk]))
    else: 
        im_spk = np.zeros_like(im_target)
    

    # Get the image size
    shape = im_target.shape

    # Mirror and copy the data
    folded_image = mirror(im_target, shape)

    if verbose:
        save_fits(os.path.join(io.outdir, basename + '_src' + lr + '.fits'), folded_image)

    # Evaluate StdDev before processing
    # ave = np.nanmean(folded_image)
    # sgm = np.nanstd(folded_image)
    logger.info('Input image StdDev = %f', np.nanstd(folded_image))
    
    # ------------------------------
    # FFT
    # ------------------------------
    # Perform 2D FFT
    fft_image = fft2(folded_image, workers=-1)
    # fft_image = fft.fft2(folded_image, workers=-1)
    if config.use_fftshift:
        fft_image = fftshift(fft_image)

    #  check if imaginary is small
    logger.info('Total imaginary component (prc) is %s', np.sum(np.abs(np.imag(fft_image))))
    
    
    # abs, real, imaginary components of Fourier transformed image (3 dimension)
    if verbose:
        oim = np.stack([np.abs(fft_image), np.abs(np.real(fft_image)), np.abs(np.imag(fft_image))], axis=0)        
        # oim = np.zeros((shape[0]*2, shape[1]*2, 3))
        # oim[:, :, 0] = np.abs(fft_image)
        # oim[:, :, 1] = np.abs(np.real(fft_image))        
        # oim[:, :, 2] = np.abs(np.imag(fft_image))

        save_fits(os.path.join(io.outdir, basename + '_fft' + lr + '.fits'), oim)
    
    # Crop
    # Extract the left top region in Fourier space (original image's size)
    fft_image_o = fft_image[:shape[0]+1, :shape[1]+1]  # ← Add one more to overlap the central axis section. 軸合わせのため1つ多め
    # real part
    real_fft = np.real(fft_image_o)

    if verbose:
        save_fits(os.path.join(io.outdir, basename + '_fft_oreal' + lr + '.fits'), np.abs(real_fft))

    # logger.info("finite of real_fft: %d", np.isfinite(real_fft).sum())
    # logger.info("min/max: %f, %f", np.nanmin(real_fft), np.nanmax(real_fft))

    # ------------------------------
    # Noise reduction
    # ------------------------------
    # Apply a noise reduction filter in freq domain
    # fft_masked, mask_area = field_peri_noise_reduction(real_fft, config)
    fft_masked, mask_area = field_peri_noise_reduction_rev7(real_fft, config)

    if verbose:
        save_fits(os.path.join(io.outdir, basename + '_fftm' + lr + '.fits'), np.abs(fft_masked))
        save_fits(os.path.join(io.outdir, basename + '_msk' + lr + '.fits'), np.abs(mask_area))
    
    # logger.info("finite of real_fft: %d", np.isfinite(fft_masked).sum())
    # logger.info("min/max: %f, %f", np.nanmin(fft_masked), np.nanmax(fft_masked))

    # Reconstruct
    # Refold the masked Fourier image (with 1-pixel overlap)
    # Mirror updated fft_masked to folded_fft_masked
    folded_fft_masked = mirror_reconstruct(fft_masked, shape)
    
    if verbose:
        save_fits(os.path.join(io.outdir, basename + '_fa4' + lr + '.fits'), np.abs(folded_fft_masked))
    
    # Set imaginary to zero
    folded_fft_masked = folded_fft_masked + 1j * np.zeros_like(folded_fft_masked)
    # folded_fft_masked = folded_fft_masked.astype(np.complex128)

    # Perform inverse Fourier Transform
    # Inverse shift the zero frequency back
    if config.use_fftshift:
        folded_fft_masked = ifftshift(folded_fft_masked)

    # ------------------------------
    # Inverse FFT
    # ------------------------------
    # Inverse Fourier Transform to recover the image
    # reconstructed_image = ifft2(folded_fft_masked)
    # logger.info('Total imaginary component (inverse FFT) is %s', np.sum(np.abs(np.imag(reconstructed_image))))
    # reconstructed_image = np.real(reconstructed_image)
    reconstructed_image = np.real(ifft2(folded_fft_masked, workers=-1))
    # reconstructed_image = np.real(fft.ifft2(folded_fft_masked, workers=-1))

    '''
    logger.info("DEBUG reconstructed_image")
    logger.info("shape: %s", reconstructed_image.shape)
    logger.info("finite after ifft (reconstructed_image) : %d", np.isfinite(reconstructed_image).sum())
    logger.info("min/max: %f, %f", np.nanmin(reconstructed_image), np.nanmax(reconstructed_image))
    logger.info("min/max: %s, %s", 
        np.nanmin(reconstructed_image) if np.isfinite(reconstructed_image).any() else "NaN",
        np.nanmax(reconstructed_image) if np.isfinite(reconstructed_image).any() else "NaN")
    '''

    im_difference = reconstructed_image - folded_image

    if verbose:
        save_fits(os.path.join(io.outdir, basename + '_rev' + lr + '.fits'), reconstructed_image)
        save_fits(os.path.join(io.outdir, basename + '_dif' + lr + '.fits'), im_difference)

    # Evaluate StdDev after processing
    # ave = np.nanmean(reconstructed_image)
    # sgm = np.nanstd(reconstructed_image)
    logger.info("Output image StdDev = %s", np.nanstd(reconstructed_image))
    
    # ------------------------------
    # Crop back to original and restore components
    # ------------------------------
    # Cut to original size
    reconstructed_image_o = reconstructed_image[1:shape[0]+1, 1:shape[1]+1]

    # Recover removed smooth component and spikes
    reconstructed_image_o += im_spk + im_smth
    
    # Write back to the original image; tanzakudata
    image[y, x] = reconstructed_image_o


    # 時間計測 Time measurement
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    logger.info("経過時間： %s", elapsed_time)
    # logger.info("Elapsed time: %.3f sec", time.process_time() - start_time)

    return


# =======================
# Main routine
# =======================

def tanzaku_rmnoise_2d(file, config, io):
    """
    Main processing function.
    Perform noise reduction and processing on a tanzaku image.
    """
    start_time = time.process_time()

    if not os.path.exists(file):
        raise FileNotFoundError(file)

    os.makedirs(io.outdir, exist_ok=True)

    basename = os.path.basename(file).replace('.fits.gz', '').replace('.fits', '')

    # Read FITS
    # Read the input tanzaku data
    im0, hd0 = read_fits(file)
    im0 = im0.astype(np.float32)

    # Extract NAXIS1 and NAXIS2 from header
    # nx = hd0['NAXIS1']
    # ny = hd0['NAXIS2']
    
    # Optional differentiation
    # Perform differentiation
    if io.raw:
        im0[:, 0] = 0
        imd = im0 - np.roll(im0, shift=1, axis=1)
        imd[:, :2] = 0
    else:
        imd = im0.copy()

    imd_org = imd.copy()
    

    # Process regions (left and right)
    # Apply noise reduction for LEFT (row1) and RIGHT (row2) if applicable
    if not io.rightonly:
        # logger.info('Processing LEFT of %s', file)
        tanzaku_noise_reduction(imd, "LEFT", basename, config, io)
    if not io.leftonly:
        # logger.info('Processing RIGHT of %s', file)
        tanzaku_noise_reduction(imd, "RIGHT", basename, config, io)

    # Reconstruct
    # Integrate back
    # Integrating to reconstruct the original form
    if io.raw:
        imd[:, 0] = 0
        imo = np.cumsum(imd, axis=1)
    else:
        imo = imd

    # Save output
    # Writing output data
    suffix = "_pnr" if io.verbose else ""
    fout = os.path.join(io.outdir, basename + suffix + '.fits')
    fits.writeto(fout, imo, hd0, overwrite=True)

    if io.verbose:        
        # Writing differential data
        ftdf = os.path.join(io.outdir, basename + '_tdf.fits')
        fits.writeto(ftdf, imd - imd_org, hd0, overwrite=True)

    # std = np.nanstd(imd - imd_org, dtype=np.float64)
    # mean = np.nanmean(imd_org, dtype=np.float64)
    # logger.info("TDF StdDev = %f, %f", std, std / mean)
    # logger.info("Total elapsed: %.3f sec", time.process_time() - start_time)

    return
    


def rmnoise_list(fits_list_path):    
    """ Batch processing. """
    input_files = read_fits_list(fits_list_path)
    
    # Initialize configs
    config = NoiseReductionConfig()
    io = IOConfig()

    for f in input_files:
        file = f + '.fits'
        tanzaku_rmnoise_2d(file, config = config, io = io)

def main():
    try:
        fits_list_path = sys.argv[1]
    except IndexError:
        raise SystemExit("Usage: python rmnoise.py <fits_list_path>")

    rmnoise_list(fits_list_path)


if __name__ == "__main__":
    main()
