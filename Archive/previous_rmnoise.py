import os
import sys
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import uniform_filter, generic_filter, gaussian_filter
from scipy import stats
from astropy.io import fits
# from astropy.stats import sigma_clip
from astropy.convolution import convolve, Gaussian2DKernel, Box2DKernel, interpolate_replace_nans
import matplotlib.pyplot as plt
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def read_fits(file):
    """Reads a FITS file and returns the image data and header."""
    with fits.open(file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header

def write_fits(file, data, header):
    """Writes data to a FITS file with the provided header."""
    hdu = fits.ImageHDU(data, header=header)
    primary_hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(file, overwrite=True)




# Read the list of FITS files
def read_fits_list(fits_list_path):
    with open(fits_list_path, 'r') as file:
        file_paths = file.read().splitlines()
    return file_paths



def replace_nans(image):
    # Replace NaN values with interpolated values
        stddev = 1.0
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            # kernel = Gaussian2DKernel(stddev, stddev)
            kernel = Gaussian2DKernel(stddev)
            image = interpolate_replace_nans(image, kernel)

            if np.isnan(image).sum() == 0:
                break
            stddev += 1
            iteration += 1
        if iteration < max_iterations:
            logger.info("Interpolated by a Gaussian 2D kernel with the stddev of %s", stddev)
        return image



def nan_uniform_filter(image, size):
    return generic_filter(image, np.nanmean, size=size, mode='constant', cval=np.nan)


def nan_box2_filter(image, width):
    kernel = Box2DKernel(width=width)
    smoothed_data = convolve(image, kernel, boundary='extend', nan_treatment='interpolate')
    return smoothed_data


def nan_gaussian_filter(image, sigma):
    kernel = Gaussian2DKernel(x_stddev=sigma)
    smoothed_data = convolve(image, kernel, boundary='extend', nan_treatment='interpolate')
    return smoothed_data



def nan_gaussian_filter2(image, sigma=5.):
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










def field_peri_noise_reduction(image, mode=0, xlim=35, ylim=200, seed=None):
    """
    Suppress noise by scaling

    mode 
        0: y >= ylim
        1: rhombus rectangle, x >= xlim & y >= ylim
        2: ellipse, x >= xlim & y >= ylim
    """

    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input image must be a 2D numpy array.")

    dat = image.astype(float).copy()
    # Dimensions of the image  # note reversed order
    ny, nx = dat.shape


    # --------------------------------------------------
    # 1. Define protected area (pct)
    # --------------------------------------------------
    # Protect specific regions based on mode
    pct = np.zeros((ny, nx), dtype=bool) # False/True

    if mode == 0:
        logger.info("Mode = 0: Protecting y >= ylim.")
        pct[ylim:, :] = True
    elif mode == 1:
        logger.info("Mode = 1: Protecting rectangular region.")
        pct[ylim:, xlim:] = True
    elif mode == 2:
        logger.info("Mode = 2: Protecting elliptical region.")

        ex = float(nx - xlim) # semi-major axis
        ey = float(ny - ylim) # semi-minor axis
        # Create a grid of coordinates
        y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        distance = ((x - nx) / ex) ** 2 + ((y - ny) / ey) ** 2
        # Create the elliptical mask (1 inside the ellipse, 0 outside)
        pct[distance <= 1.0] = True
        # pcnt = np.count_nonzero(pct) # the number of True
        # pidx = np.where(pct.ravel(order='F') == 1)[0] # Or better: use boolean masks directly.
    else:
        raise ValueError("Mode must be 0, 1, or 2.")

    pidx = pct

    # --------------------------------------------------
    # 2. Prepare mask and threshold for untouched pixels
    # --------------------------------------------------
    # Protected pixel indices
    msk = np.zeros_like(dat, dtype=bool)

    # 2. Pre-mask known noise regions

    # STDDEV of protected area
    if pidx.any():
        sgm = np.nanstd(dat[pidx])
    else:
        sgm = np.nanstd(dat)

    # set the threshold for untouch pixels.
    pmax = 3.0 * sgm

    # pixels small enough to be untouched
    nidx = np.abs(dat) <= pmax
    # ncnt = np.count_nonzero(nidx) # the number of True

    # protect region
    if pidx.any():
        msk[pidx] = True
    
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
        ave = np.nanmean(dat)
        sgm = np.nanstd(dat)
        cond = (np.abs(dat) > pmax) & (np.abs(dat - ave) > 3.0 * sgm)
        # cond = np.abs(dat - ave) > sgm * 3

        if not np.isfinite(sgm) or sgm == 0:
            break

        # cnt = np.sum(cond) # the number of True
        cnt = np.count_nonzero(cond)
        if cnt == 0:
            break

        # Replace outliers with NaN and update the mask
        # Mark noisy pixels in the mask and set them to NaN in the data
        dat[cond] = np.nan
        msk[cond] = True

    # number of NaNs
    nan_n = np.count_nonzero(np.isnan(dat))    
    if nan_n == 0:
        return image, np.zeros_like(image, dtype=int)


    logger.info('FieldPeriNoiseReduction: %d pixels are processed.', nan_n)


    # indices of masked pixels
    msk_int = msk.astype(int)  # bool → int
    fill_idx = msk
    healthy = ~msk

    # --------------------------------------------------
    # 6. Suppress noise by scaling
    # --------------------------------------------------

    # copy entire data to working data and identify pixels above the limit
    # Recalculate the data from the original image ←important
    dat = image.copy()

    # Calculate suppression factor
    sg_normal = np.std(dat[healthy])
    sg_noise = np.std(dat[fill_idx])
    if sg_noise != 0:  # Avoid division by zero
        suppression_factor = sg_normal / sg_noise
        logger.info("Suppress factor = %f", suppression_factor)

        # Suppress noise by scaling
        # Apply suppression factor to noisy pixels
        dat[fill_idx] *= suppression_factor






    # 7. Restore protected region
    # Restore protected and untouched pixels
    # overwrite protected area by the original data
    if pidx.any():
        dat[pidx] = image[pidx]

    # overwrite untouched area by the original data  # Data smaller than pmax should be restored.
    if nidx.any():
        dat[nidx] = image[nidx]

    # Update the image with processed data
    # write back to the data
    image = dat
    # logger.info("The number of NaNs : %d", np.count_nonzero(np.isnan(image)))

    return image, msk_int








def field_peri_noise_reduction_rev7(image, mode=0, xlim=35, ylim=200, seed=None):
    """
    Fill with Gaussian random values

    mode 
        0: y >= ylim
        1: rhombus rectangle, x >= xlim & y >= ylim
        2: ellipse, x >= xlim & y >= ylim

    seed  : int or None (random seed)
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input image must be a 2D numpy array.")

    # image = np.asarray(image, dtype=float)
    dat = image.astype(float).copy()
    # Dimensions of the image  # note reversed order
    ny, nx = dat.shape


    # --------------------------------------------------
    # 1. Define protected area (pct)
    # --------------------------------------------------
    # Protect specific regions based on mode
    pct = np.zeros((ny, nx), dtype=bool) # False/True

    if mode == 0:
        logger.info("Mode = 0: Protecting y >= ylim.")
        pct[ylim:, :] = True
    elif mode == 1:
        logger.info("Mode = 1: Protecting rectangular region.")
        pct[ylim:, xlim:] = True
    elif mode == 2:
        logger.info("Mode = 2: Protecting elliptical region.")
        ex = float(nx - xlim) # semi-major axis
        ey = float(ny - ylim) # semi-minor axis
        # Create a grid of coordinates
        y = (ny - 1) - np.arange(ny)
        x = (nx - 1) - np.arange(nx)
        xx, yy = np.meshgrid(x, y)
        rr = (xx / ex) ** 2 + (yy / ey) ** 2
        # Create the elliptical mask (1 inside the ellipse, 0 outside)
        pct[rr <= 1.0] = True
        # pcnt = np.count_nonzero(pct) # the number of True
        # pidx = np.where(pct.ravel(order='F') == 1)[0] # Or better: use boolean masks directly.
    else:
        raise ValueError("Mode must be 0, 1, or 2.")

    pidx = pct

    # --------------------------------------------------
    # 2. Prepare mask and threshold for untouched pixels
    # --------------------------------------------------
    # Protected pixel indices
    msk = np.zeros_like(dat, dtype=bool)

    # 2. Pre-mask known noise regions

    # STDDEV of protected area
    if pidx.any():
        sgm = np.nanstd(dat[pidx])
    else:
        sgm = np.nanstd(dat)

    # set the threshold for untouch pixels.
    pmax = 3.0 * sgm

    # pixels small enough to be untouched
    nidx = np.abs(dat) <= pmax
    # ncnt = np.count_nonzero(nidx) # the number of True

    # protect region
    if pidx.any():
        msk[pidx] = True
    
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
        ave = np.nanmean(dat)
        sgm = np.nanstd(dat)
        cond = (np.abs(dat) > pmax) & (np.abs(dat - ave) > 3.0 * sgm)

        if not np.isfinite(sgm) or sgm == 0:
            break

        if not cond.any():
            break

        # Replace outliers with NaN and update the mask
        # Mark noisy pixels in the mask and set them to NaN in the data
        dat[cond] = np.nan
        msk[cond] = True

    # number of NaNs
    nan_n = np.count_nonzero(np.isnan(dat))    
    if nan_n == 0:
        return image, np.zeros_like(image, dtype=int)


    logger.info('FieldPeriNoiseReduction: %d pixels are processed.', nan_n)


    # --------------------------------------------------
    # 5. Y-dependent statistics
    # --------------------------------------------------
    # 5. Y-dependent mean and stddev
    # Y-dependent statistics from healthy pixels
    # (row-by-row: collapse X direction) # axis=1

    ave_y = np.nanmean(dat, axis=1)
    sgm_y = np.nanstd(dat, axis=1)

    # indices of masked pixels
    msk_int = msk.astype(int)  # bool → int
    fill_idx = np.where(msk_int == 1)

    # --------------------------------------------------
    # 6. Fill masked pixels with Gaussian random values
    # --------------------------------------------------

    y_indices = fill_idx[0]
    # y_indices, x_indices = fill_idx
    noise = rng.standard_normal(len(y_indices))

    dat[fill_idx] = (
        ave_y[y_indices] +
        noise * sgm_y[y_indices]
    )


    # 7. Restore protected region
    # Restore protected and untouched pixels
    # overwrite protected area by the original data
    if pidx.any():
        dat[pidx] = image[pidx]

    # overwrite untouched area by the original data  # Data smaller than pmax should be restored.
    if nidx.any():
        dat[nidx] = image[nidx]

    # Update the image with processed data
    # write back to the data
    image = dat
    # logger.info("The number of NaNs : %d", np.count_nonzero(np.isnan(image)))

    return image, msk_int






# High-pass filter (Gaussian-based)
def hpfilter(image, ksize=2, siglim=3.0):
    # kernel size = 3→2
    # Create a working copy of the input image
    imw = image.copy()
    
    # Initial threshold and mask bright spots
    med = np.median(imw)
    sig = np.nanstd(imw - med)
    mask = np.abs(imw - med) > siglim * sig
    count = np.count_nonzero(mask)
    cnt_k = count
    # logger.info("High-pass filter: %d pixels masked.", count)

    while count > 0:
        imw[mask] = np.nan
        ims = nan_gaussian_filter(imw, sigma=ksize)

        sig = np.nanstd(imw - ims)
        mask = (imw != 0) & (np.abs(imw - ims) > siglim * sig)
        count = np.count_nonzero(mask)
        # logger.info("High-pass filter: %d pixels masked.", count)
        cnt_k += count

    logger.info("\nHigh-pass filter: %d pixels masked in total.", cnt_k)
    
    ## hpfilter 無視する場合
#    ims = np.zeros_like(im)

    imh = image - ims
    
    return imh, ims




# High-pass filter (Gaussian-based)
def hpfilter2(image, siglim=3.0, ksize=(2.0, 1.0)):
# def hpfilter2(image, ksize=2, siglim=3.0):  # kernel size = 3→2
    """
    # iterative high-pass filter
    # im: input 2D image
    # imh: high-frequency component
    # ims: smooth/background component
    # im = imh + ims
    # loop with changing kernel size
    """
    
    # Create a working copy of the input image
    imw = image.copy()

    # despike
    spikes = despiker5(imw)
    idx = np.where(spikes == 1)
    spikes_cnt = np.count_nonzero(spikes)


    # Initial threshold and mask bright spots
    # --- Sigma clipping ---
    med = np.median(imw)
    sig = np.nanstd(imw - med)
    mask = np.abs(imw - med) > siglim * sig
    count = np.count_nonzero(mask)
    cnt_k = count
    # logger.info("High-pass filter: %d pixels masked.", count)
    
    if spikes_cnt > 0:
        imw[mask] = np.nan
    
    ksize=(2.0, 1.0)

    # Accumulator for smooth component
    ima = np.zeros_like(imw)

    # Iterative smoothing
    for sigma in ksize:
        ims_iter = nan_gaussian_filter(imw, sigma=sigma)
        imw -= ims_iter
        ima += ims_iter  # sum of removed smooth components

    ## hpfilter 無視
    # ims = np.zeros_like(im)

    ims = ima
    imh = image - ims
    
    return imh, ims





# despike - Spike removal
# A simple despiking function that removes spikes from an image.
def despiker(image, sigma=3): # sigma 5→3
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
    
    while True:
        ave = np.nanmean(imw)
        sgm = np.nanstd(imw)

        # replace spikes (outliers) with NaN
        cond = np.abs(imw - ave) > sigma * sgm
        if not cond.any():
            break
        # Mark spikes in the mask and set them to NaN in the data
        imw[cond] = np.nan

    # Fill NaNs using Gaussian smoothing
    imw_filled = replace_nans(imw)

    # The despiked image is the filled working image
    image_despiked = imw_filled

    # The spikes image is the difference between the original and the despiked image
    image_spikes = image - image_despiked

    return image_despiked, image_spikes









# despike - Spike removal # Type 5
# A simple despiking function that removes spikes from an image.
def despiker5(image, sigma=3): # sigma 5→3
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
    imw = image.copy()
    
    # Initialize output mask
    spikes = np.zeros(imw.shape, dtype=np.int8)

    # Calculate the difference between the central pixel and the surrounding pixels in a 3x3 array.
    # Define convolution kernel
    kernel = np.full((3, 3), -1.0 / 8.0)
    kernel[1, 1] = 1.0

    # Convolution
    imc = nan_gaussian_filter(imw, sigma=sigma)

    # Only mask pixels that are significantly brighter than their surroundings.
    # Compute statistics (ignore NaNs)
    ave = np.nanmean(imc)
    sgm = np.nanstd(imc)

    # Detect positive spikes
    # detect only positive pixels
    mask = (imc - ave) > (sigma * sgm)

    # Create spike map
    spikes[mask] = 1

    # For debugging
    # save_fits('spikes_imw.fits', imw)
    # save_fits('spikes_imc.fits', imc)
    # save_fits('spikes_spk.fits', spikes)

    return spikes






def tanzaku_noise_reduction(image, leftright, basename=None, outdir='./', verbose=False, no_hpf=False, no_despike=False):
    # PRESERVE=preserve ? # IDL
    """Perform noise reduction on a tanzaku image."""
    start_time = time.process_time()

    # preserve=None

    if basename is None:
        verbose = False
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    lr = '_L' if leftright == 'LEFT' else '_R' if leftright == 'RIGHT' else ''
    if lr == '':
        print('leftright should be LEFT or RIGHT.')
        return
    
    # Extract target region
    if leftright == "LEFT":
        xran = np.array([0, 58]) + 6
    elif leftright == "RIGHT":
        xran = np.array([0, 58]) + 69
    else:
        raise ValueError("leftright must be LEFT or RIGHT")        
    yran = np.array([0, 300]) + 3
    im_target = image[yran[0]:yran[1], xran[0]:xran[1]].copy()
    
    # High-pass filtering
    if not no_hpf:
        # im_high, im_smth = hpfilter(im_target)
        im_high, im_smth = hpfilter2(im_target)
        if verbose:
            # save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), [im_high, im_smth])
            save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), np.hstack([im_high, im_smth]))
        im_target = im_high
    else:
        im_high = im_target
        im_smth = np.zeros_like(im_target)
    
    # Despiking
    if not no_despike:
        im_dsp, im_spk = despiker(im_target)
        if verbose:
            # save_fits(os.path.join(outdir, basename + '_dsp' + lr + '.fits'), [im_dsp, im_spk])
            save_fits(os.path.join(outdir, basename + '_dsp' + lr + '.fits'), np.hstack([im_dsp, im_spk]))
        im_target = im_dsp
    else: 
        im_dsp = im_target
        im_spk = np.zeros_like(im_target)
    
    # Get the image size
    h, w = im_target.shape

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

    folded_image = im4


    if verbose:
        save_fits(os.path.join(outdir, basename + '_src' + lr + '.fits'), folded_image)
    

    # Evaluate StdDev before processing
    ave = np.nanmean(folded_image)
    sgm = np.nanstd(folded_image)
    logger.info('Input image StdDev = %f', sgm)
    
    # Perform 2D FFT
    fft_image = fftshift(fft2(folded_image))
    #  check if imaginary is small
    logger.info('Total imaginary component (prc) is %s', np.sum(np.abs(np.imag(fft_image))))
    
    
    # abs, real, imaginary components of Fourier transformed image (3 dimension)
    if verbose:
        oim = np.stack([np.abs(fft_image), np.abs(np.real(fft_image)), np.abs(np.imag(fft_image))], axis=0)        
        # oim = np.zeros((h*2, w*2, 3))
        # oim[:, :, 0] = np.abs(fft_image)
        # oim[:, :, 1] = np.abs(np.real(fft_image))        
        # oim[:, :, 2] = np.abs(np.imag(fft_image))

        save_fits(os.path.join(outdir, basename + '_fft' + lr + '.fits'), oim)
    

    # Extract the left top region in Fourier space (original image's size)
    fft_image_o = fft_image[0:h+1, 0:w+1]  # ← Add one more to overlap the central axis section. 軸合わせのため1つ多め
    # Apply a noise reduction filter
    real_fft = np.real(fft_image_o)
    if verbose:
        save_fits(os.path.join(outdir, basename + '_fft_oreal' + lr + '.fits'), np.abs(real_fft))

    # logger.info("finite of real_fft: %d", np.isfinite(real_fft).sum())
    # logger.info("min/max: %f, %f", np.nanmin(real_fft), np.nanmax(real_fft))


    '''    
    if preserve is None or len(preserve) != 2:
        preserve = [35, 200]
    if len(preserve) != 2:
        raise ValueError("preserve must contain exactly two elements")

    xlim, ylim = preserve
    '''
    # fft_masked, mask_area = field_peri_noise_reduction(real_fft)
    # fft_masked, mask_area = field_peri_noise_reduction_rev7(real_fft, mode=2, xlim=preserve[0], ylim==preserve[1])
    # fft_masked, mask_area = field_peri_noise_reduction_rev7(real_fft, mode=2)
    fft_masked, mask_area = field_peri_noise_reduction(real_fft, mode=2)


    if verbose:
        save_fits(os.path.join(outdir, basename + '_fftm' + lr + '.fits'), np.abs(fft_masked))
        save_fits(os.path.join(outdir, basename + '_msk' + lr + '.fits'), np.abs(mask_area))
    
    # logger.info("finite of real_fft: %d", np.isfinite(fft_masked).sum())
    # logger.info("min/max: %f, %f", np.nanmin(fft_masked), np.nanmax(fft_masked))


    # Refold the masked Fourier image (with 1-pixel overlap)
    # Mirror updated far to fa
    fa4 = np.zeros((h * 2, w * 2))
    # fa4 = np.zeros(folded_image)
    
    fa4[0:h+1, 0:w+1] = fft_masked
    fa4[h:2*h, 0:w+1] = np.flip(fft_masked[1:h+1, :], axis=0)  # Vertically mirror
    fa4[0:h+1, w:2*w] = np.flip(fft_masked[:, 1:w+1], axis=1)  # Horizontally mirror (with 1-pixel overlap)
    fa4[h:2*h, w:2*w] = np.flip(np.flip(fft_masked[1:h+1, 1:w+1], axis=0), axis=1)

    folded_fft_masked = fa4
    

    if verbose:
        save_fits(os.path.join(outdir, basename + '_fa4' + lr + '.fits'), np.abs(folded_fft_masked))
    
    # Set imaginary to zero
    folded_fft_masked = folded_fft_masked + 1j * np.zeros_like(folded_fft_masked)
    # folded_fft_masked = np.complex128(folded_fft_masked)

    # Perform inverse Fourier Transform
    # Inverse shift the zero frequency back
    ifft_shifted = ifftshift(folded_fft_masked)

    # Inverse Fourier Transform to recover the image
    # reconstructed_image = np.real(ifft2(ifft_shifted))
    reconstructed_image = ifft2(ifft_shifted)
    logger.info('Total imaginary component (inverse FFT) is %s', np.sum(np.abs(np.imag(reconstructed_image))))
    reconstructed_image = np.real(reconstructed_image)

    logger.info("DEBUG reconstructed_image")
    # logger.info("shape: %s", reconstructed_image.shape)
    logger.info("finite after ifft (reconstructed_image) : %d", np.isfinite(reconstructed_image).sum())
    logger.info("min/max: %f, %f", np.nanmin(reconstructed_image), np.nanmax(reconstructed_image))
    logger.info("min/max: %s, %s", 
        np.nanmin(reconstructed_image) if np.isfinite(reconstructed_image).any() else "NaN",
        np.nanmax(reconstructed_image) if np.isfinite(reconstructed_image).any() else "NaN")

    im_difference = reconstructed_image - folded_image

    if verbose:
        save_fits(os.path.join(outdir, basename + '_rev' + lr + '.fits'), reconstructed_image)
        save_fits(os.path.join(outdir, basename + '_dif' + lr + '.fits'), im_difference)

    # Evaluate StdDev after processing
    ave = np.nanmean(reconstructed_image)
    sgm = np.nanstd(reconstructed_image)
    logger.info("Output image StdDev = %s", sgm)
    
    # Cut to original size
    reconstructed_image_o = reconstructed_image[1:h+1, 1:w+1]

    # Recover removed smooth component and spikes
    reconstructed_image_o += im_spk + im_smth
    
    # Write back to the original image; tanzakudata
    image[yran[0]:yran[1], xran[0]:xran[1]] = reconstructed_image_o


    # 時間計測 Time measurement
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    logger.info("経過時間： %s", elapsed_time)

    return



# Helper function to save a numpy array as a FITS file
def save_fits(filepath, data):
    # logger.info("filepath: %s, length: %d", filepath, len(data))
    if isinstance(data, list) or (isinstance(data, np.ndarray) and data.ndim == 3):
        header=fits.Header()
        hdu = fits.ImageHDU(data)
        hdul = fits.HDUList([fits.PrimaryHDU(header=header), hdu])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU(data)])
    hdul.writeto(filepath, overwrite=True)



# Main routine

def tanzakurmnoise2d(file, leftonly=False, rightonly=False, outdir='./', verbose=False, nodespike=False, nohpf=False, raw=False):
    # PRESERVE=preserve
    """Perform noise reduction and processing on a tanzaku image."""
    start_time = time.process_time()

    if 'outdir' not in locals():
        outdir = './'

    basename = os.path.basename(file).replace('.fits.gz', '').replace('.fits', '')

    if basename is None:
        verbose = False
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found.")

    
    # Read the input tanzaku data
    im0, hd0 = read_fits(file)
    im0 = im0.astype(np.float64)

    # Extract NAXIS1 and NAXIS2 from header
    nx = hd0['NAXIS1']
    ny = hd0['NAXIS2']
    
    # Perform differentiation
    if raw:
#    if 'raw' in locals() and raw:
        im0[:, 0] = 0
        imd = im0 - np.roll(im0, shift=1, axis=1)
        imd[:, 0:2] = 0
    else:
        imd = im0.copy()

    imd_org = imd.copy()
    
    
    # Setting flags
#    verbose = 1 if 'verbose' in locals() and verbose else 0
#    nohpf = 1 if 'nohpf' in locals() and nohpf else 0
#    nodespike = 1 if 'nodespike' in locals() and nodespike else 0
    
    # Apply noise reduction for LEFT and RIGHT if applicable
#    if 'rightonly' not in locals():
    if not rightonly:
        # logger.info('Processing LEFT of %s', file)
        tanzaku_noise_reduction(imd, 'LEFT', basename=basename, outdir=outdir, verbose=verbose, no_hpf=nohpf, no_despike=nodespike)
        # tanzaku_noise_reduction(imd, 'LEFT', basename=basename, outdir=outdir, verbose=verbose, no_hpf=nohpf, no_despike=nodespike, preserve=preserve)
#    if 'leftonly' not in locals():
    if not leftonly:
        # logger.info('Processing RIGHT of %s', file)
        tanzaku_noise_reduction(imd, 'RIGHT', basename=basename, outdir=outdir, verbose=verbose, no_hpf=nohpf, no_despike=nodespike)
        # tanzaku_noise_reduction(imd, 'RIGHT', basename=basename, outdir=outdir, verbose=verbose, no_hpf=nohpf, no_despike=nodespike, preserve=preserve)

    # Integrating to reconstruct the original form
    if raw:
#    if 'raw' in locals() and raw:
        imd[:, 0] = 0
        imi = np.cumsum(imd, axis=1)
        imo = imi.copy()
    else:
        imo = imd.copy()

    if not verbose:
        # Writing output data
        fout = os.path.join(outdir, basename + '.fits')
        fits.writeto(fout, imo, hd0, overwrite=True)


    if verbose:

        # Writing output data
        fout = os.path.join(outdir, basename + '_pnr.fits')
        fits.writeto(fout, imo, hd0, overwrite=True)
        
        # Writing differential data
        ftdf = os.path.join(outdir, basename + '_tdf.fits')
        fits.writeto(ftdf, imd - imd_org, hd0, overwrite=True)

    

    # std = np.nanstd(imd - imd_org, dtype=np.float64)
    # mean = np.nanmean(imd_org, dtype=np.float64)
    # logger.info("TDF StdDev = %f, %f", std, std / mean)

    return
    


def rmnoise_list(fits_list_path):    
    input_files = read_fits_list(fits_list_path)
    for f in input_files:
        file = f + '.fits'
        # tanzakurmnoise2d(file, leftonly=False, rightonly=False, outdir='./after_rmnoise', verbose=True, nodespike=False, nohpf=False, raw=False)
        tanzakurmnoise2d(file, leftonly=False, rightonly=False, outdir='./after_rmnoise', verbose=False, nodespike=False, nohpf=False, raw=False)




if __name__ == "__main__":
    fits_list_path = sys.argv[1]
    rmnoise_list(fits_list_path)





