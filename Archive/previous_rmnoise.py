import sys
import numpy as np
#from scipy.fft import fft2, fftshift
from scipy.fft import fft2, ifft2, fftshift, ifftshift
# from scipy.signal import butter, filtfilt, fftconvolve
from scipy.signal import convolve as scipy_convolve
from scipy.ndimage import uniform_filter, gaussian_filter
# from scipy.ndimage import convolve
from scipy import stats
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, interpolate_replace_nans
# from astropy.fft import fft2, fftshift
# import cv2
import matplotlib.pyplot as plt
import os



def read_fits(file):
    """Reads a FITS file and returns the image data and header."""
    with fits.open(file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header

def write_fits(file, data, header):
    """Writes data to a FITS file with the provided header."""
    hdu = fits.ImageHDU(data, header=header)
#    hdr = fits.Header()
    primary_hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(file, overwrite=True)


'''
'''





# derive stddev of non-signal region.
def field_sigma(image, output_path='output/test.fits'):
    # Copy the image to a local variable
    dat = np.copy(image)
    
    # Repeat until no more outliers are found
    while True:
        ave = np.nanmean(dat)  # Calculate the mean, ignoring NaNs
        sgm = np.nanstd(dat)   # Calculate the standard deviation, ignoring NaNs
        
        # Identify outliers (more than 3 sigma away from the mean)
        outliers = np.abs(dat - ave) > 3 * sgm
        
        # Count the number of outliers
        cnt = np.sum(outliers)
        
        if cnt == 0:
            break  # Exit the loop if no more outliers are found
        
        # Replace outliers with NaN
        dat[outliers] = np.nan
    
    # Write the cleaned data to a FITS file
    fits.writeto(output_path, dat, overwrite=True)

    return ave, sgm    



def replace_nans0(image, stddev):
    # Replace NaN values with interpolated values
    kernel = Gaussian2DKernel(stddev)
    image = interpolate_replace_nans(image, kernel)

    return image


def replace_nans(image, stddev):
    # Replace NaN values with interpolated values
        stddev = 1.0
        max_iterations = 20
        iteration = 0

        # use this for the scipy convolution
        img_zerod = image.copy()
        img_zerod[np.isnan(image)] = 0

        while iteration < max_iterations:
            kernel = Gaussian2DKernel(stddev, stddev)
            # scipy_conv = scipy_convolve(image, kernel, mode='same', method='direct')
            # scipy_conv_zerod = scipy_convolve(img_zerod, kernel, mode='same', method='direct')
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




def gauss_fill(image, sigma=5):
    """
    Fill NaN values in the image with Gaussian-weighted mean of their surroundings.

    Parameters:
    image (2D array): Input image with NaN values
    sigma (float): Standard deviation of Gaussian kernel
    """
    # Find the indices of NaN values in the image
    nan_idx = np.where(np.isnan(image))
    
    # Get image size
    ny, nx = image.shape
    
    # Generate x and y index arrays
    xidx = np.arange(nx)
    yidx = np.arange(ny)
    
    # Create meshgrid for the distances
    X1, Y1 = np.meshgrid(xidx, yidx)
    
    # Iterate over each NaN value
    for iy, ix in zip(nan_idx[0], nan_idx[1]):
        # Compute the squared distance from the current NaN pixel to all others
        rr = (X1 - ix)**2 + (Y1 - iy)**2
        
        # Compute Gaussian weights
        gg = np.exp(-rr / (2 * sigma**2))
        
        # Set the weight of the NaN pixel itself to NaN
        gg[iy, ix] = np.nan
        
        # Compute the weighted mean of surrounding pixels
        weighted_sum = np.nansum(image * gg)
        weight_total = np.nansum(gg)
        
        # Replace NaN with the weighted mean
        image[iy, ix] = weighted_sum / weight_total if weight_total != 0 else np.nan

    return image



def gauss_fill2(image, sigma):
    nan_mask = np.isnan(image)
    filled_image = gaussian_filter(np.nan_to_num(image, nan=0), sigma=sigma)
    filled_image[nan_mask] = np.nan
    normalization = gaussian_filter(~nan_mask.astype(float), sigma=sigma)
    image_filled = filled_image / (normalization + 1e-10)
    return image_filled


# fill NaN with Gaussian weighted mean from surroundings.
def gauss_fill02(image, sigma):
    if len(sigma) <= 0:
        sigma = 5      # Gaussian width in pix
#        return image

    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        return image

    # Gaussian-weighted filling
    filled_image = np.copy(image)
    xidx, yidx = np.indices(image.shape)
    
    for idx in zip(*np.where(~finite_mask)):
#        ix = idx % ny
        ix, iy = idx
        rr = (xidx - ix) ** 2 + (yidx - iy) ** 2
        gg = np.exp(-rr / (2 * sigma ** 2))
        gg[~finite_mask] = 0
        
        im0 = np.nansum(image * gg) / np.nansum(gg)
        filled_image[ix, iy] = im0

    return filled_image





# derive stddev of non-signal region.
def field_peri_noise_reduction(image, yrange=250):

    # Copy the first 250 rows of the image
    dat = image[:, 0:yrange].copy()
    
    # Initialize a mask
    msk = np.zeros(dat.shape, dtype=int)
    
    while True:
        # Calculate mean and standard deviation, ignoring NaNs
        ave = np.nanmean(dat)
        sgm = np.nanstd(dat)
        cond = np.abs(dat - ave) > sgm * 3     ## find mask regions
        cnt = np.sum(cond)
#        print(cnt, c)
        if cnt > 0:
            # Mark noisy pixels in the mask and set them to NaN in the data
            # dat[cond] = np.nan
            dat = np.where(cond, np.nan, dat)
            msk = np.where(cond, 1, msk)
            # img_zerod = img.copy()
            # img_zerod[np.isnan(img)] = 0
        else:
            break
    
    # Separate noisy and normal pixels
    idx = np.where(msk == 1)
    cidx = np.where(msk == 0)
    
    # If there are no noisy or normal pixels, exit
    if len(idx[0]) == 0 or len(cidx[0]) == 0:
        return
    
    print(f'FieldPeriNoiseReduction: {len(idx[0])} pixels are processed.')
    
    # Recalculate the data from the original image ←important
    dat = image[:, 0:yrange].copy()
    
    # Calculate standard deviations for normal and noise pixels
    sg_normal = np.nanstd(dat[cidx])
    sg_noise = np.nanstd(dat[idx])
    
    # Suppress noise by scaling
    dat[idx] *= sg_normal / sg_noise
    print(f"Suppress factor = {sg_normal / sg_noise}")
    
    # Update the image
    image[:, 0:yrange] = dat
    # image[55:59, 0:250] = dat[55:59, :]    # only central component

    return image



# derive stddev of non-signal region.
def field_peri_noise_reduction0(image):
    # Copy the first 250 rows of the image
    dat = image[:, 0:250].copy()
    
    # Initialize a mask
    msk = np.zeros(dat.shape, dtype=int)
    
    while True:
        # Calculate mean and standard deviation, ignoring NaNs
        ave = np.nanmean(dat)
        sgm = np.nanstd(dat)
        cond = np.abs(dat - ave) > sgm * 3
        cnt = np.sum(cond)
#        print(cnt, c)
        if cnt > 0:
            # Mark noisy pixels in the mask and set them to NaN in the data
            dat = np.where(cond, np.nan, dat)
            msk = np.where(cond, 1, msk)
        else:
            break
    
    # Separate noisy and normal pixels
    idx = np.where(msk == 1)
    cidx = np.where(msk == 0)
    
    # If there are no noisy or normal pixels, exit
    if len(idx[0]) == 0 or len(cidx[0]) == 0:
        return
    
    print(f'FieldPeriNoiseReduction: {len(idx[0])} pixels are processed.')
    
    # Recalculate the data from the original image ←important
    dat = image[:, 0:250].copy()
    
    # Calculate standard deviations for normal and noise pixels
    sg_normal = np.nanstd(dat[cidx])
    sg_noise = np.nanstd(dat[idx])
    
    # Suppress noise by scaling
    dat[idx] *= sg_normal / sg_noise
    print(f"Suppress factor = {sg_normal / sg_noise}")
    
    # Update the image
    image[:, 0:250] = dat
    # image[55:59, 0:250] = dat[55:59, :]    # only central component

    return image


# derive stddev of non-signal region.
def field_peri_noise_reduction1(image):
    # Copy the first 250 rows of the image
    dat = image[:, 0:250].copy()
    
    # Initialize a mask
    msk = np.zeros(dat.shape, dtype=int)
    
    while True:
        # Calculate mean and standard deviation, ignoring NaNs
        ave = np.nanmean(dat)
        sgm = np.nanstd(dat)
        cond = np.abs(dat - ave) > sgm * 3
        cnt = np.sum(cond)
        if cnt > 0:
            # Mark noisy pixels in the mask and set them to NaN in the data
            dat = np.where(cond, np.nan, dat)
            msk = np.where(cond, 1, msk)
        else:
            break
    
    # Separate noisy and normal pixels
    nan_n = np.isnan(msk).sum()
    idx = np.where(msk == 1)
    cidx = np.where(msk == 0)
    
    # If there are no noisy or normal pixels, exit
#    if len(idx[0]) == 0 or len(cidx[0]) == 0:
    if np.isnan(msk).sum() == 0:
        return
    
    print(f'FieldPeriNoiseReduction: {len(idx[0])}, {nan_n} pixels are processed.')
    
    # Recalculate the data from the original image ←important
    dat = image[:, 0:250].copy()
    
    # Calculate standard deviations for normal and noise pixels
    sg_normal = np.std(dat[cidx])
    sg_noise = np.std(dat[idx])
    
    # Suppress noise by scaling
    dat[idx] *= sg_normal / sg_noise
    print(f"Suppress factor = {sg_normal / sg_noise}")
    
    # Update the image
    image[:, 0:250] = dat
    # image[55:59, 0:250] = dat[55:59, :]    # only central component

    return image






def hpfilter(im, ksize=3.0, siglim=3.0):
    # Create a working copy of the input image
    imw = im.copy()
    
    # Initial threshold and mask bright spots
    med = np.median(imw)
    sig = np.nanstd(imw - med)
    cond = np.abs(imw - med) > siglim * sig
    idx = np.where(np.abs(imw - med) > siglim * sig)
    cnt_k = len(idx[0])
    count = np.sum(np.abs(imw - med) > siglim * sig)
    while len(idx[0]) > 0:
#    while count > 0:
        imw[idx] = np.nan
        # ims = uniform_filter(np.nan_to_num(imw), size=int(ksize))
        ims = uniform_filter(np.nan_to_num(imw), size=int(ksize))
#        ims = gaussian_filter(np.nan_to_num(imw), sigma=ksize, mode='nearest')
        sig = np.nanstd(imw - ims)
        idx = np.where(np.abs(imw - ims) > siglim * sig)
        cnt_k += len(idx[0])
#        count = np.sum(np.abs(imw - med) > siglim * sig)
    print(f'hpfilter: {cnt_k} pixels masked in total.')
    
    ## hpfilter 無視
#    ims = np.zeros_like(im)

    imh = im - ims
    
    return imh, ims


def hpfilter_a(im, ksize=3.0, siglim=3.0):
    """
    Apply an iterative high-pass filter to extract high-frequency components.
    
    Parameters:
    im (2D array): Input 2D image.
    ksize (float): Kernel size for the smoothing filter.
    siglim (float): Sigma limit for pixel masking based on standard deviation.

    Returns:
    imh (2D array): High-frequency component.
    ims (2D array): Low-frequency (background) component.
    """
    # Create a working copy of the input image
    imw = im.copy()
    
    # Initial threshold and mask bright spots
#    med = np.median(imw)
    med = np.nanmedian(imw)
    sig = np.nanstd(imw - med)
#    sig = stats.median_abs_deviation(imw - med, scale='normal')
    cond = np.abs(imw - med) > siglim * sig
    cnt = np.sum(cond)
    idx = np.where(np.abs(imw - med) > siglim * sig)
    cnt_n = len(idx[0])
    cnt_k = 0
#    while cnt > 0:
    while cnt_k <= cnt_n:
        imw = np.where(cond, np.nan, imw)
        ims = replace_nans0(imw, stddev=2)
#        ims = uniform_filter(np.nan_to_num(imw), size=int(ksize))
        # Recalculate the standard deviation
        sig = np.nanstd(imw - ims)
        # Find new outliers based on the new background estimate
        idx = np.where(np.abs(imw - ims) > siglim * sig)
        # Count additional masked pixels
        cnt_k += len(idx[0])
    
    print(f'hpfilter: {cnt_k} pixels masked in total.')
    
    imh = im - ims
    
    return imh, ims






# iterative high-pass filter
def hpfilter01(im, ksize=3.0, siglim=3.0):
    """
    Iterative high-pass filter for a 2D image.
    
    Parameters:
    im : 2D numpy array
        The input image.
    ksize : int, optional
        The kernel size for the smoothing operation. Default is 3.
    siglim : float, optional
        The sigma limit for detecting outliers. Default is 3.0.
        
    Returns:
    imh : 2D numpy array
        The high-frequency component of the image.
    ims : 2D numpy array
        The background or low-frequency component of the image.
    """

#    if not (isinstance(siglim, (list, tuple)) and len(siglim) == 1):
#        siglim = 3.0
#    if not (isinstance(ksize, (list, tuple)) and len(ksize) == 1):
#        ksize = 3.0

    imw = np.copy(im)  # Copy to working data

    # Initial threshold and mask bright spots
    med = np.nanmedian(imw)
    sig = np.nanstd(imw - med)    
#    sig = stats.median_abs_deviation(imw - med, scale='normal')
    idx = np.where(np.abs(imw - med) > siglim * sig)[0]
    cnt_k = 0
    cnt = len(idx)
    
    while cnt > 0:
        imw[idx] = np.nan  # Mask outliers

        # Apply a smoothing filter (you can switch between uniform and Gaussian)
        # ims = uniform_filter(imw, size=ksize, mode='nearest')
        ims = gaussian_filter(imw, sigma=ksize, mode='nearest')
        
        # Recalculate the standard deviation
        sig = np.nanstd(imw - ims)
        sig = stats.median_abs_deviation(imw - med, scale='normal')
        idx = np.where(np.abs(imw - ims) > siglim * sig)[0]
        cnt = len(idx)
        cnt_k += len(idx)

    print(f'hpfilter: {cnt_k} pixels masked in total.')
    
## hpfilter 無視
    ims = np.zeros_like(im)

    # High-frequency component
    imh = im - ims

    return imh, ims



# iterative high-pass filter
def hpfilter_1(im, ksize=3.0, siglim=3.0):
    """
    Iterative high-pass filter for a 2D image.
    
    Parameters:
    im : 2D numpy array
        The input image.
    ksize : int, optional
        The kernel size for the smoothing operation. Default is 3.
    siglim : float, optional
        The sigma limit for detecting outliers. Default is 3.0.
        
    Returns:
    imh : 2D numpy array
        The high-frequency component of the image.
    ims : 2D numpy array
        The background or low-frequency component of the image.
    """

#    if not (isinstance(siglim, (list, tuple)) and len(siglim) == 1):
#        siglim = 3.0
#    if not (isinstance(ksize, (list, tuple)) and len(ksize) == 1):
#        ksize = 3.0

    imw = np.copy(im)  # Copy to working data

    # set initial threshold and mask bright spots
    med = np.nanmedian(imw)
#    sig = np.nanstd(imw - med)
    sig = stats.median_abs_deviation(imw - med, scale='normal')
    cond = np.abs(imw - med) > siglim * sig
    cnt = np.sum(cond)
    print(f'Initial count: {cnt} masked pixels')
    cnt_k = 0

#    while cnt > 0:
    while cnt_k <= cnt:
        maskedimw = np.where(cond, np.nan, imw)
        '''
        # Replace NaN values with interpolated values
        stddev = 3.0
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            kernel = Gaussian2DKernel(stddev)
            ims = interpolate_replace_nans(maskedimw, kernel)

            if np.isnan(ims).sum() == 0:
                break
            stddev += 1
            iteration += 1
#            ims = ims
#       if iteration < max_iterations:
#            print(f"Interpolated by a Gaussian 2D kernel with the stddev of {stddev}")
        '''
#        ims = replace_nans(maskedimw, stddev=ksize)
#        ims = uniform_filter(maskedimw, size=ksize, mode='nearest')
        ims = gaussian_filter(maskedimw, sigma=ksize, mode='nearest')
        
        # Recalculate the standard deviation
#        sig = np.nanstd(maskedimw - ims)
        sig = stats.median_abs_deviation(maskedimw - ims, scale='normal')
#        cond = np.abs(maskedimw - np.nanmedian(maskedimw)) > siglim * sig
#        cnt = np.sum(cond)
#        print(cnt)
        cnt_k += 1

    print(f'hpfilter: {cnt_k} pixels masked in total.')
    
## hpfilter 無視
##    ims = np.zeros_like(im)

    # High-frequency component
    imh = im - ims

    return imh, ims





# A simple despiking function that removes spikes from an image.

def despiker3(image, sigma=5):
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
    imw = np.copy(image)
    msk = np.zeros_like(imw, dtype=int)
    
    while True:
        ave = np.nanmean(imw)
        sgm = np.nanstd(imw)

        # replace spikes (outliers) with NaN
        cond = np.abs(imw - ave) > sigma * sgm
        cnt = np.sum(cond)
        if cnt > 0:
            # Mark spikes in the mask and set them to NaN in the data
            imw = np.where(cond, np.nan, imw)
            msk = np.where(cond, 1, msk)
        else:
            break

    # Fill NaNs using Gaussian smoothing
#    imw_filled = replace_nans(imw, sigma)
    imw_filled = gauss_fill(imw, sigma)
#    imw_filled = gaussian_filter(imw, sigma=sigma, mode='nearest')

    # The despiked image is the filled working image
    image_despiked = imw_filled

    # The spikes image is the difference between the original and the despiked image
    image_spikes = image - image_despiked

    return image_despiked, image_spikes



def tanzaku_noise_reduction(image, leftright, basename=None, outdir='./', verbose=False, no_hpf=False, no_despike=False):
    """Perform noise reduction on a tanzaku image."""
    
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
    yran = np.array([0, 300]) + 3
    im_target = image[yran[0]:yran[1], xran[0]:xran[1]]
    
    # High-pass filter processing
    if not no_hpf:
        im_high, im_smth = hpfilter(im_target)
        if verbose and basename:
            im_save = im_high.copy()
#            save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), [im_high, im_smth])
#            save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), im_save.extend(im_smth))
            save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), np.hstack([im_high, im_smth]))
        im_target = im_high
    else:
        im_high = im_target
        im_smth = np.zeros_like(im_target)
    
    # Despike processing
    if not no_despike:
        im_dsp, im_spk = despiker3(im_target)
        if verbose and basename:
            save_fits(os.path.join(outdir, basename + '_dsp' + lr + '.fits'), [im_dsp, im_spk])
            save_fits(os.path.join(outdir, basename + '_dsp' + lr + '.fits'), np.hstack([im_dsp, im_spk]))
        im_target = im_dsp
    else:
        im_dsp = np.zeros_like(im_target)
        im_spk = im_target
    
    nyg, nxg = im_target.shape
    
    # Mirror and copy the data (the way to obtain zero imaginary data when FFT)
    im4 = np.zeros((nyg * 2, nxg * 2))

    im4[0:nyg, 0] = im_target[0:nyg, 0]
    im4[0, 0:nxg] = im_target[0, 0:nxg]
    im4[nyg:2*nyg, 0] = np.flip(im_target[0:nyg, 0], axis=0)
    im4[0, nxg:2*nxg] = np.flip(im_target[0, 0:nxg], axis=0)

    im4[1:nyg+1, 1:nxg+1] = im_target
    im4[nyg:2*nyg, 1:nxg+1] = np.flip(im_target, axis=0)
    im4[1:nyg+1, nxg:2*nxg] = np.flip(im_target, axis=1)
    im4[nyg:2*nyg, nxg:2*nxg] = np.flip(np.flip(im_target, axis=0), axis=1)
#    im_target = im4

    if verbose and basename:
        save_fits(os.path.join(outdir, basename + '_src' + lr + '.fits'), im4)
    
    # Evaluate StdDev before processing
    ave = np.nanmean(im4)
    sgm = np.nanstd(im4)
    print(f'Input image StdDev = {sgm}')
    
    # Perform FFT
    fa = fftshift(fft2(im4))
    #  check if imaginary is small
    print(f'Total imaginary component (prc) is {np.sum(np.abs(np.imag(fa)))}')
    
    if verbose and basename:
        oim = np.stack([np.abs(fa), np.abs(np.real(fa)), np.abs(np.imag(fa))], axis=0)
#        oim = np.zeros((nxg*2, nyg*2, 3))
#        oim[:, :, 0] = np.abs(fa)
#        oim[:, :, 1] = np.abs(np.real(fa))
#        oim[:, :, 2] = np.abs(np.imag(fa))
        save_fits(os.path.join(outdir, basename + '_fft' + lr + '.fits'), oim)
    
    # Masking noise area
    fa0 = fa[0:nyg+1, 0:nxg+1]
    far = np.real(fa0)
    far = field_peri_noise_reduction(far)

    # Mirror updated far to fa
    fa4 = np.zeros((nyg * 2, nxg * 2))
#    fa4 = np.real(fa)
    
    fa4[0:nyg, 0] = far[0:nyg, 0]
    fa4[0, 0:nxg] = far[0, 0:nxg]
    fa4[nyg:2*nyg, 0] = np.flip(far[0:nyg, 0], axis=0)
    fa4[0, nxg:2*nxg] = np.flip(far[0, 0:nxg], axis=0)

    fa4[1:nyg+1, 1:nxg+1]   = far[1:nyg+1, 1:nxg+1]
    fa4[nyg:2*nyg, 1:nxg+1] = np.flip(far[1:nyg+1, 1:nxg+1], axis=0)
    fa4[1:nyg+1, nxg:2*nxg] = np.flip(far[1:nyg+1, 1:nxg+1], axis=1)
    fa4[nyg:2*nyg, nxg:2*nxg] = np.flip(np.flip(far[1:nyg+1, 1:nxg+1], axis=0), axis=1)

    if verbose and basename:
        save_fits(os.path.join(outdir, basename + '_fa4' + lr + '.fits'), np.abs(fa4))
    
    fa = fa4 + 1j * np.zeros_like(fa4)  # Set imaginary to zero
#    fa = np.complex128(fa4)


    # Inverse FFT
    im_reverse = np.real(ifft2(ifftshift(fa)))
#    im_reverse = ifft2(ifftshift(fa))
    print(f'Total imaginary component (rev) is {np.sum(np.abs(np.imag(fa)))}')
#    im_reverse = np.real(im_reverse)
    im_diff = im_reverse - im4

    if verbose and basename:
        save_fits(os.path.join(outdir, basename + '_rev' + lr + '.fits'), im_reverse)
        save_fits(os.path.join(outdir, basename + '_dif' + lr + '.fits'), im_diff)
    
    # Evaluate StdDev after processing
    ave = np.nanmean(im_reverse)
    sgm = np.nanstd(im_reverse)
    print(f'Output image StdDev = {sgm}')
    
    # Recover removed smooth component and spikes
#    im_reverse_0 = im_reverse[0:nyg, 0:nxg]
    im_reverse_0 = im_reverse[1:nyg+1, 1:nxg+1]
    im_reverse_0 += im_spk + im_smth
    
    # Write back to the original image; tanzakudata
    image[yran[0]:yran[1], xran[0]:xran[1]] = im_reverse_0

    return



# Helper function to save a numpy array as a FITS file
def save_fits(filepath, data):
    if isinstance(data, list):
        header=fits.Header()
        hdu = fits.ImageHDU(data)
        hdul = fits.HDUList([fits.PrimaryHDU(header=header), hdu])
#        hdul = fits.HDUList([fits.PrimaryHDU(d) for d in data])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU(data)])
    hdul.writeto(filepath, overwrite=True)



# Main routine

def tanzakurmnoise2d(file, leftonly=False, rightonly=False, outdir='./', verbose=False, nodespike=False, nohpf=False, raw=False):
    """Perform noise reduction and processing on a tanzaku image."""
    
    if 'outdir' not in locals():
        outdir = './'
    
    if not os.path.exists(file):
        print(f"{file} not found.")
        return

#    if not os.path.isfile(file):
#        print(f"{file} not found.")
#        return

    basename = os.path.basename(file).replace('.fits.gz', '').replace('.fits', '')
    
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
        imd = im0 - np.roll(im0, shift=-1, axis=1)
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
        tanzaku_noise_reduction(imd, 'LEFT', basename=basename, outdir=outdir, verbose=verbose, no_hpf=nohpf, no_despike=nodespike)
#    if 'leftonly' not in locals():
    if not leftonly:
        tanzaku_noise_reduction(imd, 'RIGHT', basename=basename, outdir=outdir, verbose=verbose, no_hpf=nohpf, no_despike=nodespike)

    # Integrating to reconstruct the original form
    if raw:
#    if 'raw' in locals() and raw:
        imd[:, 0] = 0
        imi = np.cumsum(imd, axis=1)
        imo = imi.copy()
    else:
        imo = imd.copy()

    # Writing output data
    fout = os.path.join(outdir, basename + '_pnr.fits')
    fits.writeto(fout, imo, hd0, overwrite=True)
    
    # Writing differential data
    ftdf = os.path.join(outdir, basename + '_tdf.fits')
    fits.writeto(ftdf, imd - imd_org, hd0, overwrite=True)
    
    
# Example usage:
# tanzakurmnoise2d('example.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# tanzakurmnoise2d('F0400895463_4NS.fits', leftonly=False, rightonly=False, outdir='./output/', verbose=True, nodespike=True, nohpf=True, raw=True)

# tanzakurmnoise2d('F0436844853_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# 暗い 星の少ない領域
tanzakurmnoise2d('F0977264488_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# 星が1つ
tanzakurmnoise2d('F1413796139_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# 明るい領域
tanzakurmnoise2d('F0246884289_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)




'''



def rmnoise_list(fits_list_path):    
    input_files = read_fits_list(fits_list_path)
    for f in input_files:
        file = f + '.fits'
        tanzakurmnoise2d(file, leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

fits_list_path = sys.argv[1]
rmnoise_list(fits_list_path)






'''
