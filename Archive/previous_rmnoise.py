import sys
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import uniform_filter, generic_filter, gaussian_filter
from scipy import stats
from astropy.io import fits
# from astropy.stats import sigma_clip
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, Box2DKernel, interpolate_replace_nans
# from astropy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import time
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
    primary_hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(file, overwrite=True)







# Read the list of FITS files
def read_fits_list(fits_list_path):
    with open(fits_list_path, 'r') as file:
        file_paths = file.read().splitlines()
#        fitsname = os.path.basename(file_paths)
    return file_paths




# field_sigma - Remove outliers more than 3 sigma from the mean and replace with NaN
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
        max_iterations = 10
        iteration = 0

        # use this for the scipy convolution
        # img_zerod = image.copy()
        # img_zerod[np.isnan(image)] = 0

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





# gauss_fill - Fill NaN values with Gaussian-weighted mean of surroundings
def gauss_fill(image, sigma=5.):
    """
    Fill NaN values in the image with Gaussian-weighted mean of their surroundings.

    Parameters:
    image (2D array): Input image with NaN values
    sigma (float): Standard deviation of Gaussian kernel
    """
    # Find the indices of NaN values in the image
    mask = np.where(np.isnan(image))
    
    # Get image size
    rows, cols = image.shape

    # Create meshgrid for the distances of x and y index arrays
    cj, ci  = np.meshgrid(range(cols), range(rows))
    
    # Iterate over each NaN value
    for i, j in zip(mask[0], mask[1]):
        gg = np.exp(-(((ci-i)/sigma)**2+((cj-j)/sigma)**2)/2)
        # Compute the squared distance from the current NaN pixel to all others
        # rr = (ci - i)**2 + (cj - j)**2
        # Compute Gaussian weights
        # gg = np.exp(-rr / (2 * sigma**2))
        
        # Set the weight of the NaN pixel itself to NaN
        gg[i, j] = np.nan
        
        # Compute the weighted mean of surrounding pixels
        weighted_sum = np.nansum(image * gg)
        weight_total = np.nansum(gg)
        
        # Replace NaN with the weighted mean
        image[i, j] = weighted_sum / weight_total if weight_total != 0 else np.nan

    return image




def field_peri_noise_reduction(image, mode=2, xlim=35, ylim=270):
    """
    mode 
        0: y >= ylim
        1: rhombus, x >= xlim & y >= ylim
        2: ellipse, x >= xlim & y >= ylim
    """
    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input image must be a 2D numpy array.")

    ny, nx = image.shape  # Dimensions of the image

    # Initialize the working data and mask for noise detection in y < ylim
    # Copy the first 270 rows of the image
    dat = image[:ylim, :].copy()
    # Initialize a mask
    msk = np.zeros(dat.shape, dtype=int)
    # msk = np.zeros_like(dat, dtype=int)

    while True:
        # Calculate mean and standard deviation excluding NaN values
        ave = np.nanmean(dat)
        sgm = np.nanstd(dat)
        cond = np.abs(dat - ave) > sgm * 3    # condition  ## find mask regions
        # cnt = np.sum(cond) # the number of True
        cnt = np.count_nonzero(cond)
        if cnt == 0:
            break

        # Replace outliers with NaN and update the mask
        # Mark noisy pixels in the mask and set them to NaN in the data
        dat[cond] = np.nan
        msk[cond] = 1

    # number of NaNs
    nan_n = np.count_nonzero(np.isnan(dat))
    
    # Separate noisy and normal pixels
    # Identify indices of noisy and normal regions
    noise_mask = np.where(msk == 1)
    normal_mask = np.where(msk == 0)
    # normal_mask = msk == 0
    # noise_mask = msk == 1

    # If there are no noisy or normal pixels, exit
#    if len(noise_mask) == 0 or len(normal_mask) == 0:
#        print("No noise detected or insufficient normal pixels.")
#       return
        
    if nan_n == 0:
        return

    print(f'FieldPeriNoiseReduction: {nan_n} pixels are processed.')

    # copy entire data to working data and identify pixels above the limit
    # Recalculate the data from the original image ←important
    # dat = image[:ylim, :].copy()
    dat = image.copy()
    # print(dat.shape)

    # Calculate suppression factor
    # sg_normal = np.std(image[:ylim, :][normal_mask])
    # sg_noise = np.std(image[:ylim, :][noise_mask])
    sg_normal = np.std(dat[normal_mask])
    sg_noise = np.std(dat[noise_mask])
    if sg_noise != 0:  # Avoid division by zero
        suppression_factor = sg_normal / sg_noise
        print(f"Suppress factor = {suppression_factor}")

        # full image cond?

        # Suppress noise by scaling
        # Apply suppression factor to noisy pixels
        dat[noise_mask] *= suppression_factor

    # overwrite protected area by the original data
    # Protect specific regions based on mode
    if mode == 0:
        print("Mode = 0: Protecting y >= ylim.")
        dat[ylim:, :] = image[ylim:, :]
    elif mode == 1:
        print("Mode = 1: Protecting rectangular region.")
        dat[ylim:, xlim:] = image[ylim:, xlim:]
    elif mode == 2:
        print("Mode = 2: Protecting elliptical region.")
        ex = float(nx - xlim) # semi-major axis
        ey = float(ny - ylim) # semi-minor axis
        # Create a grid of coordinates
        # y, x = np.ogrid[:ny, :nx]
        x, y = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        # distance = ((x - (nx - 1)) / ex) ** 2 + ((y - (ny - 1)) / ey) ** 2
        distance = ((x - nx) / ex) ** 2 + ((y - ny) / ey) ** 2
        # Create the elliptical mask (1 inside the ellipse, 0 outside)
        # elliptical_mask = distance <= 1
        ellipse_idx = np.where(distance <= 1)
        dat[ellipse_idx] = image[ellipse_idx]
    else:
        raise ValueError("Mode must be 0, 1, or 2.")


    # Update the image with processed data
    # image[:ylim, :] = dat
    image = dat

    return image, msk



# High-pass filter (Gaussian-based)
def hpfilter(image, ksize=2, siglim=3.0):
    # kernel size = 3→2
    # Create a working copy of the input image
    imw = image.copy()
    # imw = np.copy(image)
    
    # Initial threshold and mask bright spots
    med = np.median(imw)
    sig = np.nanstd(imw - med)
    mask = np.abs(imw - med) > siglim * sig
    # count = np.sum(mask)
    count = np.count_nonzero(mask)
    cnt_k = count
    # print(f"High-pass filter: {count} pixels masked.")

    while count > 0:
        imw[mask] = np.nan
        # imw[mask] = 0   ###
#        ims = generic_filter(imw, np.nanmean, size=int(ksize), mode='constant', cval=np.nan)
        # ims = gaussian_filter(np.nan_to_num(imw), sigma=ksize, mode='nearest')
        # ims = nan_box2_filter(imw, width=ksize)
        ims = nan_gaussian_filter(imw, sigma=ksize)

        sig = np.nanstd(imw - ims)
        # mask = np.abs(imw - ims) > siglim * sig
        mask = (imw != 0) & (np.abs(imw - ims) > siglim * sig)
        count = np.count_nonzero(mask)
        # print(f"High-pass filter: {count} pixels masked.")
        cnt_k += count

    print(f"\nHigh-pass filter: {cnt_k} pixels masked in total.")
    # print(f'hpfilter: {cnt_k} pixels masked in total.')
    
    ## hpfilter 無視
#    ims = np.zeros_like(im)

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
#    imw_filled = replace_nans(imw, sigma)

    # The despiked image is the filled working image
    image_despiked = imw_filled

    # The spikes image is the difference between the original and the despiked image
    image_spikes = image - image_despiked

    return image_despiked, image_spikes



def tanzaku_noise_reduction(image, leftright, basename=None, outdir='./', verbose=False, no_hpf=False, no_despike=False):
    """Perform noise reduction on a tanzaku image."""
    start_time = time.process_time()

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
    
    # High-pass filtering
    if not no_hpf:
        im_high, im_smth = hpfilter(im_target)
        if verbose:
            save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), [im_high, im_smth])
            # save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), np.hstack([im_high, im_smth]))
        im_target = im_high
    else:
        im_high = im_target
        im_smth = np.zeros_like(im_target)
    
    # Despiking
    if not no_despike:
        im_dsp, im_spk = despiker(im_target)
        if verbose:
            save_fits(os.path.join(outdir, basename + '_dsp' + lr + '.fits'), [im_dsp, im_spk])
            # save_fits(os.path.join(outdir, basename + '_dsp' + lr + '.fits'), np.hstack([im_dsp, im_spk]))
        im_target = im_dsp
    else: 
        im_dsp = im_target
        im_spk = np.zeros_like(im_target)
    
    # Get the image size
    h, w = im_target.shape

    '''
    # Mirror and copy the data
    # Create a new image that is 4 times the size by folding the original image with 1-pixel overlap (to obtain zero imaginary data when FFT)
    # Horizontally mirror
    h_mirror = np.hstack((im_target, np.fliplr(im_target[:,:-2])))

    # and Vertically mirror
    folded_image = np.vstack((h_mirror, np.flipud(h_mirror[:-2,:])))

    # im_target = folded_image
    '''

    
    # Mirror and copy the data (the way to obtain zero imaginary data when FFT)
    im4 = np.zeros((h * 2, w * 2))

    # im4[0:h, 0] = im_target[0:h, 0]
    # im4[0, 0:w] = im_target[0, 0:w]
    # im4[h:2*h, 0] = np.flip(im_target[0:h, 0], axis=0)
    # im4[0, w:2*w] = np.flip(im_target[0, 0:w], axis=0)

    im4[1:h+1, 1:w+1] = im_target
    im4[h:2*h, 1:w+1] = np.flip(im_target, axis=0)
    im4[1:h+1, w:2*w] = np.flip(im_target, axis=1)
    im4[h:2*h, w:2*w] = np.flip(np.flip(im_target, axis=0), axis=1)
    
    folded_image = im4
    


    if verbose:
        save_fits(os.path.join(outdir, basename + '_src' + lr + '.fits'), folded_image)
    

    # Evaluate StdDev before processing
#    ave, sgm = field_sigma(folded_image)
#    print(f'Input image StdDev = {sgm}')

    # Evaluate StdDev before processing
    ave = np.nanmean(folded_image)
    sgm = np.nanstd(folded_image)
    print(f'Input image StdDev = {sgm}')
    
    # Perform 2D FFT
    fft_image = fftshift(fft2(folded_image))
    #  check if imaginary is small
    print(f'Total imaginary component (prc) is {np.sum(np.abs(np.imag(fft_image)))}')
    
    
    # abs, real, imaginary components of Fourier transformed image (3 dimension)
    if verbose:
        oim = np.stack([np.abs(fft_image), np.abs(np.real(fft_image)), np.abs(np.imag(fft_image))], axis=0)
#        oim = np.zeros((h*2, w*2, 3))
        # oim = np.zeros((fft_image.shape[0], fft_image.shape[1], 3), dtype=np.float64)
#        oim[:, :, 0] = np.abs(fft_image)
#        oim[:, :, 1] = np.abs(np.real(fft_image))
        # oim[:, :, 2] = np.real(fft_image)
#        oim[:, :, 2] = np.abs(np.imag(fft_image))
        # oim[:, :, 2] = np.imag(fft_image)
        save_fits(os.path.join(outdir, basename + '_fft' + lr + '.fits'), oim)
    

    # Extract the left top region in Fourier space (original image's size)
    # fft_image_o = fft_image[:h, :w]
    fft_image_o = fft_image[0:h+1, 0:w+1]
    # save_fits(os.path.join(outdir, basename + '_fft_o' + lr + '.fits'), np.abs(fft_image_o))
    # Apply a noise reduction filter     
    real_fft = np.real(fft_image_o)
    if verbose:
        save_fits(os.path.join(outdir, basename + '_fft_oreal' + lr + '.fits'), np.abs(real_fft))

    fft_masked, mask_area = field_peri_noise_reduction(real_fft)


    if verbose:
        save_fits(os.path.join(outdir, basename + '_fftm' + lr + '.fits'), np.abs(fft_masked))
        save_fits(os.path.join(outdir, basename + '_msk' + lr + '.fits'), np.abs(mask_area))
    
    '''
    # Refold the masked Fourier image (with 1-pixel overlap)
    # Horizontally mirror (with 1-pixel overlap)
    h_mirror_masked = np.hstack((fft_masked, np.fliplr(fft_masked[:,:-2])))
    # Vertically mirror
    folded_fft_masked = np.vstack((h_mirror_masked, np.flipud(h_mirror_masked[:-2,:])))
    '''

    
    # Mirror updated far to fa
    fa4 = np.zeros((h * 2, w * 2))
#    fa4 = np.real(folded_image)
    # fa4 = np.zeros(folded_image)
    
    # fa4[0:h, 0] = fft_masked[0:h, 0]
    # fa4[0, 0:w] = fft_masked[0, 0:w]
    # fa4[h:2*h, 0] = np.flip(fft_masked[0:h, 0], axis=0)
    # fa4[0, w:2*w] = np.flip(fft_masked[0, 0:w], axis=0)

    fa4[1:h+1, 1:w+1]   = fft_masked[1:h+1, 1:w+1]
    fa4[h:2*h, 1:w+1] = np.flip(fft_masked[1:h+1, 1:w+1], axis=0)
    fa4[1:h+1, w:2*w] = np.flip(fft_masked[1:h+1, 1:w+1], axis=1)
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
    print(f'Total imaginary component (inverse FFT) is {np.sum(np.abs(np.imag(reconstructed_image)))}')
    reconstructed_image = np.real(reconstructed_image)
    im_difference = reconstructed_image - folded_image

    if verbose:
        save_fits(os.path.join(outdir, basename + '_rev' + lr + '.fits'), reconstructed_image)
        save_fits(os.path.join(outdir, basename + '_dif' + lr + '.fits'), im_difference)
    
    
    # Evaluate StdDev after processing
#    ave, sgm = field_sigma(reconstructed_image)
#    print(f'Output image StdDev = {sgm}')

    # Evaluate StdDev after processing
    ave = np.nanmean(reconstructed_image)
    sgm = np.nanstd(reconstructed_image)
    print(f'Output image StdDev = {sgm}')
    
    # Cut to original size
    reconstructed_image_o = reconstructed_image[1:h+1, 1:w+1]
    # reconstructed_image_o = reconstructed_image[:h, :w]

    # Recover removed smooth component and spikes
    reconstructed_image_o += im_spk + im_smth
    
    # Write back to the original image; tanzakudata
    image[yran[0]:yran[1], xran[0]:xran[1]] = reconstructed_image_o


    # 時間計測
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"経過時間：{elapsed_time}")

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
    start_time = time.process_time()

    if 'outdir' not in locals():
        outdir = './'

    basename = os.path.basename(file).replace('.fits.gz', '').replace('.fits', '')

    if basename is None:
        verbose = False
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if not os.path.exists(file):
        print(f"{file} not found.")
        return

#    if not os.path.isfile(file):
#        print(f"{file} not found.")
#        return

    
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


    return
    
    
# Example usage:
# tanzakurmnoise2d('example.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# tanzakurmnoise2d('F0400895463_4NS.fits', leftonly=False, rightonly=False, outdir='./output/', verbose=True, nodespike=True, nohpf=True, raw=True)

# tanzakurmnoise2d('F0436844853_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# Cen Aを含む
# tanzakurmnoise2d('F0439699725_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)
# tanzakurmnoise2d('F0439699725_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=False, nodespike=False, nohpf=False, raw=False)

# 暗い 星の少ない領域
# tanzakurmnoise2d('F0977264488_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# 星が1つ
# tanzakurmnoise2d('F1413796139_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# 明るい領域
# tanzakurmnoise2d('F0246884289_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)


'''
'''

# 明るい
# tanzakurmnoise2d('F0541153248_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)
# tanzakurmnoise2d('F1035614886_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# 星少し
# tanzakurmnoise2d('F0473757261_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)
# tanzakurmnoise2d('F0246884289_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# 暗い
# tanzakurmnoise2d('F0668634023_4NS.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)




def rmnoise_list(fits_list_path):    
    input_files = read_fits_list(fits_list_path)
    for f in input_files:
        # file = f + '.fits'
        tanzakurmnoise2d(file, leftonly=False, rightonly=False, outdir='./output', verbose=False, nodespike=False, nohpf=False, raw=False)
        # tanzakurmnoise2d(file, leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)




if __name__ == "__main__":
    fits_list_path = sys.argv[1]
    rmnoise_list(fits_list_path)





