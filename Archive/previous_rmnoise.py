import numpy as np
#from scipy.fft import fft2, fftshift
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import uniform_filter, gaussian_filter
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter



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




def gauss_fill2(image, sigma):
    """
    Fill NaN values in the image with a Gaussian-weighted mean from the surroundings.

    Parameters:
    image : 2D numpy array
        The input image with NaN values where spikes were removed.
    sigma : float
        The standard deviation for Gaussian kernel.
    """
    nan_mask = np.isnan(image)
    filled_image = gaussian_filter(np.nan_to_num(image, nan=0), sigma=sigma)
    filled_image[nan_mask] = np.nan
    normalization = gaussian_filter(~nan_mask.astype(float), sigma=sigma)
    image_filled = filled_image / (normalization + 1e-10)
    return image_filled


# fill NaN with Gaussian weighted mean from surroundings.
def gauss_fill(image, sigma):
    if sigma <= 0:
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



# fill NaN with Gaussian weighted mean from surroundings.
def gauss_fill3(image, sigma=5):
    """
    Fill NaN values in an image using a Gaussian-weighted mean from surrounding pixels.

    Parameters:
    image : 2D numpy array
        The input image with NaN values to be filled.
    sigma : float, optional
        The Gaussian width in pixels. Default is 5.
    """
    # Find indices of NaN values in the image
    nan_indices = np.where(np.isnan(image))
    if len(nan_indices[0]) == 0:
        return image  # No NaN values, return the original image

    # Image dimensions
    nx, ny = image.shape
    
    # Generate coordinate grids
    xidx = np.arange(nx)
    yidx = np.arange(ny)
    x1 = np.ones(ny)
    y1 = np.ones(nx)
    
    # Iterate over all NaN indices
    for i in range(len(nan_indices[0])):
        ix = nan_indices[0][i]
        iy = nan_indices[1][i]
        
        # Calculate squared distance from the current NaN pixel
        rr = (xidx[:, np.newaxis] - ix)**2 * y1 + x1[:, np.newaxis] * (yidx - iy)**2
        
        # Calculate Gaussian weights
        gg = np.exp(-rr / (2 * sigma**2))
        
        # Mask the NaN position itself
        gg[ix, iy] = np.nan
        
        # Calculate the Gaussian-weighted mean
        weighted_sum = np.nansum(image * gg)
        weight_sum = np.nansum(gg)
        image[ix, iy] = weighted_sum / weight_sum
    
    return image



# derive stddev of non-signal region.
def field_peri_noise_reduction(image):
    # Copy the first 250 rows of the image
    dat = image[:, 0:250].copy()
    
    # Initialize a mask
    msk = np.zeros(dat.shape, dtype=int)
    
    while True:
        # Calculate mean and standard deviation, ignoring NaNs
        ave = np.nanmean(dat)
        sgm = np.nanstd(dat)
        
        # Identify outliers (noisy pixels)
        idx = np.where(np.abs(dat - ave) > sgm * 3)
        cnt = len(idx[0])
        
        if cnt > 0:
            # Mark noisy pixels in the mask and set them to NaN in the data
            dat[idx] = np.nan
            msk[idx] = 1
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
    sg_normal = np.std(dat[cidx])
    sg_noise = np.std(dat[idx])
    
    # Suppress noise by scaling
    dat[idx] *= sg_normal / sg_noise
    print(f"Suppress factor = {sg_normal / sg_noise}")
    
    # Update the image
    image[:, 0:250] = dat
    # image[55:59, 0:250] = dat[55:59, :]    # only central component

    return image



# iterative high-pass filter
def hpfilter(im, ksize=3, siglim=3.0):
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
    imw = np.copy(im)  # Copy to working data

    # Initial threshold and mask bright spots
    med = np.nanmedian(imw)
    sig = np.nanstd(imw - med)
    idx = np.where(np.abs(imw - med) > siglim * sig)
    cnt_k = len(idx[0])
    
    while len(idx[0]) > 0:
        imw[idx] = np.nan  # Mask outliers

        # Apply a smoothing filter (you can switch between uniform and Gaussian)
        ims = uniform_filter(imw, size=ksize, mode='nearest')
        # ims = gaussian_filter(imw, sigma=ksize, mode='nearest')
        
        # Recalculate the standard deviation
        sig = np.nanstd(imw - ims)
        idx = np.where(np.abs(imw - ims) > siglim * sig)
        cnt_k += len(idx[0])

    print(f'hpfilter: {cnt_k} pixels masked in total.')
    
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
        
        # Identify spikes (outliers)
        idx = np.abs(imw - ave) > sigma * sgm
        
        if not np.any(idx):
            break
        
        imw[idx] = np.nan  # Mask spikes
        msk[idx] = 1  # Mark spikes in the mask

    # Fill NaNs using Gaussian smoothing
    imw_filled = gauss_fill(imw, sigma)

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
        xran = np.array([0, 57]) + 6
    elif leftright == "RIGHT":
        xran = np.array([0, 57]) + 69
    yran = np.array([0, 299]) + 3
    im_target = image[yran[0]:yran[1]+1, xran[0]:xran[1]+1]
    
    # High-pass filter processing
    if not no_hpf:
        im_high, im_smth = hpfilter(im_target)
        if verbose and basename:
            save_fits(os.path.join(outdir, basename + '_hpf' + lr + '.fits'), [im_high, im_smth])
        im_target = im_high
    
    # Despike processing
    if not no_despike:
        im_dsp, im_spk = despiker3(im_target)
        if verbose and basename:
            save_fits(os.path.join(outdir, basename + '_dsp' + lr + '.fits'), [im_dsp, im_spk])
        im_target = im_dsp
    
    nxg, nyg = im_target.shape

    # Mirror and copy the data (the way to obtain zero imaginary data)
    im4 = np.zeros((nxg * 2, nyg * 2))
    im4[0:nxg, 0:nyg] = im_target
    im4[nxg:2*nxg, :nyg] = np.flip(im_target, axis=0)
    im4[0:nxg, nyg:2*nyg] = np.flip(im_target, axis=1)
    im4[nxg:2*nxg, nyg:2*nyg] = np.flip(np.flip(im_target, axis=0), axis=1)
    
#    im_target = im4

    if verbose and basename:
        save_fits(os.path.join(outdir, basename + '_src' + lr + '.fits'), im4)
    
    # Evaluate StdDev before processing
    ave = np.nanmean(im4)
    sgm = np.nanstd(im4)
    print(f'Input image StdDev = {sgm}')
    
    # Perform FFT
    fa = fftshift(fft2(im4))
#    fa = np.fft.fftshift(np.fft.fft2(im4))
    #  check if imaginary is small
    print(f'Total imaginary component (prc) is {np.sum(np.abs(np.imag(fa)))}')
    
    if verbose and basename:
        oim = np.stack([np.abs(fa), np.abs(np.real(fa)), np.abs(np.imag(fa))], axis=-1)
#        oim = np.zeros((nxg*2, nyg*2, 3))
#        oim[:, :, 0] = np.abs(fa)
#        oim[:, :, 1] = np.abs(np.real(fa))
#        oim[:, :, 2] = np.abs(np.imag(fa))
        save_fits(os.path.join(outdir, basename + '_fft' + lr + '.fits'), oim)
    
    # Masking noise area
    far = np.real(fa[0:nxg, 0:nyg])  # fa0 = fa[0:nxg, 0:nyg]
    far = field_peri_noise_reduction(far)


    # Mirror updated far to fa
    fa4 = np.real(fa)
    fa4[0:nxg, 0:nyg] = far
    fa4[nxg:2*nxg, 0:nyg] = np.flip(far[:nxg, :], axis=0)
    fa4[0:nxg, nyg:2*nyg] = np.flip(far[:, :nyg], axis=1)
    fa4[nxg:2*nxg, nyg:2*nyg] = np.flip(np.flip(far[:nxg, :nyg], axis=0), axis=1)
    
    if verbose and basename:
        save_fits(os.path.join(outdir, basename + '_fa4' + lr + '.fits'), np.abs(fa4))
    
    fa = fa4 + 1j * np.zeros_like(fa4)  # Set imaginary to zero
#    fa = np.complex128(fa4)

    # Inverse FFT
#    im_reverse = np.real(ifft2(ifftshift(fa)))
    im_reverse = ifft2(ifftshift(fa))
    print(f'Total imaginary component (rev) is {np.sum(np.abs(np.imag(fa)))}')
    im_reverse = np.real(im_reverse)
    im_diff = im_reverse - im4

    if verbose and basename:
        save_fits(os.path.join(outdir, basename + '_rev' + lr + '.fits'), im_reverse)
        save_fits(os.path.join(outdir, basename + '_dif' + lr + '.fits'), im_diff)
    
    # Evaluate StdDev after processing
    ave = np.nanmean(im_reverse)
    sgm = np.nanstd(im_reverse)
    print(f'Output image StdDev = {sgm}')
    
    # Recover removed smooth component and spikes
    im_reverse_1 = im_reverse[0:nxg, 0:nyg]
    im_reverse_1 += im_spk + im_smth
    
    # Write back to the original image; tanzakudata
    image[yran[0]:yran[1]+1, xran[0]:xran[1]+1] = im_reverse_1

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
    
    if not os.path.exists(file):
        print(f"{file} not found.")
        return
    
    basename = os.path.basename(file).replace('.fits.gz', '').replace('.fits', '')
    
    # Read the input tanzaku data
    im0, hd0 = read_fits(file)
    im0 = im0.astype(np.float64)
    
    nx = hd0['NAXIS1']
    ny = hd0['NAXIS2']
    
    # Differentiation if RAW option is set
    if raw:
        im0[:, 0] = 0
        imd = im0 - np.roll(im0, shift=1, axis=1)
        imd[:, :2] = 0
    else:
        imd = im0.copy()
    
    imd_org = imd.copy()
    
    # Process Tanzaku-Left data
    if not rightonly:
        tanzaku_noise_reduction(imd, 'LEFT', basename=basename, outdir=outdir, verbose=verbose, no_hpf=nohpf, no_despike=nodespike)
    if not leftonly:
        tanzaku_noise_reduction(imd, 'RIGHT', basename=basename, outdir=outdir, verbose=verbose, no_hpf=nohpf, no_despike=nodespike)
    
    # Integration to reconstruct original form if RAW option is set
    if raw:
        imd[:, 0] = 0
        imi = np.cumsum(imd, axis=1)
        imo = imi
    else:
        imo = imd
    
    # Write output data
    fout = os.path.join(outdir, f"{basename}_pnr.fits")
    write_fits(fout, imo, hd0)
    
    # Write differential data
    ftdf = os.path.join(outdir, f"{basename}_tdf.fits")
    write_fits(ftdf, imd - imd_org, hd0)

# Example usage:
# tanzakurmnoise2d('example.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)

# tanzakurmnoise2D('F0400895463_4NS.fits', leftonly=False, rightonly=False, outdir='./output/', verbose=True, nodespike=True, nohpf=True, raw=True)

tanzakurmnoise2d('example.fits', leftonly=False, rightonly=False, outdir='./output', verbose=True, nodespike=False, nohpf=False, raw=False)


