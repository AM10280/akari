import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from astropy.io import fits
from astropy.stats import sigma_clip, mad_std
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from photutils.detection import DAOStarFinder
from photutils.detection import find_peaks
# from photutils.detection import detect_threshold # photutils<1.0 old
from photutils.segmentation import detect_sources
from scipy.ndimage import binary_dilation
# from photutils import CircularAperture
from photutils.aperture import CircularAperture
# from photutils.segmentation import make_source_mask # photutils<1.0 old
from photutils.segmentation import SourceCatalog, detect_sources
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, Box2DKernel, interpolate_replace_nans
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import griddata
from scipy.ndimage import grey_dilation
from scipy.signal import fftconvolve
# from scipy.ndimage import median_filter
# from scipy import stats
import aplpy

# print(photutils.__version__)





class SourceMaskMaker:
    """
    Alternative class for photutils.make_source_mask
    with following the features 
    Estimation of background statistics (using sigma_clipped_stats) 
    Generation of threshold masks 
    Segmentation of sources (using detect_sources) 
    Optional extension by dilation
    """
    def __init__(self, data):
        self.data = np.asarray(data)

    def make_source_mask(self, nsigma=2.0, npixels=5, dilate_size=None, mask=None):
        """
        Generate source mask (background estimation + threshold + segmentation)
        Parameters
        ----------
        nsigma : float, optional
            Use background median + nsigma x standard deviation as threshold (default: 2.0)

        npixels : int, optional
            Minimum number of connected pixels to be recognized as a segment (default: 5)

        dilate_size : int or None, optional
            Radius for dilating output mask with dilation (default: None)

        mask : 2D bool array, optional
            mask for regions of the input data that should be excluded (default: None)

        Returns
        -------
        mask : 2D bool numpy array
            True is mask image of source region
        """
        mean, median, std = sigma_clipped_stats(self.data, mask=mask)
        threshold = median + nsigma * std
        # threshold = median + (nsigma * std)

        # detect threshold mask
        segm = detect_sources(self.data, threshold, npixels=npixels, mask=mask)

        if segm is None:
            return np.zeros_like(self.data, dtype=bool)

        source_mask = segm.data.astype(bool)

        if dilate_size:
            # create circular structuring element
            from skimage.morphology import disk
            struct = disk(dilate_size)
            source_mask = binary_dilation(source_mask, structure=struct)

        return source_mask




def read_fits_list(fits_list_path):
    """Read list of FITS files from a text file."""
    with open(fits_list_path, 'r') as file:
        return file.read().splitlines()


def read_fits(file):
    """Read data and header from a FITS file."""
    with fits.open(file) as hdul:
        return hdul[0].data, hdul[0].header



def write_fits(file, data, header):
    """Writes data to a FITS file with the provided header."""
    hdu = fits.ImageHDU(data, header=header)
    primary_hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(file, overwrite=True)



def save_fits(filename, data, header, outdir='./output'):
    """Write FITS file to specified directory."""
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, f"{filename}.fits")
    fits.writeto(output_path, data, header, overwrite=True)



def as_pair(name, value, check_odd=True):
    """Convert a scalar or 2-tuple into a tuple of 2 ints."""
    if np.isscalar(value):
        value = (int(value), int(value))
    elif len(value) != 2:
        raise ValueError(f'{name} must be a scalar or a 2-tuple')
    if check_odd and any(v % 2 == 0 for v in value):
        raise ValueError(f'{name} values must be odd')
    return value

def make_source_mask1(data, *, size=None, footprint=None):
    mask = data.astype(bool)

    if footprint is None:
        if size is None:
            return mask
        size = as_pair('size', size, check_odd=False)
        footprint = np.ones(size, dtype=bool)
    else:
        footprint = footprint.astype(bool)

    if np.all(footprint):
        # rectangular, faster
        return grey_dilation(mask, footprint=footprint)
    else:
        return fftconvolve(mask.astype(float), footprint, mode='same') > 0.5

def make_source_mask(data, nsigma=3.0, npixels=5, dilate_size=5):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = median + nsigma * std

    segm = detect_sources(data, threshold, npixels=npixels)
    if segm is None:
        return np.zeros_like(data, dtype=bool)

    mask = segm.data.astype(bool)

    if dilate_size:
        struct = disk(dilate_size)
        mask = binary_dilation(mask, structure=struct)

    return mask


def identify_point_sources_mask(data):
    
    # fits.writeto(outdir, data, header, overwrite=True)
    
    ## Identify Point Sources and Spikes
    # Set a pixel intensity threshold to detect bright sources:
    # Method A: Thresholding
    
    threshold = np.percentile(data, 99.5)  # Example: 99.5th percentile
    mask = data > threshold
    
    # Visualize the mask
    plt.imshow(mask, origin='lower', cmap='gray')
    plt.title('Point Source Mask')
    plt.savefig("point_source.png", dpi=200, bbox_inches="tight")
    plt.close()

    data_masked = np.copy(data)
    data_masked[mask] = np.nan

    return data_masked
    


def identify_point_sources2(data):    

    # Method B: Source Detection with Astropy  # photutils find_peaks
    # Find peaks above the threshold
    threshold = np.percentile(data, 99.5)  # Example: 99.5th percentile
    sources = find_peaks(data, threshold=threshold, box_size=5)
    coordinates = np.array([sources['x_peak'], sources['y_peak']]).T
    
    # Plot detected sources
    plt.imshow(data, origin='lower', cmap='gray')
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='red', marker='x', s=10, label='Point Sources')
    plt.legend()
    plt.savefig("point_source_photutils_find_peaks.png", dpi=200, bbox_inches="tight")
    plt.close()

    return coordinates[:, 0], coordinates[:, 1]


    
def identify_point_sources3(data):
    # Method C: Detect Point Sources with Astropy  # photutils DAOStarFinder
    
    # Estimate background and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    
    # Detect stars using DAOStarFinder
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)  # Adjust FWHM and threshold as needed
    # daofind = DAOStarFinder(fwhm=9.0, threshold=5.0*std)  # Adjust FWHM and threshold as needed
    sources = daofind(data)
    
    if sources is None or len(sources) == 0:
        print("No sources detected.")
        return np.array([]), np.array([])
    # Extract positions
    sources_x = np.array(sources['xcentroid'])
    sources_y = np.array(sources['ycentroid'])

    # Plot detected sources
    positions = np.transpose([sources_x, sources_y])  # shape: (N, 2)
    # positions = list(zip(sources_x, sources_y))
    apertures = CircularAperture(positions, r=4.)  # Radius for visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='gray', origin='lower', vmin=0, vmax=np.percentile(data, 99))
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    plt.colorbar()
    plt.title('Detected Point Sources')
    plt.savefig("point_source_photutils_DAOStarFinder.png", dpi=200, bbox_inches="tight")
    plt.close()

    # coordinates = np.array([sources['xcentroid'], sources['ycentroid']]).T


    return sources_x, sources_y
    
    


def identify_point_sources4(data):
    # Method C: Detect Point Sources with Astropy  # photutils DAOStarFinder
    
    # Estimate background and standard deviation
    # mean = np.mean(data)
    # std = np.std(data)

    # Calculate the background statistics
    mean, median, std = np.mean(data), np.median(data), mad_std(data)
    
    # Detect stars using DAOStarFinder
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)  # Adjust FWHM and threshold as needed
    # daofind = DAOStarFinder(fwhm=9.0, threshold=5.0*std)  # Adjust FWHM and threshold as needed
    # sources = daofind(data)
    sources = daofind(data - median)
    
    # Plot detected sources
    x_sources = sources['xcentroid']
    y_sources = sources['ycentroid']

    # Display detected sources on APLpy
    fig = aplpy.FITSFigure(data)
    fig.show_colorscale(vmin=-1.0, vmax=1.0, stretch='linear', cmap="ds9_cool")
    fig.show_markers(x_sources, y_sources, edgecolor='cyan', facecolor='none', s=50, alpha=0.8)
    fig.add_colorbar()
    fig.show()
    plt.savefig("fits_image.pdf", dpi=300, bbox_inches="tight")


    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=4.)  # Radius for visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='gray', origin='lower', vmin=0, vmax=np.percentile(data, 99))
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    plt.colorbar()
    plt.title('Detected Point Sources')
    plt.savefig("point_source_photutils_DAOStarFinder.png", dpi=200, bbox_inches="tight")
    plt.close()

    sources_x = np.array(sources['xcentroid'])
    sources_y = np.array(sources['ycentroid'])

    # coordinates = np.array([sources['xcentroid'], sources['ycentroid']]).T



    return sources_x, sources_y










    
    ### Overlay Point Sources
    
    
    # detected source positions
    # sources_x = np.array(sources['xcentroid'])
    # sources_y = np.array(sources['ycentroid'])
    
    
    # sources_x = np.array(sources['x_peak'])
    # sources_y = np.array(sources['y_peak'])
    
    '''
    fig.show_markers(sources_x, sources_y, layer='sources', edgecolor='red', facecolor='none', s=50, alpha=0.8)
    fig.show_markers(sources_x, sources_y, edgecolor='red', facecolor='none', marker='o', s=50, alpha=0.7)
    
    fig.show()
    fig.add_grid()
    fig.grid.set_color('white')
    fig.grid.set_alpha(0.5)
    fig.add_colorbar()
    fig.add_scalebar()
    # fig.show_contour(fits_file, levels=[0.1, 0.5, 0.9], colors='red')
    plt.savefig("points_sources.png", dpi=300, bbox_inches="tight")
    
    '''
    




    
def mask_poinnt_sources_phoseg(data):
    # Create a mask around the detected sources
    # masker = SourceMaskMaker(data)
    # mask = masker.make_source_mask(nsigma=3, npixels=5, dilate_size=5)

    mask = make_source_mask(data, nsigma=3, npixels=5, dilate_size=5)  # Adjust nsigma, npixels, and dilate_size as needed
    
    # Apply the mask to the data
    data_masked = np.copy(data)
    data_masked[mask] = np.nan

    # Visualize the masked data
    # fig.show_colorscale(cmap='gray', stretch='linear', vmin=None, vmax=None)
    # fig.show_markers(x_sources, y_sources, edgecolor='red', facecolor='none', s=50, alpha=0.8)
    # fig.show()
    return data_masked
    

def mask_poinnt_sources_phoseg1(data):    
    # Define a Gaussian kernel for smoothing
    kernel = Gaussian2DKernel(x_stddev=2)
    
    # Detect sources
    threshold = 3 * std  # Example threshold: 3-sigma above the background
    segmentation_map = detect_sources(data, threshold, npixels=5)
    
    # Create a mask
    mask = segmentation_map.data > 0
    
    # Apply the mask to your data
    data_masked = np.copy(data)
    data_masked[mask] = np.nan
    return data_masked


    
def mask_point_sources(data, sources_x, sources_y, radius=5.0):
    
    ## Remove or Mask Point Sources
    
    
    # Mask the sources
    # mask and replace point sources with NaNs
    data_masked = np.copy(data)
    # data_masked[mask] = np.nan
    
    
    # Mask sources (example using a circular mask)
    for x, y in zip(sources_x, sources_y):
        rr, cc = np.ogrid[:data.shape[0], :data.shape[1]]
        mask = (rr - y)**2 + (cc - x)**2 <= radius **2  # Circle with radius 5
        data_masked[mask] = np.nan
    
    '''
    # Create masks for the sources
    mask = np.zeros_like(data, dtype=bool)
    for x, y in zip(sources['xcentroid'], sources['ycentroid']):
        rr, cc = np.ogrid[:data.shape[0], :data.shape[1]]
        circle = (rr - y)**2 + (cc - x)**2 <= radius**2  # Circle of radius 4
        mask[circle] = True
    '''
    return data_masked


def mask(data, sources_x, sources_y, radius=5.0):
    
    ## Remove or Mask Point Sources
    
    
    # Mask the sources
    # mask and replace point sources with NaNs
    # data_masked = np.copy(data)
    # data_masked[mask] = np.nan
    
    '''
    # Mask sources (example using a circular mask)
    for x, y in zip(sources_x, sources_y):
        rr, cc = np.ogrid[:data.shape[0], :data.shape[1]]
        mask = (rr - y)**2 + (cc - x)**2 <= radius **2  # Circle with radius 5
        data_masked[mask] = np.nan
    
    '''
    # Create masks for the sources
    mask = np.zeros_like(data, dtype=bool)
    for x, y in zip(sources_x, sources_y):
        rr, cc = np.ogrid[:data.shape[0], :data.shape[1]]
        circle = (rr - y)**2 + (cc - x)**2 <= radius**2  # Circle of radius 4
        mask[circle] = True
    
    return mask








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




def fill_nan(data):
        # mask NaNs
        mask = np.isnan(data)
        coords = np.array(np.nonzero(~mask)).T
        values = data[~mask]
        it = griddata(coords, values, np.array(np.nonzero(mask)).T, method='nearest')
        
        data_filled = data.copy()
        data_filled[mask] = it
    
        return data_filled





def interpolate1(data_masked, sigma=5):    

    # Option A: Mask and Interpolate with scipy
    # Fill NaN values using Gaussian interpolation
    # sigma = 3  # Adjust based on your data
    despiked_data = nan_gaussian_filter(data_masked, sigma=sigma)
    
    # Plot the result
    plt.imshow(despiked_data, origin='lower', cmap='gray', vmin=np.percentile(despiked_data, 5), vmax=np.percentile(despiked_data, 95))
    plt.colorbar()
    plt.title('Despiked Data')
    plt.savefig("despiked_image_scipy.png", dpi=200, bbox_inches="tight")
    plt.close()

    return despiked_data


    
def interpolate2(data_masked, sigma=5):
    
    # Option B: Inpainting with astropy
    
    despiked_image = data_masked.copy()
    
    # NaN removed by linear interpolation
    # mask Nans
    data_masked_wonan = fill_nan(despiked_image)

    # Create a kernel for interpolation
    kernel = Gaussian2DKernel(x_stddev=2)
    
    # Interpolate over masked regions
    # despiked_image = interpolate_replace_nans(data_masked, kernel)
    despiked_image = interpolate_replace_nans(data_masked_wonan, kernel)

    # save_fits('filename', data_masked, header)
    
    # Display cleaned data
    plt.imshow(despiked_image, origin='lower', cmap='gray', vmin=np.percentile(despiked_image, 5), vmax=np.percentile(despiked_image, 95))
    plt.colorbar()
    plt.savefig("despiked_image_astropy.png", dpi=200, bbox_inches="tight")
    plt.close()

    return despiked_image



    
def interpolate3(data_masked, stddev=5):
    
    # Option B: Inpainting with astropy
    
    despiked_image = data.copy()
    despiked_image = replace_nans(data_masked, stddev)
    
    # Display cleaned data
    plt.imshow(despiked_image, origin='lower', cmap='gray', vmin=np.percentile(despiked_image, 5), vmax=np.percentile(despiked_image, 95))
    plt.colorbar()
    plt.savefig("despiked_image_astropy.png", dpi=200, bbox_inches="tight")
    plt.close()

    return despiked_image



def interpolate4(data, mask, sigma=5):
    # Option C: Mask or Interpolate Detected Sources with # Astropy  photutils CircularAnnulus and # scipy median_filter
    
    # Replace masked regions with interpolated values
    despiked_image = data.copy()
    despiked_image[mask] = median_filter(data, size=10)[mask]  # Adjust filter size
    
    # Plot cleaned image
    plt.figure(figsize=(10, 8))
    plt.imshow(despiked_image, cmap='gray', origin='lower', vmin=0, vmax=np.percentile(data, 99))
    plt.colorbar()
    plt.title('Cleaned FITS Image')
    plt.savefig("despiked_image_scipy_cam.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    return despiked_image
    
    
    
    
    

    
def save_despiked_fits(despiked_image):
    
    # Save cleaned data
    # Save the updated FITS file
    hdu = fits.PrimaryHDU(image_cleaned)
    hdul = fits.HDUList([hdu])
    hdul[0].data = data_cleaned
    hdu.writeto('cleaned_image.fits', overwrite=True)
    hdul.writeto('cleaned_image.fits', overwrite=True)
    
    
    ## Re-Plot the Cleaned Data
    fig_cleaned = aplpy.FITSFigure('despiked_image.fits')
    fig_cleaned.show_colorscale(cmap='gray', stretch='linear', vmin=None, vmax=None)
    
    # Optionally, overlay markers again to ensure sources were removed
    fig_cleaned.show_markers(sources_x, sources_y, layer='sources', edgecolor='red', facecolor='none', s=50, alpha=0.8)
    # fig_cleaned.show()
    plt.savefig("despiked_image_saigo.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    
    
    




def return_cena_ellipse(image, despiked_data):

    # Get dimensions of the image
    ny, nx = image.shape
    center_v, center_h = ny // 2, nx // 2
    
    # Semi-major and semi-minor axes for the elliptical mask
    horizontal_radius = 450
    vertical_radius = 450


    # Create a grid of coordinates
    y, x = np.ogrid[:ny, :nx]
    distance = ((x - center_h) / horizontal_radius) ** 2 + ((y - center_v) / vertical_radius) ** 2

    # Create the elliptical mask (1 inside the ellipse, 0 outside)
    # elliptical_mask = distance <= 1
    elliptical_mask = np.where(distance <= 1)

    '''
    ex = float(nx - xlim)
    ey = float(ny - ylim)
    x, y = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    rr = ((x - (nx - 1)) / ex) ** 2 + ((y - (ny - 1)) / ey) ** 2
    ellipse_idx = np.where(rr <= 1)
    '''
    despiked_data[elliptical_mask] = image[elliptical_mask]
    

    return despiked_data



def despike_all_mask1(fits_list_path):
    fits_files = read_fits_list(fits_list_path)
    for f in fits_files:
        file = f + '.fits'
        data, header = read_fits(file)
        # Set up WCS for coordinate transformations
        wcs = WCS(header)
        # nx = hd0['NAXIS1']
        # ny = hd0['NAXIS2']

        # sources_x, sources_y = identify_point_sources3(data)
        mask = identify_point_sources_mask(data)
        data_masked = np.copy(data)
        data_masked[mask] = np.nan
        # data_masked = mask_poinnt_sources(data, sources_x, sources_y, radius=5.0)
        # mask = mask(data, sources_x, sources_y, radius=5.0) #Bool values
        despiked_data = interpolate2(data_masked, sigma=5)
        # despiked_data = interpolate4(data, mask, sigma=5)

        # despiked_data = despiker(data)
        # despiked_image = despiker(image)
        returned_image = return_cena_ellipse(data, despiked_data)
        save_fits(f, returned_image, header, outdir='despiked')
        
    return




def despike_all_mask2(fits_list_path):
    fits_files = read_fits_list(fits_list_path)
    for f in fits_files:
        file = f + '.fits'
        data, header = read_fits(file)
        # Set up WCS for coordinate transformations
        wcs = WCS(header)
        # nx = hd0['NAXIS1']
        # ny = hd0['NAXIS2']

        sources_x, sources_y = identify_point_sources3(data)
        # data_masked = mask_poinnt_sources(data, sources_x, sources_y, radius=5.0)
        mask = mask(data, sources_x, sources_y, radius=5.0) #Bool values
        # despiked_data = interpolate2(data_masked, sigma=5)
        despiked_data = interpolate4(data, mask, sigma=5)

        # despiked_data = despiker(data)
        # despiked_image = despiker(image)
        returned_image = return_cena_ellipse(data, despiked_data)
        save_fits(f, returned_image, header, outdir='despiked')
        
    return




def despike_all(fits_list_path):
    fits_files = read_fits_list(fits_list_path)
    for file in fits_files:
        # file = f + '.fits'
        filename = file.rstrip('.fits')
        data, header = read_fits(file)
        # Set up WCS for coordinate transformations
        # wcs = WCS(header)

        # nx = hd0['NAXIS1']
        # ny = hd0['NAXIS2']

        # sources_x, sources_y = identify_point_sources3(data)
        # data_masked = mask_point_sources(data, sources_x, sources_y, radius=5.0)
        data_masked = identify_point_sources_mask(data)
        # data_masked = mask_poinnt_sources_phoseg(data)
        despiked_data = interpolate2(data_masked, sigma=5)

        # despiked_data = despiker(data)
        # despiked_image = despiker(image)
        returned_image = return_cena_ellipse(data, despiked_data)
        output_filename = filename + '_despiked'
        save_fits(output_filename, returned_image, header, outdir='despiked')
        
    return






if __name__ == "__main__":
    fits_list_path = sys.argv[1]
    despike_all(fits_list_path)
    # despike_all_mask1(fits_list_path)


