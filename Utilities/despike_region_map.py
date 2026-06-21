import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from photutils.detection import DAOStarFinder
from photutils.detection import find_peaks
from photutils import CircularAperture
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from scipy.ndimage import gaussian_filter
# from scipy.ndimage import median_filter
# from scipy import stats
import aplpy




# Load the FITS file
fits_file = "example.fits"
hdul = fits.open(fits_file)
data = hdul[0].data
header = hdul[0].header
hdul.close()

# Set up WCS for coordinate transformations
wcs = WCS(header)



'''
## Visualize
fits_file = "example.fits"
fig = aplpy.FITSFigure(fits_file)
fig.show_colorscale(cmap='gray', stretch='linear', vmin=None, vmax=None)
fig.show_colorscale()

# Add grid or labels if needed
fig.add_grid()
fig.grid.set_color('white')
fig.grid.set_alpha(0.5)

# plt.show()
plt.savefig("hajime.png", dpi=300, bbox_inches="tight")
plt.close()



# Plot the image
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='gray', origin='lower', vmin=0, vmax=np.percentile(data, 99))
plt.colorbar()
plt.title('Original FITS Image')
plt.savefig("hajime_1.png", dpi=300, bbox_inches="tight")
plt.close()
'''




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



'''
# Method C: Detect Point Sources with Astropy  # photutils DAOStarFinder

# Estimate background and standard deviation
mean = np.mean(data)
std = np.std(data)

# Detect stars using DAOStarFinder
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)  # Adjust FWHM and threshold as needed
sources = daofind(data)


# Plot detected sources
positions = (sources['xcentroid'], sources['ycentroid'])
apertures = CircularAperture(positions, r=4.)  # Radius for visualization
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='gray', origin='lower', vmin=0, vmax=np.percentile(data, 99))
apertures.plot(color='red', lw=1.5, alpha=0.5)
plt.colorbar()
plt.title('Detected Point Sources')
plt.savefig("point_source_photutils_DAOStarFinder.png", dpi=200, bbox_inches="tight")
plt.close()
'''


'''
### Overlay Point Sources


# detected source positions
sources_x = np.array(sources['x_peak'])
sources_y = np.array(sources['y_peak'])

fig.show_markers(sources_x, sources_y, layer='sources', edgecolor='red', facecolor='none', s=50, alpha=0.8)
fig.show_markers(sources_x, sources_y, edgecolor='red', facecolor='none', marker='o', s=50, alpha=0.7)

# fig.add_grid()
# fig.grid.set_color('white')
# fig.grid.set_alpha(0.5)
fig.add_colorbar()
fig.add_scalebar(11)
# fig.show_contour(fits_file, levels=[0.1, 0.5, 0.9], colors='red')
plt.savefig("points_sources.png", dpi=300, bbox_inches="tight")
plt.close()
'''







## Remove or Mask Point Sources


# Mask the sources
# mask and replace point sources with NaNs
data_masked = np.copy(data)
data_masked[mask] = np.nan


'''
# Mask sources (example using a circular mask)
for x, y in zip(sources_x, sources_y):
    rr, cc = np.ogrid[:data.shape[0], :data.shape[1]]
    mask = (rr - y)**2 + (cc - x)**2 <= 5**2  # Circle with radius 5
    data_masked2[mask] = np.nan
'''



# Option A: Mask and Interpolate with scipy
# Fill NaN values using Gaussian interpolation
sigma = 3  # Adjust based on your data
data_cleaned = gaussian_filter(data_masked, sigma=sigma, mode='nearest')

# Plot the result
plt.imshow(data_cleaned, origin='lower', cmap='gray', vmin=np.percentile(data_cleaned, 5), vmax=np.percentile(data_cleaned, 95))
plt.colorbar()
plt.title('Cleaned Data')
plt.savefig("despiked_image_scipy.png", dpi=200, bbox_inches="tight")
plt.close()


# Option B: Inpainting with astropy
# Create a kernel for interpolation
kernel = Gaussian2DKernel(x_stddev=2)

# Interpolate over masked regions
data_cleaned = interpolate_replace_nans(data_masked, kernel)

# Display cleaned data
plt.imshow(data_cleaned, origin='lower', cmap='gray', vmin=np.percentile(data_cleaned, 5), vmax=np.percentile(data_cleaned, 95))
plt.colorbar()
plt.savefig("despiked_image_astropy.png", dpi=200, bbox_inches="tight")
plt.close()



'''

# Option C: Mask or Interpolate Detected Sources with # Astropy  photutils CircularAnnulus and # scipy median_filter

# Create masks for the sources
mask = np.zeros_like(data, dtype=bool)
for x, y in zip(sources['xcentroid'], sources['ycentroid']):
    rr, cc = np.ogrid[:data.shape[0], :data.shape[1]]
    circle = (rr - y)**2 + (cc - x)**2 <= (4.0)**2  # Circle of radius 4
    mask[circle] = True

# Replace masked regions with interpolated values
image_cleaned = data.copy()
image_cleaned[mask] = median_filter(data, size=10)[mask]  # Adjust filter size

# Plot cleaned image
plt.figure(figsize=(10, 8))
plt.imshow(image_cleaned, cmap='gray', origin='lower', vmin=0, vmax=np.percentile(data, 99))
plt.colorbar()
plt.title('Cleaned FITS Image')
plt.savefig("despiked_image_scipy_cam.png", dpi=200, bbox_inches="tight")
plt.close()
'''












# Save cleaned data
# Save the updated FITS file
hdu = fits.PrimaryHDU(data_cleaned)
# hdul = fits.HDUList([hdu])
# hdul[0].data = data_cleaned
hdu.writeto('cleaned_image.fits', overwrite=True)
# hdul.writeto('cleaned_image.fits', overwrite=True)


## Re-Plot the Cleaned Data
fig_cleaned = aplpy.FITSFigure('cleaned_image.fits')
fig_cleaned.show_colorscale(cmap='gray', stretch='linear', vmin=None, vmax=None)

# Optionally, overlay markers again to ensure sources were removed
# fig_cleaned.show_markers(sources_x, sources_y, layer='sources', edgecolor='red', facecolor='none', s=50, alpha=0.8)
# fig_cleaned.show()
plt.savefig("despiked_image_saigo.png", dpi=200, bbox_inches="tight")
plt.close()














# Load the FITS file
# fits_file = "example.fits"  # Replace with your FITS file path
fits_file = "cleaned_image.fits"
hdul = fits.open(fits_file)
data = hdul[0].data
header = hdul[0].header
hdul.close()

# Set up WCS for coordinate transformations
wcs = WCS(header)

# Create a figure with WCS projection
# fig = plt.figure(figsize=(10, 8))  # Specify figure size
#fig, ax = plt.subplots(figsize=(16, 16))
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection=wcs)


plt.rcParams['xtick.direction'] = 'out' # in, out
plt.rcParams['ytick.direction'] = 'out'





# Plot the data with specified colormap and scaling
img = ax.imshow(
    data, 
    origin="lower", 
    cmap="viridis",
    # cmap="ds9_bb",
    # cmap="ds9_cool",
    # cmap="ds9_sls",
    # norm=plt.Normalize(vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))  # Scale
    norm=plt.Normalize(vmin=-1.0, vmax=1.0)  # Scale
)

# Add a color bar
cbar = plt.colorbar(img, ax=ax, orientation="vertical", pad=0.05)
cbar.set_label("Pixel Intensity")

# ax.grid(color="white", ls="dotted", lw=0.5)
ax.set_xlabel("RA")
ax.set_ylabel("Dec")

plt.savefig("cleaned_image2.png", dpi=300, bbox_inches="tight")
# plt.close()







# Create a figure with APLpy
fits_file = 'example.fits'
fig = aplpy.FITSFigure(fits_file)
fig.show_colorscale()
fig.show_grayscale()

# Add a title and labels
fig.set_title("Infrared Image")
fig.add_colorbar()
plt.savefig("cleaned_image3.png", dpi=300, bbox_inches="tight")


