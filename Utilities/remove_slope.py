import sys
import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Gaussian2DKernel, interpolate_replace_nans

"""
簡易的な黄道光除去の代わり
A simple alternative to zodiacal light removal.
"""

def remove_sloping_brightness_fits(input_fits, output_fits):
    # Open the FITS file and load the image data
    with fits.open(input_fits) as hdul:
        image_data = hdul[0].data  # Extract image as a 2D NumPy array
        header = hdul[0].header  # Keep the header for saving later

    # Ensure the image is a 2D array
    if len(image_data.shape) != 2:
        raise ValueError("FITS image is not 2D.")

    # Get image dimensions
    height, width = image_data.shape

    # Create coordinate matrices
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Flatten the matrices for least squares fitting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = image_data.flatten()

    # Fit a plane (ax + by + c)
    A = np.column_stack((X_flat, Y_flat, np.ones_like(X_flat)))
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)

    # Compute the estimated background plane
    background = coeffs[0] * X + coeffs[1] * Y + coeffs[2]

    # Subtract the estimated background
    corrected_image = image_data - background

    # Save the corrected image as a new FITS file
    hdu = fits.PrimaryHDU(corrected_image, header=header)
    hdu.writeto(output_fits, overwrite=True)

    # Plot the original and corrected images
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].imshow(image_data, cmap='gray', origin='lower')
    axs[0].set_title("Original FITS Image")
    
    axs[1].imshow(corrected_image, cmap='gray', origin='lower')
    axs[1].set_title("Corrected Image")

    plt.show()

'''
    os.makedirs('image', exist_ok=True)
    filename = os.path.basename(input_fits).replace('.fits', '')
    plt.figure(figsize=(8, 6))
    plt.plot(alpha=0.7)
    # plt.plot(median_profile, label='Median Profile', color='r', alpha=0.7)
    # plt.ylim(-16.2, 7)
    # plt.ylim(0, 225)
    # plt.xlabel('Pixel (X-direction)')
    # plt.ylabel('Profile Value')
    # plt.title('Average Profiles')
    # plt.title('Median Profiles of FITS Image (Y-Integrated)')
    # plt.legend()
    # plt.grid(True)
    plot_filename = os.path.join('image', f'{filename}.png')
    plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()
'''

# Example usage
remove_sloping_brightness_fits("input.fits", "output_corrected.fits")
