import sys
import argparse
import numpy as np
from astropy.io import fits
# import cv2
import matplotlib.pyplot as plt
import os


# os.makedirs('test')

def read_fits(file):
    """Reads a FITS file and returns the image data and header."""
    with fits.open(file) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
    return data, header



def write_fits(data, header, filename, outdir='./ratio_image'):
    os.makedirs(outdir, exist_ok=True)
    # hdu = fits.PrimaryHDU(data, header=header)
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(os.path.join(outdir, f"{filename}.fits"), overwrite=True)
    
    return



def divide_fits(image1, image2, filename='ratio'):

    im1 ,header = read_fits(image1)
    im2 ,header = read_fits(image2)


    # --- Handle potential issues ---
    # Avoid division by zero or NaN
    im2[im2 == 0] = np.nan

    # --- Compute ratio ---
    ratio = im1 / im2

    # --- Optional: mask extreme or invalid values ---
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    # --- Save ratio image as FITS ---
    write_fits(ratio, header, filename)

    return



parser = argparse.ArgumentParser()
parser.add_argument('im1', help='')
parser.add_argument('im2', help='')
parser.add_argument('ratio_image')
args = parser.parse_args()

divide_fits(args.im1, args.im2, args.ratio_image)

# Usage:
# python ~/fits_ratio.py 'numerator.fits' 'denominator.fits' 'n_over_d'



