import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os





# Read the list of FITS files
def read_fits_list(fits_list_path):
    with open(fits_list_path, 'r') as file:
        file_path = file.read().splitlines()
#        fitsname = os.path.basename(file_path)
    return file_path



def stack_fits_list_m(fits_list_path, output_file):
    data_list = []
    input_files = read_fits_list(fits_list_path)
    for file in input_files:
        with fits.open(file) as hdul:
                    data = hdul[0].data
                    if data is not None:
                        data_list.append(data)
                    else:
                        print(f"Warning: {file} does not contain valid data.")

    if len(data_list) == 0:
        print("No valid FITS data found in the specified files.")
        return

    # Stack the data by averaging along the axis of the list
    stacked_data = np.mean(data_list, axis=0)
    
    # Create a new FITS file with the stacked data
    hdu = fits.PrimaryHDU(stacked_data)
    hdu.writeto(output_file, overwrite=True)
        




def stack_fits_list(fits_list_path, output_file):
    

    with open(fits_list_path, 'r') as file:
        input_files = file.read().splitlines()
        # file = f + '.fits'
        fits_concat = [fits.getdata(file) for file in input_files]
        stacked_image = np.sum(fits_concat, axis=0)

    
    # Create a new FITS file with the stacked data
    hdu = fits.PrimaryHDU(stacked_image)
    hdu.writeto(output_file, overwrite=True)



# fits_list_path = 'nocenA_list.txt'
fits_list_path = sys.argv[1]

output_file = "stack/stacked_image.fits"
outdir='./stack'
if not os.path.exists(outdir):
        os.makedirs(outdir)



#stack_fits_list(fits_list_path, output_file)
stack_fits_list_m(fits_list_path, output_file)






