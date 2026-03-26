import sys
import os
from astropy.io import fits

def fix_fits_headers_overwrite(input_dir):
    """Convert all FITS header keywords to uppercase and overwrite the original files."""
    fits_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.fits')]

    for fname in fits_files:
        file_path = os.path.join(input_dir, fname)
        print(f"Processing: {fname}")

        try:
            with fits.open(file_path, ignore_missing_end=True, verify="ignore") as hdul:
                hdr = hdul[0].header
                data = hdul[0].data

                # Create a new header (capitalize the key)
                new_header = fits.Header()
                for key, value in hdr.items():
                    if key == "":
                        continue
                    new_header[key.upper()] = value

                # Overwrite and save
                fits.writeto(file_path, data, new_header, overwrite=True)
                print(f"  → Overwritten: {file_path}")

        except Exception as e:
            print(f"  ⚠ Error processing {fname}: {e}")

    print("\n✔︎ All FITS files processed and overwritten.")


if __name__ == "__main__":
    input_dir = sys.argv[1]
    fix_fits_headers_overwrite(input_dir)

# Usage: 
# Specify the directory containing the FITS files for which you want to capitalize the headers
# python fits_header_capitalization.py DIR
