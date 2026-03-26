import os
from astropy.io import fits


def extract_af_tim_from_fits(directory, output_file):
    results = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".fits"):
            file_path = os.path.join(directory, filename)
            try:
                with fits.open(file_path) as hdul:
                    header = hdul[0].header
                    af_tim = header.get("AF_TIM", "N/A")  # Default to "N/A" if AF_TIM is not present
                    results.append(f"{filename}\t{af_tim}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                results.append(f"{filename}\tERROR")

    # Write results to a text file
    with open(output_file, "w") as f:
        f.write("Filename\tAF_TIM\n")
        f.write("\n".join(results))

# Example usage
input_directory = "path/to/fits/directory"  # Replace with the path to your FITS files
output_text_file = "af_tim_results.txt"
extract_af_tim_from_fits(input_directory, output_text_file)
