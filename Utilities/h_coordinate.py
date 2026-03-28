import sys
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import random


"""
This program plots filenames on a coordinate plane to indicate their positions.
"""

def read_fits_list(fits_list_path):
    """Read list of FITS files from a text file."""
    with open(fits_list_path, 'r') as file:
        return file.read().splitlines()


def read_fits(file):
    """Read data and header from a FITS file."""
    with fits.open(file) as hdul:
        return hdul[0].data, hdul[0].header


def load_fits_images(fits_files):
    """Load all FITS images into a list of data arrays."""
    return [read_fits(file)[0] for file in fits_files]


def save_fits(file, filename, outdir='./profile_image'):
    os.makedirs(outdir, exist_ok=True)
    hdu = fits.PrimaryHDU(file)
    hdu.writeto(os.path.join(outdir, f"{filename}.fits"), overwrite=True)
    # hdu.writeto(f"./profile_image/{filename}.fits", overwrite=True)



def write_fits(filename, data, header, outdir):
    """Write FITS file to specified directory."""
    output_path = os.path.join(outdir, f"{filename}.fits")
    fits.writeto(output_path, data, header, overwrite=True)




def get_ra_dec_from_fits(fits_file):
    """
    Extract RA and Dec from CRVAL1 and CRVAL2 in FITS header.
    
    Parameters:
        fits_file (str): Path to the FITS file.
    
    Returns:
        tuple: (SkyCoord object, file name)
    """
    with fits.open(fits_file) as hdul:
        header = hdul[0].header

        # Get RA and Dec from CRVAL1 and CRVAL2
        ra = header.get('CRVAL1')  # In degrees
        dec = header.get('CRVAL2')  # In degrees

        if ra is None or dec is None:
            raise ValueError("CRVAL1 or CRVAL2 not found in header.")

        # Create a SkyCoord object (RA and Dec in degrees)
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        # fname = os.path.basename(fits_file).replace('_4NS_mixw.fits', '')
        fname = os.path.basename(fits_file).replace('_4NS_mixw.fits', '')[3:]
        return coord, fname





def save_profile(coord_list, filename, outdir='./filename_csv'):
    os.makedirs(outdir, exist_ok=True)
    # filename = os.path.basename(fits_list_path).replace('.txt', '')

    profile_filename = os.path.join(outdir, f'{filename}.csv')
    
    np.savetxt(profile_filename, coord_list)
    # np.savetxt('./profile_ave_csv/1st_season.csv', profile_x, delimiter=',')

    '''
    with open(profile_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(coord_list)
    '''



# def plot_coordinates(coord_list, filename, outdir='./filename', label_fraction=0.05, title="FITS File Locations"):
def plot_coordinates(coord_list, filename, outdir='./filename', title="FITS File Locations"):
    os.makedirs(outdir, exist_ok=True)
    # filename = os.path.basename(fits_list_path).replace('.txt', '')

    """
    Plot RA/Dec coordinates from multiple FITS files.
    
    Parameters:
        coord_list (list): List of tuples (SkyCoord, filename)
    """
    # plt.figure(figsize=(8, 8))
    # transparent canvas
    plt.figure(figsize=(8, 8), facecolor='none')

    '''
    colors = ['orangered', 'slateblue', 'royalblue', 'plum', '#ef397e', 'yellow', 'salmon', 'paleturquoise', 'darkkhaki', 'darkslategray', 'mediumseagreen', 'olivedrab']
    for i in range(12):
        ax = plt.scatter(df_s.iloc[i, 15], df_s.iloc[i, 14], color=colors[i], s=75, alpha=0.6, edgecolor='none', label=df_s.index[i])
    plt.plot([0,1006], [0,1006], color='gray', lw=0.6, alpha=0.3)  # x=y line
    '''



    ra_vals = [coord.ra.deg for coord, _ in coord_list]
    dec_vals = [coord.dec.deg for coord, _ in coord_list]


    # gradient color file name
    # Convert fname string to int
    fname_nums = [int(label) for _, label in coord_list]


    # Normalize these numbers to 0-1
    norm = mcolors.Normalize(vmin=min(fname_nums), vmax=max(fname_nums))
    cmap = cm.get_cmap('magma')
    # cmap = cm.get_cmap('viridis')  # You can change 'viridis' to another colormap

    # Get color for each point
    colors = [cmap(norm(num)) for num in fname_nums]

    # Scatter plot with gradient colors
    plt.scatter(ra_vals, dec_vals, c=colors, marker='o', s=10, alpha=0.8)


    # plt.scatter(ra_vals, dec_vals, c='C0', marker='o', s=10, alpha=0.8, label="FITS Files")
    # plt.scatter(ra_vals, dec_vals, c='viridis', marker='o', s=10, label="FITS Files")

    # Label only a random subset to reduce overlap
    # Label a fraction of points
    # label_fraction=0.05
    # num_to_label = max(1, int(len(coord_list) * label_fraction))
    # labeled_points = random.sample(coord_list, num_to_label)

    # for coord, label in labeled_points:
            # plt.text(coord.ra.deg, coord.dec.deg, label, fontsize=8, ha='left', va='bottom')

    for coord, fname in coord_list:
        # plt.plot(coord.ra.deg, coord.dec.deg, 'o', s=10 label=fname, alpha=0.7)
        plt.text(coord.ra.deg, coord.dec.deg, fname, alpha=0.8, fontsize=8, ha='right', va='top')
        # plt.text(coord.ra.deg, coord.dec.deg, fname, fontsize=7, ha='left', va='bottom')

    # plt.xlabel("Right Ascension (deg)")
    # plt.ylabel("Declination (deg)")
    plt.xlabel("R.A. (deg)")
    plt.ylabel("Dec. (deg)")
    plt.title(title)
    plt.grid(True)
    # plt.legend()

    # Invert the x-axis for descending R.A. # R.A. axis in descending order
    plt.gca().invert_xaxis()
    plt.tight_layout()

    # gradient color
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for older matplotlib versions
    # color bar
    # plt.colorbar(sm, label='fname (8-digit number)')
    
    plot_filename = os.path.join(outdir, f'{filename}.png')
    # 白背景
    # plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    # 透過背景
    plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1, transparent=True)
    #plt.show()
    plt.close()






# scatter map
# 離散的カラーリング
# ファイル名の数値の差が10000以上の場合、異なる色, 10000未満の場合、同じ色
# 色分けグループ
def assign_color_groups(labels):
    """
    Given a list of 8-digit number strings, assign group IDs based on jumps of ≥10000.
    """
    numbers = [int(label) for label in labels]
    group_ids = [0]  # first file is in group 0

    current_group = 0
    for i in range(1, len(numbers)):
        if abs(numbers[i] - numbers[i - 1]) >= 10000:
            current_group += 1
        group_ids.append(current_group)

    return group_ids



# Plot with different colors per group
# def plot_coordinates_g(coord_list, filename, outdir='./filename', label_fraction=0.05, title="FITS File Locations"):
def plot_coordinates_g(coord_list, filename, outdir='./filename', title="FITS File Locations"):
    os.makedirs(outdir, exist_ok=True)    
    # plt.figure(figsize=(8, 8))
    # transparent canvas
    plt.figure(figsize=(8, 8), facecolor='none')

    ra_vals = [coord.ra.deg for coord, _ in coord_list]
    dec_vals = [coord.dec.deg for coord, _ in coord_list]
    labels = [label for _, label in coord_list]

    # Assign group numbers based on value jumps ≥ 10000
    group_ids = assign_color_groups(labels)

    # Set up a colormap with as many colors as groups
    num_groups = max(group_ids) + 1
    # cmap = cm.get_cmap('Set3', num_groups)  # use tab10 or Set3 for distinct colors
    cmap = cm.get_cmap('tab10', 5)
    colors = [cmap(group_id % 5) for group_id in group_ids]
    # Create a ListedColormap with just those 5, 他でも使うなら
    # custom_cmap = mcolors.ListedColormap(colors)

    # Scatter plot with group-colored points
    plt.scatter(ra_vals, dec_vals, c=colors, marker='o', s=10, alpha=0.9)

    # Label all or some points if needed
    for coord, label in coord_list:
        plt.text(coord.ra.deg, coord.dec.deg, label, alpha=0.8, fontsize=8, ha='right', va='bottom')
    
    # ラベル文字にも色付け
    # Label with matching colors
    # for (coord, label), group_id in zip(coord_list, group_ids):
    #     text_color = cmap(group_id % 5)
    #     plt.text(coord.ra.deg, coord.dec.deg, label, fontsize=8, alpha=0.8, ha='right', va='bottom',
    #             color=text_color)
    
    plt.xlabel("R.A. (deg)")
    plt.ylabel("Dec. (deg)")
    plt.title(title)
    plt.grid(True)
    # plt.legend()
    plt.gca().invert_xaxis()  # R.A. axis descending
    plt.tight_layout()

    # sm.set_array([])  # required for older matplotlib versions
    # color bar
    # plt.colorbar(sm, label='label')
    
    plot_filename = os.path.join(outdir, f'{filename}.png')
    # 白背景
    # plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    # 透過背景
    plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1, transparent=True)
    #plt.show()
    plt.close()







# === USAGE EXAMPLE ===
if __name__ == "__main__":

    fits_list_path = sys.argv[1]
    fits_files = read_fits_list(fits_list_path)
    filename = os.path.basename(fits_list_path).replace('.txt', '')

    # List your FITS file paths here
    # fits_files = ["example1.fits", "example2.fits"]

    coordinates = []
    for file in fits_files:
        try:
            coord, fname = get_ra_dec_from_fits(file)
            coordinates.append((coord, fname))
        except Exception as e:
            print(f"Skipping {file}: {e}")

    if coordinates:
        # plot_coordinates(coordinates, filename)
        plot_coordinates_g(coordinates, filename)
