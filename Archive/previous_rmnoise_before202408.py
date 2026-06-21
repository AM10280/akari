import sys
import numpy as np
import csv
import astropy.io.fits as fits
from astropy.convolution import convolve, Box1DKernel
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import statsmodels.api as sm
import math
import scipy.optimize
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy import stats
from scipy.ndimage import convolve1d
import copy
import os
import time
import bottleneck as bn

# time1 = time.time()

YES,NO = (1,0)
pltave=NO   # サンプルをプロットする際に平均をプロットするか

# 短冊FITS一つを格納する構造体
# データ処理しやすいようにdata[y][x]ではなくdata[x][y]の形式で保持
class Ycut:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        # Initializing data with shape (nx, ny) as a complex array.
        self.data = np.zeros((nx, ny), dtype=complex)
        # X-axis array (gx) for plotting purposes, ranging from 0 to ny - 1.
        self.gx = np.arange(ny)
        # Average array (ave) for holding the average values of each column.
        self.ave = np.zeros(ny, dtype=complex)


def read_fits_list(fits_list_path):
    """Read the list of FITS files"""
    with open(fits_list_path, 'r') as file:
        file_paths = file.read().splitlines()
#        fitsname = os.path.basename(file_paths)
    return file_paths


# FITSを読み込んで二次元配列 data[][] に格納する
def read_fits(fname):
    """Reads a FITS file and returns the image data and header."""
    with fits.open(fname) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        # print(header[17])
    return data, header


# data[y][x] → ycut.data[x][y-offy]
def data_flip_xy(data, offy):
    # Determine the dimensions of the resulting array after the offset is applied.
    ny, nx = len(data) - offy, len(data[0])
    # Initialize Ycut with the calculated dimensions.
    ycut = Ycut(nx, ny)
    # Apply y-offset, transpose the result, and set it as complex by adding +0j.
    ycut.data = np.transpose(np.array(data)[offy:, :]) + 0j
    
    return ycut


'''

def imhist(data, x1, x2, y1, y2, bins):
    """Draw a histogram of a certain area"""
    region = np.array(data)[y1-1:y2, x1-1:x2].flatten()
    # Plot histogram
    plt.title('Histogram of Sky')
    plt.xlabel('ADU')
    plt.ylabel('Quantity')
    plt.hist(region, bins=bins)
    plt.show()

def data_all(data):
    """Draw a histogram without the blue part of the fits image"""
    exclude_indices = {0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68}
    # Use list comprehension to filter out excluded indices
    alldata = [data[i] for i in range(127) if i not in exclude_indices]
    # Plot histogram
    #plt.xlim(-200,600)
    plt.hist(alldata, bins=400, histtype='barstacked')
    plt.show()
'''


def escape_star(ycut, data, xlim1, xlim2, ylim1, ylim2):
    """Extract information on stars (over and on median *3)"""
    # Calculate the median of the data array for thresholding
    # ave,std = imstat(data, 10,50, 50,250)
    med = np.median(data) #to be updated 04/19/2024
    nx, ny = ycut.nx, ycut.ny
    
    # Initialize yescape to store bright stars
    yescape = Ycut(nx, ny)
    
    # Apply thresholding and data extraction over specified x and y ranges
    star_mask_1 = (ycut.data[xlim1:xlim2, ylim1:ylim2] >= med * 3)
    yescape.data[xlim1:xlim2, ylim1:ylim2] = np.where(star_mask_1, ycut.data[xlim1:xlim2, ylim1:ylim2], 0)
    ycut.data[xlim1:xlim2, ylim1:ylim2][star_mask_1] = np.nan
    
    # Repeat extraction for offset x range
    star_mask_2 = (ycut.data[xlim1+63:xlim2+63, ylim1:ylim2] >= med * 3)
    yescape.data[xlim1+63:xlim2+63, ylim1:ylim2] = np.where(star_mask_2, ycut.data[xlim1+63:xlim2+63, ylim1:ylim2], 0)
    ycut.data[xlim1+63:xlim2+63, ylim1:ylim2][star_mask_2] = np.nan
    
    # Interpolate NaN values in ycut.data using a Gaussian kernel
    stddev = 1.0
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        kernel = Gaussian2DKernel(stddev)
        ycut.data = interpolate_replace_nans(ycut.data, kernel)
        
        if not np.isnan(ycut.data).any():
            break
        stddev += 1
        iteration += 1
#    if iteration < max_iterations:
#        print(f"Interpolated by a Gaussian 2D kernel with the stddev of {stddev}")

    return ycut, yescape


def return_star(ycut, yescape):
    # Restore values where yescape has star data
    restore_mask = yescape.data > 0
    ycut.data[restore_mask] = yescape.data[restore_mask]
    return ycut


# def to1darray(ycut):
def to1darray(ycut, filename):
    # FITSデータ(2次元配列)を1次元配列にする
    ldata = np.zeros(58*ycut.ny)
    for x in range(6, 64):
        for y in range(0, ycut.ny):
            ldata[(x-6)*ycut.ny+y] = ycut.data[x][y].real
            
    rdata = np.zeros(58*ycut.ny)
    for x in range(69, ycut.nx):
        for y in range(0, ycut.ny):
            rdata[(x-69)*ycut.ny+y] = ycut.data[x][y].real
    

    # Prepare the x-axis for the autocorrelation plot
    gx1 = np.linspace(0, 1, 17632)

    '''
    # Plot the time series data (original)
    plt.figure(figsize=(8, 3))
    # plt.ylim(-0.01, 0.1)
    # plt.xlim(-0.01, 0.5)
    plt.title('Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Brightness')
    plt.plot(gx1, np.abs(ldata), color='C0')
    os.makedirs('./tdata', exist_ok=True)
    plot_filename = os.path.join('tdata', f'{filename}_L1.png')
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(8, 3))
    # plt.ylim(-0.01, 0.1)
    # plt.xlim(-0.01, 0.5)
    plt.title('Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Brightness')
    plt.plot(gx1, np.abs(rdata), color='C0')
    os.makedirs('./tdata', exist_ok=True)
    plot_filename = os.path.join('tdata', f'{filename}_R1.png')
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    '''

    #列の合間にダミーデータを入れる
    lave = statistics.mean(ldata)
    dummyl = 0
    ldata2 = [lave]*58*ycut.ny*2
    for x in range(6, 64):
        for y in range(0, ycut.ny):
            ldata2[(x-6)*ycut.ny+dummyl*ycut.ny+y] = ycut.data[x][y].real
        dummyl += 1
    
    rave = statistics.mean(rdata)
    dummyr = 0
    rdata2 = [rave]*58*ycut.ny*2
    for x in range(69, ycut.nx):
        for y in range(0, ycut.ny):
            rdata2[(x-69)*ycut.ny+dummyr*ycut.ny+y] = ycut.data[x][y].real
        dummyr += 1

    # Prepare the x-axis for the autocorrelation plot
    gx2 = np.linspace(0, 1, 35264)

    '''
    # Plot the time series data (original)
    plt.figure(figsize=(8, 3))
    # plt.ylim(-0.01, 0.1)
    # plt.xlim(-0.01, 0.5)
    plt.title('Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Brightness')
    plt.plot(gx2, np.abs(ldata2), color='C0')
    os.makedirs('./tdata', exist_ok=True)
    plot_filename = os.path.join('tdata', f'{filename}_L2.png')
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(8, 3))
    # plt.ylim(-0.01, 0.1)
    # plt.xlim(-0.01, 0.5)
    plt.title('Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Brightness')
    plt.plot(gx2, np.abs(rdata2), color='C0')
    os.makedirs('./tdata', exist_ok=True)
    plot_filename = os.path.join('tdata', f'{filename}_R2.png')
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    '''

    return ldata2, rdata2



def to1darray_1(ycut):
    # convert FITS data (2-dimensional array) to 1-dimensional array
    # Extract rows 6 to 63 (ldata) and 69 to nx-1 (rdata) from the real part of ycut.data
    ldata = ycut.data[6:64, :].real.flatten()
    rdata = ycut.data[69:ycut.nx, :].real.flatten()

    # Calculate the mean of ldata and rdata for dummy values
    lave = np.mean(ldata)
    # lave = statistics.mean(ldata)
    # lave = stats.mstats.tmean(ldata)
    rave = np.mean(rdata)


    # Create ldata2 and rdata2 with alternating rows of lave/rave and actual data values
    ldata2 = np.full(58 * ycut.ny * 2, lave)
    ldata2[::2 * ycut.ny] = ycut.data[6:64, :].real.flatten()

    rdata2 = np.full(58 * ycut.ny * 2, rave)
    rdata2[::2 * ycut.ny] = ycut.data[69:ycut.nx, :].real.flatten()

    return ldata2, rdata2



def ycut_fft(ycut, x1, x2):
    """Perform 1D FFT on all pixel data of Ycut structure
    Perform 1D FFT on each row of Ycut data between specified columns"""
    ny = ycut.ny
    scale_factor = 2.0 / ny  # Precompute scaling factor
    yf = Ycut(ycut.nx, ny)

    # Use vectorized FFT across selected rows
    yf.data[x1-1:x2, :] = fft(ycut.data[x1-1:x2, :], axis=1) * scale_factor
    yf.ave = fft(ycut.ave) * scale_factor
    yf.gx = np.linspace(0, 1.0 / 1.0, ny)
    return yf


def yf_ifft(yf, x1, x2):
    """Perform 1D inverse FFT on all pixel data of Ycut structure
    Perform 1D inverse FFT on each row of yf data between specified columns"""
    ny = yf.ny
    scale_factor = ny / 2.0  # Precompute scaling factor
    ycut = Ycut(yf.nx, ny)

    # Vectorized inverse FFT across selected rows
    ycut.data[x1-1:x2, :] = ifft(yf.data[x1-1:x2, :] * scale_factor, axis=1).real
    ycut.ave = ifft(yf.ave * scale_factor).real
    #ycut.gx  = np.linspace(0, 1.0/1.0, ny)
    return ycut





def calculate_autocorrelation(data2, series, lag_max, ext):
    """Calculates autocorrelation function up to specified lag_max."""
        
    # 自己相関関数の計算 (既存のルーチン)
    llist = pd.Series(data2)
    llist.index = pd.Series(np.ndarray(58*ycut.ny*2))
    lss = sm.tsa.stattools.acf(llist, nlags=lag_max, missing='conservative')
    
    #autocorr_fft(ss)
    yfl1,gxl1 = autocorr_fft(lss, filename, ext)
    N_yfl = len(yfl1)
    yfl = np.zeros(N_yfl)
    for y in range(0,608):
        yfl[y] = np.abs(yfl1[y])

    return np.correlate(series - np.mean(series), series - np.mean(series), mode='full')[len(series)-1:len(series)-1+lag_max]






def autocorr_fft(ss, filename, ext):
    """Perform fft on the autocorrelation function"""
    # Prepare the x-axis for the autocorrelation plot
    gx1 = np.linspace(0, 1, 305)

    '''
    # Create the plot for autocorrelation function
    plt.figure(figsize=(8, 3))
    # plt.subplot(211)
    plt.title('Autocorrelation function')
    plt.xlabel('$\u03c4$')
    plt.ylabel('R($\u03c4$)')
    plt.ylim(-1, 1)
    plt.plot(gx1, ss, color='C0' if ext in ['_L', '_R'] else 'C3')
    os.makedirs('./autocorr', exist_ok=True)
    plot_filename = os.path.join('autocorr', f'{filename}{ext}.png')
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()
    '''


    # generate d1 for two-sided FFT
    d1 = np.concatenate([ss[::-1], ss])  # Reverse and concatenate ss
    yf1 = fft(d1) / len(d1)  # Perform FFT

    # Prepare the x-axis for the FFT plot
    gx1 = np.linspace(0, 1, len(d1))


    # Plot the power spectrum
    plt.figure(figsize=(8, 3))
    plt.ylim(-0.01, 0.1)
    plt.xlim(-0.01, 0.5)
    plt.title('Power Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.plot(gx1, np.abs(yf1), color='C0' if ext in ['_L', '_R'] else 'C3')
    plot_filename = os.path.join('plot', f'{filename}{ext}.png')
    # plot_filename = os.path.join('/Users/yamamura/Desktop/to_weka/IRCMap_doublestar/Qnoise/u_20230818/test/plot/' + filename + ext + '.png')
    # print(plot_filename)
    os.makedirs('./ps', exist_ok=True)
    plot_filename = os.path.join('ps', f'{filename}{ext}.png')
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return yf1, gx1


def autocorr_fft_1(ss):
    """Compute FFT on autocorrelation function."""
    d1 = np.concatenate([ss[::-1], ss])
    yf1 = fft(d1) / len(d1)
    return yf1



#def delta_m_ave_self_fft(gx,yf,filename,side):
def delta_m_ave_self_fft(gx, yf, filename, side, snumber):
    """Determine mask position from difference between moving average and actual function"""
    # rolling_yf = np.convolve(np.abs(yf), np.ones(25)/25, mode='same')
    rolling_yf = bn.move_mean(np.abs(yf),window=25)
    
    # Apply the modification to remove 'nan_roll' effect in a single operation
    # rolling_yf[12:500] = rolling_yf[24:500] #(0,292)–(0,583)
    # Adjust rolling_yf in a single operation  #(0,292)–(0,583)
    rolling_yf[12:512] = rolling_yf[24:524]

    
    delta_rangel = 0.09
    # fity_selfFFT_std = np.zeros(304 - math.floor(304 * delta_rangel * 2))
    # fity_selfFFT = np.zeros(608)
    # msk_range = np.zeros(608)

    # Compute the difference between the power spectrum and rolling average
    fity_selfFFT = np.where((gx > delta_rangel) & (gx <= 0.5), np.abs(yf) - rolling_yf, 0)
    fity_selfFFT_std = fity_selfFFT[fity_selfFFT != 0]

#    fity_selfFFT[delta_rangel < gx] = np.abs(yf[delta_rangel < gx]) - rolling_yf[delta_rangel < gx]
#    fity_selfFFT_std[:len(fity_selfFFT)] = fity_selfFFT[delta_rangel < gx]
    
    std = np.std(fity_selfFFT_std)
    # std = bn.nanstd(fity_selfFFT_std)
    # std = statistics.stdev(fity_selfFFT_std)  # 標本標準偏差
    # print(std)

    # Determine mask range
    # Set mask for values exceeding 3*std
    msk_range = (fity_selfFFT >= 3 * std).astype(int)
    # msk_range[fity_selfFFT >= 3 * std] = 1

    '''
    # Plotting the power spectrum, moving average, and mask range
    plt.figure(figsize=(8, 3))
    plt.xlim(-0.01, 0.5)
    plt.ylim(-0.01, 0.1)
    plt.plot(gx, np.abs(yf), label='power spectrum')
    plt.plot(gx, rolling_yf, label='moving average', alpha=0.7)
    plt.title('Power spectrum, moving average and mask range')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.plot(gx, msk_range, alpha=0.5)
    #plt.legend()
    # Save the plot
    os.makedirs('./premask', exist_ok=True)
    pmask_filename = os.path.join('premask', f'{filename}{snumber}.png')
    plt.savefig(pmask_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    '''

    return msk_range, std, rolling_yf





def rm_noise_PS(gx, yf, rolling_yf, msk_range, filename, snumber):
    length = len(yf)
    mid = length // 2   # 305

    # Vectorized noise removal
    noisy_indices = np.where(msk_range[:mid] == 1)[0]
    for y in noisy_indices:
        # Get non-noisy neighbors within a range of 3 on each side
        neighbors = [yf[y + i] for i in range(1, 4) if y + i < length and msk_range[y + i] == 0] + \
                    [yf[y - i] for i in range(1, 4) if y - i >= 0 and msk_range[y - i] == 0]
        
        if neighbors:
            yf[y] = np.mean(neighbors)

    # Enforce symmetry
    # yf[mid + 1:] = yf[:mid + 1][::-1]
    yf[mid:] = yf[:mid][::-1]

    '''
    # Plotting
    plt.figure(figsize=(8, 3))
    plt.xlim(-0.01, 0.5)
    plt.ylim(-0.01, 0.1)
    plt.plot(gx, np.abs(yf), label='power spectrum')
    plt.plot(gx, rolling_yf, label='moving average', alpha=0.7)
    plt.title('Power Spectrum and Moving Average')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.plot(gx, msk_range, alpha=0.7)
    # plt.legend()
    
    # File saving with efficient directory creation and file naming
    rnp_filename = os.path.join('rnp', f'{filename}{snumber}.png')
    os.makedirs(os.path.dirname(rnp_filename), exist_ok=True)
    plt.savefig(rnp_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    '''

    return yf




def delta_PS_move_ave(gx, yf, rolling_yf, base_std, filename, ext):
    delta_rangel = 0.09
    # fity_selfFFT = np.zeros(608)
    # msk_range = np.zeros(608)

    # Computation of fity_selfFFT
    fity_selfFFT = np.where((gx > delta_rangel) & (gx <= 0.5), np.abs(yf) - rolling_yf, 0)
    # mask_indices = (gx > delta_rangel) & (gx <= 0.5)
    # fity_selfFFT[mask_indices] = np.abs(yf[mask_indices]) - rolling_yf[mask_indices]

    # Determine the mask range based on the standard deviation threshold
    # msk_range[fity_selfFFT >= 5 * base_std] = 1

    # Extend the mask to adjacent values
    # msk_range[1:] = np.logical_or(msk_range[1:], msk_range[:-1])
    # msk_range[:-1] = np.logical_or(msk_range[:-1], msk_range[1:])

    # msk_range = np.logical_or(msk_range, np.roll(msk_range, 1))
    # msk_range = np.logical_or(msk_range, np.roll(msk_range, -1))


    # Apply the threshold and set mask with vectorized operations, including neighboring points
    msk_range = (fity_selfFFT >= 5 * base_std).astype(int)
    msk_range[:-1] |= msk_range[1:]
    msk_range[1:] |= msk_range[:-1]


    save_diagram_data(np.abs(yf), 'power_spectrum_y', ext)
    save_diagram_data(msk_range, 'mask_range', ext)

    # Plot power spectrum, moving average, and mask range
    plt.figure(figsize=(8, 3))
    plt.xlim(-0.01, 0.5)
    plt.ylim(-0.01, 0.1)
    plt.plot(gx, np.abs(yf), label='power spectrum')
    plt.plot(gx, rolling_yf, label='moving average', alpha=0.7)
    plt.plot(gx, msk_range, label='mask range', alpha=0.5)
    plt.title('Power Spectrum and Moving Average with Mask Range')
    plt.xlabel('Frequency')
    plt.ylabel('Power')

    mask_filename = os.path.join('mask', f"{filename}{ext}.png")
    os.makedirs('./mask', exist_ok=True)
    plt.savefig(mask_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return msk_range




# power spectrum and mask range diagram



# fits_list_path
def save_diagram_data(target, diagram_name, ext):
# def save_diagram_data(self, yf, ext):
    """Saves yf data to appropriate files based on 'ext' (Left or Right channel)"""
    # abs_yf = np.abs(yf)  # Compute absolute values only once for efficiency

    if ext == '_L':
        # Save data in text and CSV formats for the left channel
        with open(f'diagram_l_{diagram_name}.csv', 'a') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(target.tolist())           # Write to CSV file
        np.save(f'power_l_{diagram_name}.npy', target)  # Save binary data to .npy for faster access

    elif ext == '_R':
        # Save data in text and CSV formats for the right channel
        with open(f'diagram_r_{diagram_name}.csv', 'a') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(target.tolist())           # Write to CSV file    





# pt diagram

# fits_list_path
def save_diagram_data_ps(yf, ext):
# def save_diagram_data(self, yf, ext):
    """Saves yf data to appropriate files based on 'ext' (Left or Right channel)"""
    abs_yf = np.abs(yf)  # Compute absolute values only once for efficiency

    if ext == '_L':
        # Save data in text and CSV formats for the left channel
        with open('diagram_l.txt', 'a') as f_txt, open('diagram_l.csv', 'a') as f_csv:
            # f_txt.write(str(abs_yf.tolist()) + "\n")  # Write to text file
            writer = csv.writer(f_csv)
            writer.writerow(abs_yf.tolist())           # Write to CSV file
        np.save('power_l.npy', abs_yf)  # Save binary data to .npy for faster access

    elif ext == '_R':
        # Save data in text and CSV formats for the right channel
        with open('diagram_r.txt', 'a') as f_txt, open('diagram_r.csv', 'a') as f_csv:
            # f_txt.write(str(abs_yf.tolist()) + "\n")  # Write to text file
            writer = csv.writer(f_csv)
            writer.writerow(abs_yf.tolist())           # Write to CSV file
        np.save('power_r.npy', abs_yf)  # Save binary data to .npy for faster access

def save_npz(self, a, b):
    """Save multiple arrays in a single compressed file."""
    np.savez('power.npz', a=a, b=b)





    
# pt diagram

#    with open('diagram.txt', 'a') as f:
#        print(gx, filename, np.abs(yf), sep='#', end='owari', file=f)
#        f.write(gx, filename, np.abs(yf))


#Left Right
    if ext == '_L' :
#        np.save('power_l.npy', np.abs(yf))
        with open('diagram_l.txt', 'a') as f:
            print(np.abs(yf).tolist(), file=f)
#            np.save(f, np.abs(yf))

        with open('diagram_l.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(np.abs(yf).tolist())

    if ext == '_R' :
#        np.save('power_r.npy', np.abs(yf))
        with open('diagram_r.txt', 'a') as f:
            print(np.abs(yf), file=f)
#            np.save(f, np.abs(yf))

        with open('diagram_r.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(np.abs(yf).tolist())

#        print(np.abs(yf).tolist())



# np.savez('power.npz', a=a, b=b)
    





def rm_noise_6data_astropy(yf, msk_range, xlim1, xlim2):
    """Remove noise for FFT-ed data
    Fill with the average of 6 surrounding non-noise points"""
    ny = yf.ny
    
    # Define a 1D box kernel with a width of 7 (covers up to 3 points on each side)
    kernel = Box1DKernel(7)

    # Generate a boolean mask indicating noisy points from msk_range
    noise_mask = (msk_range[::2] == 1)
    
    for x in range(xlim1, xlim2):
        # Get the data for the current column, apply noise mask
        data_col = yf.data[x].real
        masked_data = np.where(noise_mask, np.nan, data_col)

        # Use astropy's convolve to average around noisy points, ignoring NaNs
        smoothed_col = convolve(masked_data, kernel, boundary='extend', nan_treatment='interpolate')

        # Replace only the noisy points in yf.data with the smoothed values
        yf.data[x].real[noise_mask] = smoothed_col[noise_mask]

    return


def rm_noise_6data_scipy(yf, msk_range, xlim1, xlim2):
    """Remove noise for FFT-ed data
    Fill with the average of 6 surrounding non-noise points"""
    ny = yf.ny
    # Define the convolution kernel for averaging with up to 3 non-noisy neighbors on each side
    kernel = np.array([1, 1, 1, 0, 1, 1, 1])

    for x in range(xlim1, xlim2):
        # Mask to identify non-noisy and noisy points in the current column
        is_noisy = msk_range[::2] == 1

        # Apply convolution only to non-noisy points for averaging
        smooth_values = convolve1d(yf.data[x].real, kernel, mode='constant', cval=0.0) / kernel.sum()
        
        # Replace noisy points with the averaged values from their neighbors
        yf.data[x].real[is_noisy] = smooth_values[is_noisy]
        
    return


# Remove noise for FFT-ed data
# Fill with the average of 6 surrounding non-noise points
def rm_noise_6data(yf, msk_range, xlim1, xlim2):
    """Remove noise for FFT-ed data
    Fill with the average of 6 surrounding non-noise points"""
    ny = yf.ny
    for x in range(xlim1, xlim2):
        for y in range(ny):
            mask_idx = 2 * y
            if msk_range[mask_idx] == 1: # Noisy point
                nm = 0  # Accumulator for non-noisy neighbors
                n = 0   # Counter for valid non-noisy neighbors
                # Look forward for up to 3 non-noisy neighbors
                for i in range(1, 4):
                    if y + i < ny and msk_range[mask_idx + i] == 0:
                        nm += yf.data[x][y + i]
                        n += 1
                # Look backward for up to 3 non-noisy neighbors
                for j in range(1, 4):
                    if y - j >= 0 and msk_range[mask_idx - j] == 0:
                        nm += yf.data[x][y - j]
                        n += 1
                if n > 0:
                    ave = nm / n
                    yf.data[x][y] = ave
    return





def data_rflip_xy_save(ycut, data, header, offy, filename):
    """ycut.data[x][y-offy]→ data[y][x]->save fits
    data flipping and saving to FITS"""
    data[offy:offy + ycut.ny, :ycut.nx] = ycut.data.real.T
    
    # Save to FITS using astropy
    path = os.getcwd()
    hdu = fits.PrimaryHDU(data, header)
    hdulist = fits.HDUList([hdu])
    os.makedirs('./output', exist_ok=True)
    hdulist.writeto(f'{path}/output/{filename}.fits', overwrite=True)
    return data

def data_rflip_xy(ycut, data, offy):
    """data flipping without saving (just modification)"""
    data[offy:offy + ycut.ny, :ycut.nx] = ycut.data.real.T
    return data


'''
# datax2を同じカラースケールで並べて表示する
def fitsdsp_comp(data1,data2, vmin,vmax):
    ave,std = imstat(data1, 10,60, 100,250)
    plt.subplot(1,2,1)
    plt.title('before (%.1f$\pm$%.1f)' % (ave,std))
    plt.imshow(data1, vmin=vmin, vmax=vmax, origin='lower', cmap='plasma')
    
    ave,std = imstat(data2, 10,60, 100,250)
    plt.subplot(1,2,2)
    plt.title('after (%.1f$\pm$%.1f)' % (ave,std))
    plt.imshow(data2, vmin=vmin, vmax=vmax, origin='lower', cmap='plasma')
    
    plt.colorbar(aspect=40, pad=0.08, orientation='vertical')
    plt.show()
    return()


# data[y][x] の (x1:x2,y1:y2) の領域の統計をとる
def imstat(data, x1,x2, y1,y2):
    n=0
    Sx=0.
    for y in range(y1-1, y2):
        for x in range(x1-1, x2):
            Sx += data[y][x]
            n  += 1
    ave = Sx/n

    n=0
    Sx=0.
    Sxx=0.
    for y in range(y1-1, y2):
        for x in range(x1-1, x2):
            Sx += data[y][x]
            Sxx+= (data[y][x]-ave)*(data[y][x]-ave)
            n  += 1
    ave = Sx/n
    std = np.sqrt(Sxx/n)
    return(ave, std)

# datax2の差分を表示する
def fitsdsp_diff(data1,data2, vmin,vmax):
    data_diff = data2 - data1
    ave,std = imstat(data_diff, 10,60, 60,300)
    plt.title('diff (%.1f$\pm$%.1f)' % (ave,std))
    plt.imshow(data_diff, vmin=vmin, vmax=vmax, origin='lower', cmap='plasma')
    plt.colorbar(aspect=40, pad=0.08, orientation='vertical')
    plt.show()
    return()
'''




# フーリエ変換の結果をプロットする
def yf_plot(yf, x1,x2, filename, mode, ext):
    plt.figure(figsize=(8,3))
    plt.xlim(1e-5,1.0-1e-5)
    #plt.ylim(1,1e6)
    if   mode == "abs":
        for x in range(x1-1, x2):
            plt.title("FFT of raw data")
            plt.ylim(0,50)
            #plt.plot(yf.gx*2, np.abs(yf.data[x-1]) * 10**(x-1))
            #plt.plot(yf.gx*2, np.abs(yf.data[x-1]))
            plt.plot(yf.gx*2, np.abs(yf.data[x-1]), color = 'C0' if ext == '_b' else 'C3')
        if(pltave):
            plt.plot(yf.gx*2, np.abs(yf.ave) * 10**(x2-1), color = 'C0' if ext == '_b' else 'C3')
        #plt.xscale('log')
        #plt.yscale('log')
    elif mode == "real":
        for x in range(x1-1, x2):
            plt.plot(yf.gx*2, yf.data[x-1].real, color = 'C0' if ext == '_b' else 'C3') # + 50*x)
        if(pltave):
            plt.plot(yf.gx*2, yf.ave.real, color = 'C0' if ext == '_b' else 'C3') #  + 50*x2)
        #plt.ylim(0,500)
        plt.ylim(-50,50)        
#        popt,pcov = fit_yfreal(yf.gx, yf.data[x-1].real)
#        plt.plot(yf.gx, funcR(yf.gx, popt[0],popt[1],popt[2]))
#        plt.plot(yf.gx, yf.data[x-1].real-funcR(yf.gx, popt[0],popt[1],popt[2]))
    elif mode == "imag":
        for x in range(x1-1, x2):
            plt.plot(yf.gx*2, yf.data[x-1].imag, color = 'C0' if ext == '_b' else 'C3')# + 40*x)
            plt.ylim(-100,100)
            #plt.ylim(-100,100)
#            popt,pcov = fit_yfimag(yf.gx, yf.data[x-1].imag)
#            plt.plot(yf.gx, yf.data[x-1].imag-funcI(yf.gx, popt[0],popt[1],popt[2],popt[3]))
#            plt.plot(yf.gx, funcI(yf.gx, popt[0],popt[1],popt[2],popt[3]))
            

        if(pltave):
            plt.plot(yf.gx*2, yf.ave.imag + 40*x2, color = 'C0' if ext == '_b' else 'C3')
    else:
        print('No')
    #プロット グラフ表示
    plt.title('FFT of raw data')
    #plt.title('FFT of raw data (masked)')
    plt.xlabel('frequency')
    plt.ylabel('power')

    os.makedirs('./fft', exist_ok=True)
    fft_filename = os.path.join('fft/' + filename + ext + '.png')
    plt.savefig(fft_filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
#    plt.savefig('ypplot' + ext + '.png')
#    plt.show()
    plt.close()
    return()




####  main() ####


def rmnoise(filename):
    
    time1 = time.time()
#    elapsed_time = time1 - time0
#    print(f"初期化時間：{elapsed_time}")

#    args = sys.argv
#    fitsname = args[1]

    #fitsname = args[2]
    #filename = fitsname.rstrip('.fits')
    #fitsname_RS = fitsname.replace('_RS.fits','')
    fitsname = filename + '.fits' # fitsname = '{}{}'.format(filename, '.fits')
    fitsname_RS = filename + '_RS.fits'
    
    x1,x2 = (40,40) # サンプルとして表示する x(pixel) の範囲
    offy  = 2       # スキップする y(sampling) の範囲
    lag_max = 304
    
    (data,header) = read_fits(fitsname) # FITSを読み込む
    data0 = copy.deepcopy(data)
    
    # FITSデータを加工する
    # (data[y][x]→data[x][y]変換, 開始からoffy回分のsamplingを除く)
    ycut = data_flip_xy(data, offy)
    
    #天体の情報を抜く（改良版）
    # More than 3 times the median of the histogram of surface brightness is excluded from the mask level as a bright signal. # median multiplied by 3 or more
    ycut, yescape = escape_star(ycut,data, 0,63, 0,lag_max)
    
    # convert FITS data (2-dimensional array) to 1-dimensional array
    # Extract rows 6 to 63 (ldata) and 69 to nx-1 (rdata) from the real part of ycut.data
    # Create ldata2 and rdata2 with alternating rows of lave/rave and actual data values
    ldata2, rdata2 = to1darray(ycut, filename)

    
    # 自己相関関数の計算 (既存のルーチン)
    llist = pd.Series(ldata2)
    llist.index = pd.Series(np.ndarray(58*ycut.ny*2))
    lss = sm.tsa.stattools.acf(llist, nlags=lag_max, missing='conservative')
    
    #autocorr_fft(ss)
    yfl1,gxl1 = autocorr_fft(lss, filename, "_L")
    N_yfl = len(yfl1)
    yfl = np.zeros(N_yfl)
    for y in range(0,608):
        yfl[y] = np.abs(yfl1[y])
    
    #moving averageと実際の関数の差からマスク位置を決定
    msk_rangel1,stdl1,rolling_yfl1 = delta_m_ave_self_fft(gxl1,yfl1,filename,'left','_l1')
    #msk_rangel1,stdl1,rolling_yfl1 = delta_m_ave_self_fft(gxl1,yfl1,filename,'left')
    
    #パワースペクトルにマスクをかけてスムージング
    yfl2 = rm_noise_PS(gxl1,yfl1,rolling_yfl1,msk_rangel1,filename,'_l1')
    #yfl2 = rm_noise_PS(yfl1,msk_rangel1,filename)
    
    #パワースペクトルにマスクをかけてスムージング
    msk_rangel2,stdl2,rolling_yfl2 = delta_m_ave_self_fft(gxl1,yfl2,filename,'left','_l2')
    #msk_rangel2,stdl2,rolling_yfl2 = delta_m_ave_self_fft(gxl1,yfl2,filename,'left')
    yfl3 = rm_noise_PS(gxl1,yfl2,rolling_yfl2,msk_rangel2,filename,'_l2')
    #yfl3 = rm_noise_PS(yfl2,msk_rangel2,filename)
    
    #moving averageと実際の関数の差からマスク位置を決定
    msk_rangel3,stdl3,rolling_yfl3 = delta_m_ave_self_fft(gxl1,yfl3,filename,'left','_l3')
    #msk_rangel3,stdl3,rolling_yfl3 = delta_m_ave_self_fft(gxl1,yfl3,filename,'left')
    
    #２行目の自己相関関数
    rlist = pd.Series(rdata2)
    rlist.index = pd.Series(np.ndarray(58*ycut.ny*2))
    rss = sm.tsa.stattools.acf(rlist, nlags=lag_max, missing='conservative')
    
    yfr1,gxr1 = autocorr_fft(rss, filename, "_R")
    N_yfr = len(yfr1)
    yfr = np.zeros(N_yfr)
    for y in range(0,608):
        yfr[y] = np.abs(yfr1[y])
    
    
    
    #moving averageと実際の関数の差からマスク位置を決定
    msk_ranger1,stdr1,rolling_yfr1 = delta_m_ave_self_fft(gxr1,yfr1,filename,'right','_r1')
    #msk_ranger1,stdr1,rolling_yfr1 = delta_m_ave_self_fft(gxr1,yfr1,filename,'right')
    
    #パワースペクトルにマスクをかけてスムージング
    yfr2 = rm_noise_PS(gxr1,yfr1,rolling_yfr1,msk_ranger1,filename,'_r1')
    #yfr2 = rm_noise_PS(yfr1,msk_ranger1,filename)
    
    #パワースペクトルにマスクをかけてスムージング
    msk_ranger2,stdr2,rolling_yfr2 = delta_m_ave_self_fft(gxr1,yfr2,filename,'right','_r2')
    #msk_ranger2,stdr2,rolling_yfr2 = delta_m_ave_self_fft(gxr1,yfr2,filename,'right')
    yfr3 = rm_noise_PS(gxr1,yfr2,rolling_yfr2,msk_ranger2,filename,'_r2')
    #yfr3 = rm_noise_PS(yfr2,msk_ranger2,filename)
    
    #moving averageと実際の関数の差からマスク位置を決定
    msk_ranger3,stdr3,rolling_yfr3 = delta_m_ave_self_fft(gxr1,yfr3,filename,'right','_r3')
    #msk_ranger3,stdr3,rolling_yfr3 = delta_m_ave_self_fft(gxr1,yfr3,filename,'right')
    
    #moving averageと実際の関数の差からマスク位置を決定(正確)
    mask_rangel = delta_PS_move_ave(gxl1,yfl,rolling_yfl3,stdl3,filename,'_L')
    mask_ranger = delta_PS_move_ave(gxr1,yfr,rolling_yfr3,stdr3,filename,'_R')
    
    
    #FFTをかける
    yf = ycut_fft(ycut, 1,ycut.nx)
    
    # yf_plot(yf, x1,x2, filename, "abs", "_b")
    
    #自動で決定した範囲のノイズを除去
    #rm_noise_left_6data(yf,mask_rangel)
    #rm_noise_right_6data(yf,mask_ranger)
    rm_noise_6data(yf,mask_rangel,0,64)
    rm_noise_6data(yf,mask_ranger,64,yf.nx)
    
    # yf_plot(yf, x1,x2, filename, "abs", "_a")     # x1,x2で指定した範囲を1列ずつFFTして表示
    
    # inverse FFTをかける
    ycut = yf_ifft(yf, 1,yf.nx)
    
    
    #天体の情報を元に戻す（オリジナル）
    ycut = return_star(ycut,yescape)
    
    
    # 処理後のデータのFFTプロット
    # FITSデータ(2次元配列)を1次元配列にする
    ldata = np.zeros(58*ycut.ny)
    for x in range(6, 64):
        for y in range(0, ycut.ny):
            ldata[(x-6)*ycut.ny+y] = ycut.data[x][y].real
            
    rdata = np.zeros(58*ycut.ny)
    for x in range(69, ycut.nx):
        for y in range(0, ycut.ny):
            rdata[(x-69)*ycut.ny+y] = ycut.data[x][y].real
    
    #列の合間にダミーデータを入れる
    # lave = statistics.mean(ldata)
    lave = np.mean(ldata)
    dummyl = 0
    ldata2 = [lave]*58*ycut.ny*2
    for x in range(6, 64):
        for y in range(0, ycut.ny):
            ldata2[(x-6)*ycut.ny+dummyl*ycut.ny+y] = ycut.data[x][y].real
        dummyl += 1
    
    # rave = statistics.mean(rdata)
    rave = np.mean(rdata)
    dummyr = 0
    rdata2 = [rave]*58*ycut.ny*2
    for x in range(69, ycut.nx):
        for y in range(0, ycut.ny):
            rdata2[(x-69)*ycut.ny+dummyr*ycut.ny+y] = ycut.data[x][y].real
        dummyr += 1
    
    lag_max = 304
    
    llist = pd.Series(ldata2)
    llist.index = pd.Series(np.ndarray(58*ycut.ny*2))
    lss = sm.tsa.stattools.acf(llist, nlags=lag_max, missing='conservative')
    yfl1,gxl1 = autocorr_fft(lss, filename, "_LA")
    
    rlist = pd.Series(rdata2)
    rlist.index = pd.Series(np.ndarray(58*ycut.ny*2))
    rss = sm.tsa.stattools.acf(rlist, nlags=lag_max, missing='conservative')
    yfr1,gxr1 = autocorr_fft(rss, filename, "_RA")
    
    #天体の情報を元に戻す（2024/04/26）
    #ycut = return_star(ycut,yescape)
    
    # データをXY-flipしてFITSに戻す
    data_rflip_xy_save(ycut, data, header, offy, filename)
    #data_rflip_xy(ycut, data, offy)
        
    
    # FITSを表示
    #fitsdsp_comp(data0, data, -10,250)
    #fitsdsp_diff(data0, data, -10,10)
    
    time2 = time.time()
    elapsed_time = time2-time1
    print(f"経過時間：{elapsed_time}")
    
    return()


def rmnoise_list(fits_list_path):    
    input_files = read_fits_list(fits_list_path)
    for f in input_files:
        rmnoise(f)


if __name__ == "__main__":
    file_list_path = sys.argv[1]
    rmnoise_list(file_list_path)



# pt diagram
