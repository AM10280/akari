#自動でノイズ範囲を決定→マスク（高速）
import sys
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import statistics
import statsmodels.api as sm
import pandas as pd
import math
import scipy.optimize
from scipy import stats
import copy
import os
import time
import bottleneck as bn

time1 = time.time()

YES,NO = (1,0)
pltave=NO   # サンプルをプロットする際に平均をプロットするか

# 短冊FITS一つを格納する構造体
# データ処理しやすいようにdata[y][x]ではなくdata[x][y]の形式で保持
class Ycut:
    def __init__(self, nx,ny):
        self.nx = nx
        self.ny = ny
        self.data = np.zeros((nx,ny), dtype=complex) # data[nx][ny] 複素数
        self.gx   = np.arange(0,ny)                  # gx[ny]  X軸用
        self.ave  = np.zeros(ny, dtype=complex)      # ave[ny] 平均

# FITSを読み込んで二次元配列 data[][] に格納する
def data_from_fits(fname):
    hdulist = fits.open(fname)
    hdu  = hdulist[0]
    data = hdu.data
    header = hdu.header
    print(header[17])
    return(data, header)

# data[y][x]→ ycut.data[x][y-offy]
def data_flip_xy(data, offy):
    ny,nx = (len(data)-offy, len(data[0]))
    ycut = Ycut(nx,ny)
    for x in range(0, nx):
        for y in range (0, ny):
            ycut.data[x][y] = complex(data[y+offy][x], 0)
    return(ycut)

# ある領域のヒストグラムを書く
def imhist(data, x1,x2, y1,y2, bins):
    num = (x2-x1+1)*(y2-y1+1)
    arr = np.zeros(num)
    i=0;
    for y in range(y1-1, y2):
        for x in range(x1-1, x2):
            arr[i] = data[y][x]
            i += 1
    plt.title('histogram of sky')
    plt.xlabel('ADU')
    plt.ylabel('quantity')
    plt.hist(arr, bins=bins)
    plt.show()
    return()

##fits画像の青い部分を抜いたヒストグラムを画く
def data_all(data, i):
    alldata = []
    for i in range (0, 127):
        if i == (0,1,2,3,4,5,64,65,66,67,68):
            i += 1
        else:
            alldata.append(data[i])
            i += 1
    #plt.xlim(-200,600)
    plt.hist(alldata,histtype='barstacked',bins = 400)
    plt.show()
    return()

#天体(over400)の情報を抜く
def escape_star(ycut,data,xlim1,xlim2, ylim1,ylim2):
    ave,std = imstat(data, 10,50, 50,250)
    nx,ny = (ycut.nx, ycut.ny)
    yescape = Ycut(nx,ny)
    for x in range(xlim1, xlim2):
        for y in range(ylim1, ylim2):
            if ycut.data[x][y] >= 400:
                yescape.data[x][y] = ycut.data[x][y]
                ycut.data[x][y] = ave
            else:
                yescape.data[x][y] = 0
    for x in range(xlim1+63, xlim2+63):
        for y in range(ylim1, ylim2):
            if ycut.data[x][y] >= 400:
                yescape.data[x][y] = ycut.data[x][y]
                ycut.data[x][y] = ave
            else:
                yescape.data[x][y] = 0

    return(ycut,yescape)

#天体の情報を元に戻す
def return_star(ycut,yescape):
    for x in range(0, ycut.nx):
        for y in range(0, ycut.ny):
            if yescape.data[x][y] > 0:
                ycut.data[x][y] = yescape.data[x][y]
    return(ycut)

#ノイズを除去_左(6data)
def rm_noise_left_6data(yf,msk_range):
    for x in range(0,64):
        for y in range(0,yf.ny):
            if msk_range[2*y] == 0.1:
                nm = 0
                n = 0
                i = 0
                while n < 3:
                    if msk_range[2*y+1+i] == 0:
                        nm += yf.data[x][y-3+n]+yf.data[x][y+1+i]
                        n += 1
                        i += 1
                    elif msk_range[2*y+1+i] != 0:
                        i += 1
                ave = nm/n/2
                yf.data[x][y] = ave
                yf.data[x][y-1] = ave
                yf.data[x][y+1] = ave
            if msk_range[2*y+1] == 0.1:
                nm = 0
                n = 0
                i = 0
                while n < 3:
                    if msk_range[2*y+2+i] == 0:
                        nm += yf.data[x][y-3+n]+yf.data[x][y+2+i]
                        n += 1
                        i += 1
                    elif msk_range[2*y+2+i] != 0:
                        i += 1
                ave = nm/n/2
                yf.data[x][y] = ave
                yf.data[x][y+1] = ave
                yf.data[x][y-1] = ave
                yf.data[x][y+2] = ave
    return()

#ノイズを除去_右(6data)
def rm_noise_right_6data(yf,msk_range):
    for x in range(64,yf.nx):
        for y in range(0,yf.ny):
            if msk_range[2*y] == 0.1:
                nm = 0
                n = 0
                i = 0
                while n < 3:
                    if msk_range[2*y+1+i] == 0:
                        nm += yf.data[x][y-3+n]+yf.data[x][y+1+i]
                        n += 1
                        i += 1
                    elif msk_range[2*y+1+i] != 0:
                        i += 1
                ave = nm/n/2
                yf.data[x][y] = ave
                yf.data[x][y-1] = ave
                yf.data[x][y+1] = ave
            if msk_range[2*y+1] == 0.1:
                nm = 0
                n = 0
                i = 0
                while n < 3:
                    if msk_range[2*y+2+i] == 0:
                        nm += yf.data[x][y-3+n]+yf.data[x][y+2+i]
                        n += 1
                        i += 1
                    elif msk_range[2*y+2+i] != 0:
                        i += 1
                ave = nm/n/2
                yf.data[x][y] = ave
                yf.data[x][y+1] = ave
                yf.data[x][y-1] = ave
                yf.data[x][y+2] = ave
    return()

#パワースペクトルにマスクをかけてスムージング
def rm_noise_PS(yf,msk_range):
    for y in range(0,304):
        if msk_range[y] == 0.1:
            nm = 0
            n = 0
            i = 0
            while n < 3:
                if msk_range[y+i] == 0:
                    nm += np.abs(yf[y-3+n])+np.abs(yf[y+1+i])
                    n += 1
                    i += 1
                elif msk_range[y+i] != 0:
                    i += 1
            ave = nm/n/2
            yf[y] = ave

    for y in range(0,303):
        yf[607-y] = yf[y]
    
    return(yf)

# Ycut構造体の全pixelデータについて、1次元FFTをする
def ycut_fft(ycut, x1,x2):
    nx,ny = (ycut.nx, ycut.ny)
    yf = Ycut(nx,ny)
    for x in range(x1-1, x2):
        yf.data[x] = np.fft.fft(ycut.data[x])/(ny/2.)
    yf.ave = np.fft.fft(ycut.ave)/(ny/2.)
    yf.gx  = np.linspace(0, 1.0/1.0, ny)
    return(yf)

# Ycut構造体の全pixelデータについて、1次元のinverse FFTをする
def yf_ifft(yf, x1,x2):
    nx,ny = (yf.nx, yf.ny)
    ycut = Ycut(nx,ny)
    for x in range(x1-1, x2):
        ycut.data[x] = np.fft.irfft(yf.data[x]*(ny/2.), ny)
    ycut.ave = np.fft.irfft(yf.ave*(ny/2.), ny)
    #ycut.gx  = np.linspace(0, 1.0/1.0, ny)
    return(ycut)

# ycut.data[x][y-offy]→ data[y][x]->save fits
def data_rflip_xy_save(ycut, data, offy, filename):
    for x in range(0, ycut.nx):
        for y in range (0, ycut.ny):
            data[y+offy][x] = ycut.data[x][y].real
            
    path = os.getcwd()
    os.chdir('%s/output'%path)
    hdu = fits.PrimaryHDU(data, header)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto('%s.fits' %filename)
    return(data)

# ycut.data[x][y-offy]→ data[y][x]
def data_rflip_xy(ycut, data, offy):
    for x in range(0, ycut.nx):
        for y in range (0, ycut.ny):
            data[y+offy][x] = ycut.data[x][y].real
    return(data)

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


# 自己相関関数をFFT
def test_fft_compare(ss):
    # 両側FFT
    d1 = np.zeros(304*2)
    for y in range(0,304):
        d1[304-y] = ss[y]
        d1[304+y] = ss[y]
    yf1 = np.fft.fft(d1)/(304*2/2)
    gx1 = np.linspace(0,1,304*2)

    # グラフ表示
    #plt.figure(figsize=(8,5))
    #plt.subplot(211)
    #plt.title('autocorrelation function and power spectrum')
    #plt.title('autocorrelation function')
    #plt.xlabel('$\u03c4$')
    #plt.ylabel('R($\u03c4$)')
    #plt.ylim(-1,1)
    #plt.plot(gx1*2-1,d1)
    #plt.show()
    
    plt.subplot(212)
    plt.ylim(-0.01,0.1)
    plt.xlim(-0.01,0.5)
    plt.title('power spectrum')
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.plot(gx1, np.abs(yf1))
    
    plot_filename = os.path.join('plot/' + filename + '.png')
    plt.savefig(plot_filename, pad_inches=1)
    #plt.show()
    
    return(yf1,gx1)

#moving averageと実際の関数の差からマスク位置を決定
def delta_m_ave_self_fft(gx,yf,side):
    rolling_yf = bn.move_mean(np.abs(yf),window=25)
    nan_roll = 0
    for nan_roll in range(0,500):#(0,292)~(0,583)
        rolling_yf[12+nan_roll] = rolling_yf[24+nan_roll]
        
    delta_rangel = 0.09
    fity_selfFFT_std = np.zeros(304-math.floor(304*delta_rangel*2))
    fity_selfFFT = np.zeros(608)
    msk_range = np.zeros(608)
    i = 0
    
    for y in range(0,608):
        if(gx[y]>delta_rangel and gx[y]<=0.5):
            fity_selfFFT[y] = np.abs(yf[y]) - rolling_yf[y]
            fity_selfFFT_std[i] = np.abs(yf[y]) - rolling_yf[y]
            i += 1

    std = statistics.stdev(fity_selfFFT_std)

    for y in range(0,608):
        if fity_selfFFT[y] >= 3*std:
            msk_range[y] = 0.1
    
    #plt.xlim(-0.01,0.5)
    #plt.ylim(-0.01,0.1)
    #plt.plot(gx, np.abs(yf), label='power spectrum')
    #plt.plot(gx, rolling_yf, label='moving average') 
    #plt.title('power spectrum and moving average')
    #plt.title('mask range')
    #plt.xlabel('frequency')
    #plt.ylabel('power')
    #plt.plot(gx,msk_range)
    #plt.legend()
    #plt.show()
    return(msk_range,std,rolling_yf)

#パワースペクトルと移動平均からマスクの範囲を決定
def delta_PS_move_ave(gx,yf,rolling_yf,base_std):
    delta_rangel = 0.09
    fity_selfFFT_std = np.zeros(304-math.floor(304*delta_rangel*2))
    fity_selfFFT = np.zeros(608)
    msk_range = np.zeros(608)

    for y in range(0,608):
        if(gx[y]>delta_rangel and gx[y]<=0.5):
            fity_selfFFT[y] = np.abs(yf[y]) - rolling_yf[y]

    for y in range(0,608):
        if fity_selfFFT[y] >= 5*base_std:
            msk_range[y] = 0.1

    #plt.xlim(-0.01,0.5)
    #plt.ylim(-0.01,0.1)
    #plt.plot(gx, np.abs(yf), label='power spectrum')
    #plt.plot(gx,rolling_yf,label='moving average') 
    #plt.title('power spectrum and moving average')
    #plt.title('noise range')
    #plt.xlabel('frequency')
    #plt.ylabel('power')
    #plt.plot(gx,msk_range)
    #plt.legend()
    #plt.show()
            
    return(msk_range)

# フーリエ変換の結果をプロットする
def yf_plot(yf, x1,x2, mode):
    plt.xlim(1e-5,1.0-1e-5)
    #plt.ylim(1,1e6)
    if   mode == "abs":
        for x in range(x1-1, x2):
            plt.title("FFT of raw data")
            plt.ylim(0,50)
            #plt.plot(yf.gx*2, np.abs(yf.data[x-1]) * 10**(x-1))
            #plt.plot(yf.gx*2, np.abs(yf.data[x-1]))
            plt.plot(yf.gx*2, np.abs(yf.data[x-1]))
        if(pltave):
            plt.plot(yf.gx*2, np.abs(yf.ave) * 10**(x2-1))
        #plt.xscale('log')
        #plt.yscale('log')
    elif mode == "real":
        for x in range(x1-1, x2):
            plt.plot(yf.gx*2, yf.data[x-1].real) # + 50*x)
        if(pltave):
            plt.plot(yf.gx*2, yf.ave.real) #  + 50*x2)
        #plt.ylim(0,500)
        plt.ylim(-50,50)        
#        popt,pcov = fit_yfreal(yf.gx, yf.data[x-1].real)
#        plt.plot(yf.gx, funcR(yf.gx, popt[0],popt[1],popt[2]))
#        plt.plot(yf.gx, yf.data[x-1].real-funcR(yf.gx, popt[0],popt[1],popt[2]))
    elif mode == "imag":
        for x in range(x1-1, x2):
            plt.plot(yf.gx*2, yf.data[x-1].imag)# + 40*x)
            plt.ylim(-100,100)
            #plt.ylim(-100,100)
#            popt,pcov = fit_yfimag(yf.gx, yf.data[x-1].imag)
#            plt.plot(yf.gx, yf.data[x-1].imag-funcI(yf.gx, popt[0],popt[1],popt[2],popt[3]))
#            plt.plot(yf.gx, funcI(yf.gx, popt[0],popt[1],popt[2],popt[3]))
            

        if(pltave):
            plt.plot(yf.gx*2, yf.ave.imag + 40*x2)
    else:
        print('No')
    #plt.title('FFT of raw data(masked)')
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.show()
    return()




####  main() ####

args = sys.argv
fitsname = args[1]
filename = fitsname.rstrip('.fits')
filename_RS = fitsname.replace('_RS.fits','')

x1,x2 = (40,40) # サンプルとして表示する x(pixel) の範囲
offy  = 2       # スキップする y(sampling) の範囲

(data,header) = data_from_fits(fitsname) # FITSを読み込む
data0 = copy.deepcopy(data)

# FITSデータを加工する
# (data[y][x]→data[x][y]変換, 開始からoffy回分のsamplingを除く)
ycut = data_flip_xy(data, offy)

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

lag_max = 304

# 自己相関関数の計算 (既存のルーチン)
llist = pd.Series(ldata2)
llist.index = pd.Series(np.ndarray(58*ycut.ny*2))
lss = sm.tsa.stattools.acf(llist, nlags=lag_max)

# (テスト) FFT2種類の方法の比較
#test_fft_compare(ss)
yfl1,gxl1 = test_fft_compare(lss)
N_yfl = len(yfl1)
yfl = np.zeros(N_yfl)
for y in range(0,608):
    yfl[y] = np.abs(yfl1[y])

#moving averageと実際の関数の差からマスク位置を決定
msk_rangel1,stdl1,rolling_yfl1 = delta_m_ave_self_fft(gxl1,yfl1,'left')

#パワースペクトルにマスクをかけてスムージング
yfl2 = rm_noise_PS(yfl1,msk_rangel1)

#パワースペクトルにマスクをかけてスムージング
msk_rangel2,stdl2,rolling_yfl2 = delta_m_ave_self_fft(gxl1,yfl2,'left')
yfl3 = rm_noise_PS(yfl2,msk_rangel2)

#moving averageと実際の関数の差からマスク位置を決定
msk_rangel3,stdl3,rolling_yfl3 = delta_m_ave_self_fft(gxl1,yfl3,'left')

#２行目の自己相関関数
rlist = pd.Series(rdata2)
rlist.index = pd.Series(np.ndarray(58*ycut.ny*2))
rss = sm.tsa.stattools.acf(rlist, nlags=lag_max)

yfr1,gxr1 = test_fft_compare(rss)
N_yfr = len(yfr1)
yfr = np.zeros(N_yfr)
for y in range(0,608):
    yfr[y] = np.abs(yfr1[y])

#moving averageと実際の関数の差からマスク位置を決定
msk_ranger1,stdr1,rolling_yfr1 = delta_m_ave_self_fft(gxr1,yfr1,'right')

#パワースペクトルにマスクをかけてスムージング
yfr2 = rm_noise_PS(yfr1,msk_ranger1)

#パワースペクトルにマスクをかけてスムージング
msk_ranger2,stdr2,rolling_yfr2 = delta_m_ave_self_fft(gxr1,yfr2,'right')
yfr3 = rm_noise_PS(yfr2,msk_ranger2)

#moving averageと実際の関数の差からマスク位置を決定
msk_ranger3,stdr3,rolling_yfr3 = delta_m_ave_self_fft(gxr1,yfr3,'right')

#moving averageと実際の関数の差からマスク位置を決定(正確)
mask_rangel = delta_PS_move_ave(gxl1,yfl,rolling_yfl3,stdl3)
mask_ranger = delta_PS_move_ave(gxr1,yfr,rolling_yfr3,stdr3)

#天体の情報を抜く
ycut,yescape = escape_star(ycut,data, 0,63, 0,304)

#FFTをかける
yf = ycut_fft(ycut, 1,ycut.nx)

#yf_plot(yf, x1,x2, "abs")

#自動で決定した範囲のノイズを除去
rm_noise_left_6data(yf,mask_rangel)
rm_noise_right_6data(yf,mask_ranger)

yf_plot(yf, x1,x2, "abs")

# invers FFTをかける
ycut = yf_ifft(yf, 1,yf.nx)

#天体の情報を元に戻す
ycut = return_star(ycut,yescape)

# データをXY-flipしてFITSに戻す
data_rflip_xy_save(ycut, data, offy, filename)
#data_rflip_xy(ycut, data, offy)
    
# FITSを表示
#fitsdsp_comp(data0, data, -10,250)
#fitsdsp_diff(data0, data, -10,10)

#天体の位置
#000(x1,x2)(y1,y2)
#725(6,40)(70,103)
#142(10,63)(0,25)
#401(30,63)(30,65)
#231(40,63)(155,185)
#727(15,63)(275,304)
#698(6,50)(60,90)
#402(6,30)(90,120)
#092()()
#326()()
#302(6,40)(130,165)
#152(6,55)(260,295)
#274(20,63)(200,230)
#406(40,63)(130,160)

time2 = time.time()
elapsed_time = time2-time1
print(f"経過時間：{elapsed_time}")
