"""
Program: plot.py
Author: Jordan Hoffmann
Date: 30/09/2022
Purpose: Creates relevant plots from running redisperse.py and consequent results from FREDDA

Inputs:
    FRB_list = List of FRB ID numbers
    
Outputs:
    FRB_SN.pdf = SNR plot as a function of DM
    FRB_DM_comparison.pdf = Plot comparing the detected DM from FREDDA and the actual redispersed DM
    FRB_start_pos.pdf = Starting position of the FRB as a function of DM
    FRB_noise_DM = SNR plot for a single DM with a variety of runs to compare the fluctuations
    FRB_offset_DM = SNR plot for a single DM with a variety of different integration offsets
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import sys
from scipy.optimize import minimize
from scipy import signal
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os

plt.rcParams["figure.figsize"] = (8,6)

def main():

    # Parse command line parameters
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument('-f', dest='frb', help='FRB name e.g. 181112', type=str)
    parser.add_argument('-e', '--export_smooth', dest='export_smoothed', default=False, action='store_true', help='Plot the SNR as a function of DM data points and the smoothed curve and export the smoothed curve to file')
    parser.add_argument('-a', '--all_smooth', dest='all_smooth', default=False, action='store_true', help='Plot of all smoothed curves. If this is specified, put the directory containing the smoothed SNR curves instead of the FRB name')
    parser.add_argument('-S', '--SN', dest='SN', default=False, action='store_true', help='Produce plots of SNR as a function of DM. Uses low_max and high_min and requires Dispersed_<frb>/fredda_outputs/extracted_outputs.txt')
    parser.add_argument('--plt_inc', dest='plt_inc', default=False, action='store_true', help='Overplot incoherent results on SNR graph. Automatically makes SN true')
    parser.add_argument('-p', '--pfb', dest='pfb', default=False, action='store_true', help='Compare plots in Dispersed_FRB_nopfb/ and Dispersed_FRB/')
    parser.add_argument('-d', '--plot_dm', dest='plot_DM', default=False, action='store_true', help='Produce plots of the detected DM vs redispersed DM')
    parser.add_argument('-s', '--start_pos', dest='start_pos', default=False, action='store_true', help='Produce plots of the detected start position of the frb. Requires Dispersed_<frb>/job/start_pos.txt')
    parser.add_argument('-n', '--noise', dest='noise', default=False, action='store_true', help='Produce plots of noise testing. Uses dm and requires files in Dispersed_<frb>/noise_<dm>/outputs/extracted_outputs.txt')
    parser.add_argument('-o', '--offset', dest='offset', default=False, action='store_true', help='Produce plots of offset testing. Uses dm and requires files in Dispersed_<frb>/offset_<dm>/outputs/extracted_outputs.txt')
    parser.add_argument('-w', '--width', dest='width', default=False, action='store_true', help='Produce plots of detected boxcar width. Requires files in Dispersed_<frb>/fredda_outputs/extracted_outputs.txt')
    parser.add_argument('-r', '--reverted', dest='reverted', default=False, action='store_true', help='Produce plots of reverted fredda verstions. Uses low_max and high_min and requires Dispersed_<frb>/fredda_outputs/extracted_outputs.txt')
    parser.add_argument('--in_dir', dest='in_dir', default='', help='Directory containing Dispersed_FRB directory')
    parser.add_argument('--out_dir', dest='out_dir', default='', help='Directory to save plots in')
    parser.add_argument('--snr_dir', dest='snr_dir', default='', help='Directory containing SNR curves')
    parser.add_argument('--norm', dest='norm', default=False, action='store_true', help='Produce plot of normalisation testing. Uses dm and requires files in Dispersed_<frb>/normalise_<dm>/extracted_outputs.txt')
    parser.add_argument('--sd', dest='sd', default=False, action='store_true', help='Produce plot of standard deviation testing. Uses dm and requires files in Dispersed_<frb>/sd_<dm>/extracted_outputs.txt')
    parser.add_argument('--dm', dest='DM', default=None, type=float, help='DM to be plotted for noise, offset, normalisation and standard deviation tests')
    parser.add_argument('--low_max', dest='low_max', default=None, type=float, help='When fitting models the low DM regime will be fit from the start of the DM range to low_max. If both low_max and high_min are None the whole range will be fit')
    parser.add_argument('--high_min', dest='high_min', default=None, type=float, help='When fitting models the high DM regime will be fit from high_min to the end of the DM range. If both low_max and high_min are None the whole range will be fit')
    parser.add_argument('--norm_dm', dest='norm_DM', default=500, type=float, help='DM by which to normalise models by. If None, fitting occurs instead')
    parser.add_argument('--increment', dest='increment', default=None, type=float, help='Only use points which are multiples of increment for nice plotting')
    parser.set_defaults(verbose=False, nxy="1,1")

    global values
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if values.all_smooth:
        all_smooth()

    if values.plot_DM:
        plot_DM_comparison()
    if values.export_smoothed:
        export_smoothed()
    if values.SN or values.plt_inc:
        plot_SN()
    if values.pfb:
        plot_pfb()
    if values.reverted:
        plot_reverted()
    if values.start_pos:
        plot_start_pos()
    if values.noise:
        plot_noise_test()
    if values.offset:
        plot_offset_test()
    if values.width:
        plot_boxcar_width()
    if values.norm:
        plot_normalise_test()
    if values.sd:
        plot_sd_test()

#==============================================================================
"""
Function: all_smooth
Date: 01/08/2023
Plot of all smoothed curves

Imports:

Exports:
    FRB_DM_comparison.pdf = Plot comparing the detected DM from FREDDA and the actual redispersed DM

"""
def all_smooth():

    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    ax.set_xlabel(r'Redispersed DM (pc cm$^{-3}$)')
    ax.set_ylabel(r'SNR')

    files = sorted(os.scandir(values.snr_dir), key=lambda e: e.name)
    ax.set_prop_cycle(plt.cycler(linestyle=['-', '-.']) * plt.cycler(color=list('bcgrmyk')))

    for file in files:
        frb = file.name[:6]

        # Read data
        # data = np.genfromtxt("Dispersed_" + frb + "/fredda_outputs/extracted_outputs.txt", skip_header=1)
        frb_data = np.genfromtxt(os.path.join(values.in_dir,"Dispersed_" + frb + "/data/" + frb + "_param.dat"))
        smoothed = np.load(os.path.join(values.snr_dir, file.name))

        # Normalise to detection SNR at detection DM
        dmi_idx = np.argmin(np.abs(smoothed[0,:] - frb_data[1]))
        real_SN = frb_data[5]
        smoothed[1,:] = smoothed[1,:] / smoothed[1,dmi_idx] * real_SN

        dots, = ax.plot(smoothed[0,:], smoothed[1,:], label=frb)
        ax.plot(frb_data[1], real_SN, 'x', ms=7, mew=2, c=dots.get_color())
    
    ax.axhline(9, c='k', ls='--', label="Threshold (SNR=9)")
    ax.legend(ncol=2)
    fig.savefig(os.path.join(values.out_dir,"all_smoothed.pdf"), bbox_inches='tight')

#==============================================================================
"""
Function: export_smoothed
Date: 30/09/2022
Create a plot of the SNR as a function of DM.

Imports:
    break_dm = DM at which Cordes model starts
    real_SN = Detected SNR
    increment = Only take data points which are multiples of increment (for nice plotting)

Exports:
    <frb>_SN.pdf

"""
def export_smoothed():

    # Read data
    if int(values.frb) < 190830:
        file = "Dispersed_" + values.frb + "/fredda_reverted/fredda_outputs_190101/extracted_outputs.txt"
    elif int(values.frb) < 200406:
        file = "Dispersed_" + values.frb + "/fredda_reverted/fredda_outputs_200228/extracted_outputs.txt"
        break_dm = 1e6
    else:
        file = "Dispersed_" + values.frb + "/fredda_outputs/extracted_outputs.txt"
    data = np.genfromtxt(file, skip_header=1)
    frb_data = np.genfromtxt("Outputs/Dispersed_" + values.frb + "/data/" + values.frb + "_param.dat")

    if (values.increment == None):
        pltx = data[:,0]
        plty = data[:,1]
    else:
        pltlen = int(np.ceil((data[-1,0] - data[0,0])/values.increment)) + 1
        pltx = np.zeros(pltlen)
        plty = np.zeros(pltlen)
        j=0
        for i in range(len(data[:,0])):
            if (data[i,0] % values.increment == 0):
                pltx[j] = data[i,0]
                plty[j] = data[i,1]
                j = j+1

    # Normalise to detection SNR at detection DM
    # dmi_idx = np.argwhere(data[:,0] >= frb_data[1])[0][0]
    # real_SN = frb_data[5]
    # plty = plty / data[dmi_idx,1] * real_SN

    # Determine where FREDDA stops searching as the max dm
    f_mid = frb_data[2]
    f_low = f_mid - 167.5
    f_high = f_mid + 167.5
    int_t = frb_data[3]
    D = 4.148808e-3 # Dispersion constant in GHz^2 pc^-1 cm^3 s

    max_dt = int_t * 4096 * 1e-3   # FREDDA searches up to 4096 time bins
    max_dm = max_dt / (D * ((f_low/1e3)**(-2) - (f_high/1e3)**(-2)))

    x = np.linspace(pltx[0], max_dm, int(1e3))

    # Default break_dm to 2000
    if break_dm == None:
        break_dm = 2e3

    # Determine index where fits switch
    if break_dm <= 0:
        y = cordes(1, x, frb_data)

        # bandwidth = (frb_data[3] - frb_data[2])
        # vc = (frb_data[2] + bandwidth/2) / 1e3

        # print(x)
        # print((8.3*x*1/1e3/vc**3), flush=True)
        # y = fit_cordes(pltx, plty, None, None, x, frb_data)
        # norm_dm = np.argwhere(x >= frb_data[1])[0][0]
        # y = y / y[norm_dm] * real_SN

    elif break_dm <= max_dm:
        break_idx = np.argwhere(x >= break_dm)[0][0]
        break_idx_data = np.argwhere(pltx >= break_dm)[0][0]

        buf = np.clip(buf, 0, max_dm - break_dm)
        overlap_idx = np.argwhere(x >= break_dm+buf)[0][0]

        y = np.zeros(len(x), dtype=float)
        y_2 = np.zeros(len(x), dtype=float)

        # Cordes curve for high DM regime
        y[break_idx:], k = fit_cordes(pltx, plty, break_idx_data, len(pltx), x[break_idx:], frb_data)

        # Numerical fit for low DM regime
        y_2[:overlap_idx] = smooth(pltx, plty, x[:overlap_idx])

        weights = np.zeros(len(x), dtype=float)
        weights[break_idx:overlap_idx] = np.linspace(0,1,(overlap_idx - break_idx))
        weights[overlap_idx:] = 1

        y = y*weights + y_2 * (1-weights)
        y = y/k
        plty = plty/k

    else:
        y = smooth(pltx, plty, x)
        _, k = fit_cordes(pltx[-10:], plty[-10:], None, None, pltx[-10:], frb_data)
        
        y = y/k
        plty = plty/k

    plt.figure()
    plt.xlabel(r"Redispersed DM (pc/cm$^3$)", fontsize='x-large') 
    plt.ylabel(r"$\eta$", fontsize='x-large')
    plt.plot(pltx, plty, 'b.', label='Data')
    plt.plot(x, y, 'k-')
    # plt.plot(data[dmi_idx,0], real_SN, 'r.')
    plt.ylim(bottom=0, top=np.max(plty)*1.2)
    plt.savefig("plots/" + values.frb + "_smoothed.pdf", bbox_inches='tight')
    plt.close()

    np.save("SNR_curves/"+values.frb+".npy", np.stack([x,y]))

#==============================================================================
"""
Function: plot_SN
Date: 31/07/2023
Create a plot of the SNR as a function of DM fitted with various models.

Imports:
    values = Command line parameters

Exports:
    plots/frb_SN.pdf
"""
def plot_SN():

    frb_data = np.genfromtxt(os.path.join(values.in_dir,"Dispersed_" + values.frb + "/data/" + values.frb + "_param.dat"))

    # Initialise plotting
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    ax.set_xlabel(r"Redispersed DM (pc/cm$^3$)", fontsize='x-large') 
    ax.set_ylabel(r"SNR", fontsize='x-large')
    # ax.set_ylim(top=85, bottom=15)

    ax.set_prop_cycle(plt.cycler(linestyle=['-', '--']) * plt.cycler(color=list('bcgmy')))

    # Plot our results
    raw, smoothed = plot_data(os.path.join(values.in_dir,"Dispersed_" + values.frb + "/fredda_outputs/extracted_outputs.txt"), frb_data, ax, values, label='This work', zorder=10, skip_header=1)
    # raw, smoothed = plot_data("Dispersed_" + frb + "_fix_fch1/fredda_outputs/extracted_outputs.txt", frb_data, ax, values, label='fixed fch1', zorder=10, skip_header=1)

    # Plot Harry's curves (pulse injection)
    plot_pulse_injection(frb_data, ax, smoothed[0], smoothed[1])

    # # Plot old FREDDA versions
    # if int(frb) < 190830:
    #     plot_data("Dispersed_" + frb + "/fredda_reverted/fredda_outputs_190101/extracted_outputs.txt", frb_data, ax, values, label='Version 1', skip_header=1)
    # elif int(frb) < 200806:
    #     plot_data("Dispersed_" + frb + "/fredda_reverted/fredda_outputs_200228/extracted_outputs.txt", frb_data, ax, values, label='Version 2', skip_header=1)

    # # Plot incoherent results
    # if values.plt_inc:
    #     inc_filename = "Dispersed_" + frb + "/incoherent_dedispersion/" + frb + "_incoherent_SNR.txt"
    #     plot_data(inc_filename, frb_data, ax, values, label='This work')

    # Plot models
    plot_models(ax, raw[0], raw[1], smoothed[0], frb_data)

    # Save plot
    ax.legend()
    fig.savefig(os.path.join(values.out_dir,values.frb+"_SN.pdf", bbox_inches='tight'))

#==============================================================================
"""
Function: plot_data
Date: 31/07/2023
Plot data with the smoothed version and fitted models

Imports:
    filename = Extracted FREDDA outputs filename
    frb_data = Data about FRB from FRB_param.dat
    ax = Axis for plotting
    label = Label on plot
    values = 
    increment = Increment to use for data points

Exports:
    raw = tuple of pltx and plty (raw data)
    smoothed = tuple of x_full and y_smoothed (smoothed data)
"""
def plot_data(filename, frb_data, ax, values, label='', zorder=1, skip_header=0):
    data = np.genfromtxt(filename, skip_header=skip_header)

    # Check for increment
    if (values.increment == None):
        pltx = data[:,0]
        plty = data[:,1]
    else:
        pltlen = int(np.ceil((data[-1,0] - data[0,0])/values.increment)) + 1
        pltx = np.zeros(pltlen)
        plty = np.zeros(pltlen)
        j=0
        for i in range(len(data[:,0])):
            if (data[i,0] % values.increment == 0):
                pltx[j] = data[i,0]
                plty[j] = data[i,1]
                j = j+1

    # Get smoothed version
    x_full = np.linspace(pltx[0], pltx[-1], int(1e4))
    y_smoothed = smooth(pltx, plty, x_full)

    # Normalisation
    if norm_DM==-1:
        DMi = frb_data[1]
        norm_DM = DMi
        DMi_idx = np.argmin(np.abs(x_full - DMi))

        plty = plty / y_smoothed[DMi_idx] * frb_data[5]
        y_smoothed = y_smoothed / y_smoothed[DMi_idx] * frb_data[5]

        ax.plot(x_full[DMi_idx], y_smoothed[DMi_idx], 'mx', zorder=zorder+2)

    dots, = ax.plot(pltx, plty, '.', label=label, zorder=zorder)

    # Plot smoothed version
    ax.plot(x_full, y_smoothed, '-', c=dots.get_color(), zorder=zorder+1)

    return [pltx, plty], [x_full, y_smoothed]

#==============================================================================
"""
Function: plot_models
Date: 31/07/2023
Plot pulse injection

Imports:
    frb_data = Data about FRB from FRB_param.dat
    ax = Axis for plotting
    
Exports:
    ax = Axis with plots
    x_full = DM array for smoothed function
    y_smoothed = Smoothed sensitivity function
"""
def plot_models(ax, pltx, plty, x_full, frb_data):

    low_max_dm = values.low_max
    high_min_dm = values.high_min

    # Get range to fit 
    # If neither range is set then fit the whole range
    if low_max_dm==None and high_min_dm==None:
        low_max_idx = None  # Make the low DM range the entire range
        low_max_dm = -1     # Make sure the low DM range is fitted
    else:
        if low_max_dm != None:
            # Get the corresponding index of where the highest point in the low region is
            low_max_idx = np.argwhere(pltx < low_max_dm)
            if len(low_max_idx[:,0]) > 0:
                low_max_idx = low_max_idx[-1][0]
            else:
                print("No points in low DM regime")
                low_max_dm = None   # Don't plot low DM regime
        
        if high_min_dm != None:
            # Get the corresponding index ofwhere the lowest point in the high region is
            high_min_idx = np.argwhere(pltx > high_min_dm)
            if len(high_min_idx[:,0]) > 0:
                high_min_idx = high_min_idx[0][0]
            else:
                print("No points in high DM regime")
                high_min_dm = None  # Don't plot high DM regime

    # Do low range if it is given (or do full range if neither is given)
    if low_max_dm != None:
        # Vectors to plot models
        if low_max_idx == None:
            x_low = np.linspace(pltx[0], pltx[-1], int(1e4))
        else:
            x_low = np.linspace(pltx[0], pltx[low_max_idx], int(1e4))

        # Calculating Sammons model
        y_low_sammons, k = fit_sammons(pltx, plty, 0, low_max_idx, x_low, frb_data)
        y_low_sammons_full = sammons(k, x_full, frb_data)

        # Calculating Cordes model
        y_low_cordes, k = fit_cordes(pltx, plty, 0, low_max_idx, x_low, frb_data)
        y_low_cordes_full = cordes(k, x_full, frb_data)

        # Add models to plot
        ax.plot(x_low,y_low_sammons,'r-', label='Arcus et al. (2021)')
        ax.plot(x_full, y_low_sammons_full, 'r', linestyle='dashed')

        ax.plot(x_low,y_low_cordes,'k-', label='Cordes & McLaughlin (2003)')
        ax.plot(x_full, y_low_cordes_full, 'k', linestyle='dashed')

    # Same for high range
    if high_min_dm != None:
        # Vectors to plot models
        x_high = np.linspace(pltx[high_min_idx], pltx[-1], int(1e4))

        # Calculating Sammons model
        y_high_sammons, k = fit_sammons(pltx, plty, high_min_idx, len(pltx), x_high, frb_data)
        y_high_sammons_full = sammons(k, x_full, frb_data)

        # Calculating Cordes model
        y_high_cordes, k = fit_cordes(pltx, plty, high_min_idx, len(pltx), x_high, frb_data)
        y_high_cordes_full = cordes(k, x_full,frb_data)

        # Add models to plot
        ax.plot(x_high,y_high_sammons,'r-', label='Arcus et al. (2021)')
        ax.plot(x_full, y_high_sammons_full, 'r', linestyle='dashed') 

        ax.plot(x_high,y_high_cordes,'k-', label='Cordes & McLaughlin (2003)')
        ax.plot(x_full, y_high_cordes_full, 'k', linestyle='dashed')

#==============================================================================
"""
Function: plot_pulse_injection
Date: 31/07/2023
Plot pulse injection

Imports:
    frb_data = Data about FRB from FRB_param.dat
    ax = Axis for plotting
    
Exports:
    ax = Axis with plots
    x_full = DM array for smoothed function
    y_smoothed = Smoothed sensitivity function
"""
def plot_pulse_injection(frb_data, ax, x_full, y_smoothed):
    # label = 'Qiu et al. (2023)'
    # label = 'Qiu et al. (2023) FWHM ' + str(round(w, 1)) + 'ms'

    # For plotting multiple injections
    dirs = sorted(os.scandir(os.path.join(values.in_dir, "Dispersed_" + values.frb + "/")), key=lambda e: e.name)
    for dir in dirs:
        if dir.name[:15] == "pulse_injection":
            if os.path.exists(os.path.join(values.in_dir, "Dispersed_"+values.frb+"/" + dir.name + "/extracted_outputs.txt")):
                harry_data, dm_array, w = get_harry_data(os.path.join(values.in_dir, "Dispersed_"+values.frb+"/" + dir.name + "/extracted_outputs.txt"), frb_data[4])
            
                dm_full = np.linspace(dm_array[0], dm_array[-1], int(1e4))

                harry_smoothed = smooth(dm_array, harry_data, dm_full)

                # Normalisation
                if norm_DM == -1:
                    norm_idx = -1
                else:
                    norm_idx = np.argmin(np.abs(dm_full - norm_DM))

                norm_DM_real = dm_full[norm_idx]
                harry_data = harry_data / harry_smoothed[norm_idx] * y_smoothed[np.argmin(np.abs(x_full - norm_DM_real))]

                harry_smoothed = harry_smoothed / harry_smoothed[norm_idx] * y_smoothed[np.argmin(np.abs(x_full - norm_DM_real))]

                # Plotting
                dots, = ax.plot(dm_array, harry_data, '.', label='Qiu et al. (2023)')
                # label = 'Qiu et al. (2023) half band'
                ax.plot(dm_full, harry_smoothed, '-', c=dots.get_color())

#==============================================================================

"""
Function: smooth
Date: 20/03/2023
Create a smoothed curve of the given data

Imports:
    datax = X coordinates of the data
    datay = Y coordinates of the data
    x = X coordinates to interpolate onto
    win_len = Window length for fitting

Exports:
    y_smoothed_interp = Smoothed array interpolated to x_full
"""

def smooth(datax, datay, x, lam=0.1, win_len=None, fun='Savgol'):
    if fun=='Savgol':
        if win_len==None:
            win_len = 10
            # win_len = int(len(datax) / 5)

        if win_len >= datax.shape[0]:
            win_len = datax.shape[0] - 2

        if win_len % 2 == 0:
            win_len = win_len + 1
        

        if win_len < 5:
            win_len = 5

        if datay[0] > 1.2* datay[1]:
            y_smoothed = signal.savgol_filter(datay[1:], window_length=win_len, polyorder=4)
            y_smoothed = np.concatenate(([datay[0]], y_smoothed))
        else:
            y_smoothed = signal.savgol_filter(datay, window_length=win_len, polyorder=4)

    elif fun=='Euler-Lagrange':
        # Gradient descent
        # y_smoothed = np.copy(datay)

        # for i in range(100):
        #     y_smoothed[2:-2] -= (0.05 / lam) * (y_smoothed - datay)[2:-2] + np.convolve(y_smoothed, [1, -4, 6, -4, 1])[4:-4]

        # Linear algebra solving
        A = np.diag(np.ones(len(datay))*(1-6*lam))
        A += np.diag(np.ones(len(datay)-1)*4*lam, -1)
        A += np.diag(np.ones(len(datay)-1)*4*lam, 1)
        A += np.diag(-np.ones(len(datay)-2)*lam, -2)
        A += np.diag(-np.ones(len(datay)-2)*lam, 2)
        A[:2,:] = 0
        A[0,0] = 1
        A[1,1] = 1

        A[-2:,:] = 0
        A[-1,-1] = 1
        A[-2,-2] = 1

        print(A)

        y_smoothed = np.matmul(A, datay.T)

    y_smoothed_interp = np.interp(x, datax, y_smoothed)

    return y_smoothed_interp

#==============================================================================
"""
Function: get_harry_data
Date: 30/03/2023
Get a vector of Harry's data

Imports:
    
Exports:
    data = array of SNR values
    dm_array = array of DM values
    w = width of pulse used
"""
def get_harry_data(file, width, TOL = 0.01):
    # harry_data_hi = np.load("Data/fredda_hifreq_single.npy")
    # harry_data_lo = np.load("Data/fredda_lofreq_single.npy")
    
    # weights_hi = np.load("Data/hifreq_single_response.npy")
    # weights_lo = np.load("Data/lofreq_single_response.npy")
    
    # harry_data_hi = harry_data_hi * weights_hi.T
    # harry_data_lo = harry_data_lo * weights_lo.T

    txt = np.genfromtxt(file, skip_header=1)
    
    dm_list = []

    w_array = txt[:,0]
    w = w_array[np.argmin(np.abs(w_array - width/2))]
    
    data = []

    i=np.argwhere(txt[:,0] == w)[0]
    while i<txt.shape[0] and txt[i,0] == w:
        curr_dm = txt[i,1]
        total = 0
        num = 0
        j = 0
        while i+j<txt.shape[0] and txt[i+j,1] == curr_dm and txt[i+j,0] == w:
            if (np.abs(curr_dm - txt[i+j,7]) < curr_dm * TOL + 10):
                total += txt[i+j,2]
                num += 1
            j += 1

        if num != 0:
            data.append(total / num)
            dm_list.append(curr_dm)
        
        if j != 0:
            i += j
        else:
            i += 1

    dm_array = np.array(dm_list)
    data = np.array(data)

    return data.T[0], dm_array.T[0], w*2.4

#==============================================================================
"""
Function: fit_cordes
Date: 20/03/2023
Fit the Cordes model in the given region

Imports:
    datax = X coordinates of the data
    datay = Y coordinates of the data
    min_idx = Lower index of fitting
    max_idx = Upper index of fitting
    x = X coordinates to interpolate onto
    frb_data = FRB statistics from param file

Exports:
    y_smoothed_interp = Smoothed array interpolated to x_full
    k = Normalisation constant
"""
def fit_cordes(datax, datay, min_idx, max_idx, x, frb_data, fit_width=False):

    # Do fitting
    if (fit_width):
        k = minimize(cordes_nll, (1,1), args=(datax[min_idx:max_idx], datay[min_idx:max_idx], frb_data)).x[0:2]
        print("Fitted (eta, w):" + str(k))
    else:
        k = minimize(cordes_nll, 1, args=(datax[min_idx:max_idx], datay[min_idx:max_idx], frb_data)).x[0]
    
    y = cordes(k, x, frb_data)

    return y, k

#==============================================================================
"""
Function: fit_sammons
Date: 20/03/2023
Fit the Sammons model in the given region

Imports:
    datax = X coordinates of the data
    datay = Y coordinates of the data
    min_idx = Lower index of fitting
    max_idx = Upper index of fitting
    x = X coordinates to interpolate onto
    frb_data = FRB statistics from param file

Exports:
    y_smoothed_interp = Smoothed array interpolated to x_full
"""

def fit_sammons(datax, datay, min_idx, max_idx, x, frb_data, fit_width=False):

    # Do fitting
    if (fit_width):
        k = minimize(sammons_nll, (1,1), args=(datax[min_idx:max_idx], datay[min_idx:max_idx], frb_data)).x[0:2]
        print("Fitted (eta, w):" + str(k))
    else:
        k = minimize(sammons_nll, 1, args=(datax[min_idx:max_idx], datay[min_idx:max_idx], frb_data)).x[0]
    
    y = sammons(k, x, frb_data)

    return y, k

#==============================================================================

def cordes(k, DM, frb_data):
    # Model parameters evaluated from FRB data
    vc = frb_data[2] / 1e3
    int_t = frb_data[3]

    if k.size==2:
        w = k[1]
        eta = k[0]
    else:
        w = frb_data[4]
        eta = k

    return eta/np.sqrt(np.sqrt((8.3*DM*1/1e3/vc**3)**2 + (int_t)**2 + w**2))

#==============================================================================

def cordes_nll(k_true, DM, SN, frb_data, sigma=1):
    
    return 0.5 * np.sum( (cordes(k_true, DM, frb_data) - SN)**2 / sigma**2 )

#==============================================================================

def sammons(k, DM, frb_data):
    # Model parameters evaluated from FRB data
    #k, c1, c2 = theta
    c1 = 0.94
    c2 = 0.37
    vc = frb_data[2] / 1e3
    int_t = frb_data[3]
        
    if k.size==2:
        w = k[1]
        eta = k[0]
    else:
        w = frb_data[4]
        eta = k

    return eta/np.sqrt(c1*(8.3*DM*1/1e3/vc**3) + c2*(int_t) + w)

#==============================================================================

def sammons_nll(k_true, DM, SN, frb_data, sigma=1):
    
    return 0.5 * np.sum( (sammons(k_true, DM, frb_data) - SN)**2 / sigma**2 )

#==============================================================================

def combined(k1, alpha, DM, frb_data):
    vc = frb_data[2] / 1e3
    int_t = frb_data[3]
    w = frb_data[4]

    #return alpha*k1/np.sqrt(np.sqrt((8.3*DM*1/1e3/vc**3/w)**2 + (int_t/w)**2 + 1)) + (1-alpha)*k2/np.sqrt(0.94*(8.3*DM*1/1e3/vc**3/w) + (int_t/w)*0.37 + 1)
    return k1/np.sqrt(alpha*np.sqrt((8.3*DM*1/1e3/vc**3/w)**2 + (int_t/w)**2 + 1) + (1-alpha)*(0.94*(8.3*DM*1/1e3/vc**3/w) + (int_t/w)*0.37 + 1))

#==============================================================================

def combined_nll(theta, DM, SN, frb_data, sigma=1):

    k1_true, alpha = theta
    return 0.5 * np.sum( (combined(k1_true, alpha, DM, frb_data) - SN)**2 / sigma**2 )

#==============================================================================
"""
Function: plot_DM_comparison
Date: 30/09/2022
Create a plot comparing the detected DM from FREDDA with the actual
redispersed DM.

Imports:

Exports:
    FRB_DM_comparison.pdf = Plot comparing the detected DM from FREDDA and the actual redispersed DM

"""
def plot_DM_comparison():
    
    plt.figure()
    plt.xlabel(r"Redispersed DM (pc/cm$^3$)", fontsize='x-large') 
    plt.ylabel(r"DM deviation (pc/cm$^3$)", fontsize='x-large')

    # Read data
    dirs = sorted(os.scandir(os.path.join(values.in_dir, "Dispersed_" + values.frb + "/fredda_reverted")), key=lambda e: e.name)
    for dir in dirs:
        if dir.name[:15] == "fredda_outputs_":
            if os.path.exists("Dispersed_" + values.frb + "/fredda_reverted/" + dir.name + "/extracted_outputs.txt"):
                data = np.genfromtxt("Dispersed_" + values.frb + "/fredda_reverted/" + dir.name + "/extracted_outputs.txt", skip_header=1)
                if len(data.shape) != 1:
                    if dir.name[15:] == '190101':
                        label = 'Version 1'
                    elif dir.name[15:] == '200228':
                        label = 'Version 2'
                    else:
                        label = dir.name[15:]
                    plt.plot(data[:,0], data[:,6] - data[:,0], '.', label=label)
            else:
                print("No file output.txt in " + dir.name)

    data = np.genfromtxt("Dispersed_" + values.frb + "/fredda_outputs/extracted_outputs.txt", skip_header=1)
    plt.plot(data[:,0], data[:,6] - data[:,0], '.', label="Version 3")

    plt.legend()
    plt.savefig(os.path.join(values.out_dir,values.frb+"_DM_comparison.pdf"), bbox_inches='tight')
    plt.close()

#==============================================================================
"""
Function: plot_pfb
Date: 30/09/2022
Create a plot of the SNR as a function of DM for a variety of FREDDA versions.

Exports:
    FRB_pfb.pdf

"""
def plot_pfb():

    plt.figure()
    plt.xlabel(r"Redispersed DM (pc/cm$^3$)", fontsize='x-large')
    plt.ylabel(r"SNR", fontsize='x-large')

    data = np.genfromtxt("Dispersed_" + values.frb + "_nopfb/fredda_outputs/extracted_outputs.txt", skip_header=1)
    data_pfb = np.genfromtxt("Dispersed_" + values.frb + "/fredda_outputs/extracted_outputs.txt", skip_header=1)
    plt.plot(data[:,0], data[:,1], '.', label="No pfb")
    plt.plot(data_pfb[:,0], data_pfb[:,1], '.', label="pfb")
    plt.legend()
    plt.savefig("plots/"+values.frb+"_pfb.pdf", bbox_inches='tight')
    plt.close()

#==============================================================================
"""
Function: plot_reverted
Date: 30/09/2022
Create a plot of the SNR as a function of DM for a variety of FREDDA versions.

Exports:
    FRB_reverted.pdf

"""
def plot_reverted():

    plt.figure()
    plt.xlabel(r"Redispersed DM (pc/cm$^3$)", fontsize='x-large')
    plt.ylabel(r"SNR", fontsize='x-large')

    # Read data
    dirs = sorted(os.scandir("Dispersed_" + values.frb + "/fredda_reverted"), key=lambda e: e.name)
    for dir in dirs:
        if dir.name[:15] == "fredda_outputs_":
            if os.path.exists("Dispersed_" + values.frb + "/fredda_reverted/" + dir.name + "/extracted_outputs.txt"):
                data = np.genfromtxt("Dispersed_" + values.frb + "/fredda_reverted/" + dir.name + "/extracted_outputs.txt", skip_header=1)
                if len(data.shape) != 1:
                    if dir.name[15:] == '190101':
                        label = 'Version 1'
                    elif dir.name[15:] == '200228':
                        label = 'Version 2'
                    else:
                        label = dir.name[15:]
                    plt.plot(data[:,0], data[:,1], '.', label=label)
            else:
                print("No file output.txt in " + dir.name)

    data = np.genfromtxt("Dispersed_" + values.frb + "/fredda_outputs/extracted_outputs.txt", skip_header=1)
    plt.plot(data[:,0], data[:,1], '.', label="Version 3")

    plt.legend()
    plt.savefig("plots/"+values.frb+"_reverted.pdf", bbox_inches='tight')
    plt.close()

#==============================================================================
"""
Function: plot_start_pos
Date: 30/09/2022
Create a plot of the starting position of the FRB as a function of DM.

Imports:

Exports:
    FRB_start_pos.pdf = Starting position of the FRB as a function of DM.

"""

def plot_start_pos():
    
    # Read data
    data = np.genfromtxt("Dispersed_" + values.frb + "/job/start_pos.txt", skip_header=1)
                
    # Plotting
    plt.figure()
    plt.clf
    plt.xlabel(r"DM", fontsize='x-large')
    plt.ylabel(r"Start pos", fontsize='x-large')
    plt.plot(data[:,0], data[:,1], '.')
    plt.savefig("plots/"+values.frb+"_start_pos.pdf", bbox_inches='tight')
    plt.close()

#==============================================================================
"""
Function: plot_noise_test
Date: 30/09/2022
Create a plot of the SNR for a given DM to observe variations due to noise.

Imports:
    DM = Dispersion measure of the trial

Exports:
    FRB_noise_DM = SNR plot for a single DM with a variety of runs to compare the fluctuations

"""

def plot_noise_test(DM):

    # Read data
    if DM == None:
        dirs = os.scandir("Dispersed_" + values.frb)
        for dir in dirs:
            if dir.name[:6] == "noise_":
                DM = dir.name[6:8]
                break
        dirs.close()
    
    if not os.path.exists("Dispersed_" + values.frb + "/noise_" + str(DM)):
        print("No directory found at Dispersed_" + values.frb + "/noise_" + str(DM))
        return
    elif not os.path.exists("Dispersed_" + values.frb + "/noise_" + str(DM) + "/outputs/extracted_outputs.txt"):
        print("No file found at Dispersed_" + values.frb + "/noise_" + str(DM) + "/outputs/extracted_outputs.txt")
        return
    else:
        data = np.genfromtxt("Dispersed_" + values.frb + "/noise_" + str(DM) + "/outputs/extracted_outputs.txt", skip_header=1)
    
    mean = np.mean(data[:,1])
    sd = np.std(data[:,1])

    print("Noise testing: mean = " + str(mean) + ", std = " + str(sd))

    # Plotting   
    plt.figure()
    plt.clf
    plt.xlabel(r"Trial", fontsize='x-large') 
    plt.ylabel(r"SNR", fontsize='x-large')
    plt.plot(np.arange(len(data[:,1])), data[:,1], '.')
    plt.savefig("plots/"+values.frb+"_noise_" + str(DM) + ".pdf", bbox_inches='tight')
    plt.close()

#==============================================================================
"""
Function: plot_offset_SN
Date: 30/09/2022
Create a plot of the SNR for varying offsets for a given DM.

Imports:
    DM = Dispersion measure of the trial

Exports:
    FRB_offset_DM = SNR plot for a single DM with a variety of different integration offsets

"""

def plot_offset_test(DM):

    # Read data
    if DM == None:
        dirs = os.scandir("Dispersed_" + values.frb)
        for dir in dirs:
            if dir.name[:8] == "offset_":
                DM = dir.name[8:10]
                break
        dirs.close()
    
    if not os.path.exists("Dispersed_" + values.frb + "/offset_" + str(DM)):
        print("No directory found at Dispersed_" + values.frb + "/offset_" + str(DM))
        return
    elif not os.path.exists("Dispersed_" + values.frb + "/offset_" + str(DM) + "/outputs/extracted_outputs.txt"):
        print("No file found at Dispersed_" + values.frb + "/offset_" + str(DM) + "/outputs/extracted_outputs.txt")
        return
    else:
        data = np.genfromtxt("Dispersed_" + values.frb + "/offset_" + str(DM) + "/outputs/extracted_outputs.txt", skip_header=1)
 
    mean = np.mean(data[:,2])
    sd = np.std(data[:,2])

    print("Offset testing: mean = " + str(mean) + ", std = " + str(sd))

   
    # Plotting
    plt.figure()
    plt.clf
    plt.xlabel(r"Offset (% of t$_{\mathrm{int}}$)", fontsize='x-large') 
    plt.ylabel(r"SNR", fontsize='x-large')
    plt.plot(data[:,1], data[:,2], '.')
    plt.savefig("plots/"+values.frb+"_offset_" + str(DM) + ".pdf", bbox_inches='tight')
    plt.close()

#==============================================================================
"""
Function: plot_boxcar_width
Date: 30/09/2022
Create a plot of the detected boxcar width

Imports:

Exports:
    FRB_boxcar_widths.pdf = Plot of the detected boxcar widths

"""
def plot_boxcar_width():
    
    plt.figure()
    plt.xlabel(r"Redispersed DM (pc/cm$^3$)", fontsize='x-large') 
    plt.ylabel(r"Boxcar width (number of samples)", fontsize='x-large')

    # Read data
    dirs = sorted(os.scandir("Dispersed_" + values.frb + "/fredda_reverted"), key=lambda e: e.name)
    for dir in dirs:
        if dir.name[:15] == "fredda_outputs_":
            if os.path.exists("Dispersed_" + values.frb + "/fredda_reverted/" + dir.name + "/extracted_outputs.txt"):
                data = np.genfromtxt("Dispersed_" + values.frb + "/fredda_reverted/" + dir.name + "/extracted_outputs.txt", skip_header=1)
                if len(data.shape) != 1:
                    if dir.name[15:] == '190101':
                        label = 'Version 1'
                    elif dir.name[15:] == '200228':
                        label = 'Version 2'
                    else:
                        label = dir.name[15:]
                    plt.plot(data[:,0], data[:,4], '.', label=label)
            else:
                print("No file output.txt in " + dir.name)

    data = np.genfromtxt("Dispersed_" + values.frb + "/fredda_outputs/extracted_outputs.txt", skip_header=1)
    plt.plot(data[:,0], data[:,4], '.', label="Version 3")

    plt.legend()
    plt.savefig("plots/"+values.frb+"_boxcar_widths.pdf", bbox_inches='tight')
    plt.close()

#==============================================================================
"""
Function: plot_normalise_test
Date: 10/10/2022
Create a plot of the SNR for using normalisation and not.

Imports:
    DM = Dispersion measure of the trial

Exports:
    FRB_normalise_DM = SNR plot

"""

def plot_normalise_test(DM):

    # Read data
    if DM == None:
        dirs = os.scandir("Dispersed_" + values.frb)
        for dir in dirs:
            if dir.name[:10] == "normalise_":
                DM = dir.name[10:12]
                break
        dirs.close()
    
    if not os.path.exists("Dispersed_" + values.frb + "/normalise_" + str(DM)):
        print("No directory found at Dispersed_" + values.frb + "/normalise_" + str(DM))
        return
    elif not os.path.exists("Dispersed_" + values.frb + "/normalise_" + str(DM) + "/extracted_outputs.txt"):
        print("No file found at Dispersed_" + values.frb + "/normalise_" + str(DM) + "/extracted_outputs.txt")
        return
    else:
        data = np.genfromtxt("Dispersed_" + values.frb + "/normalise_" + str(DM) + "/extracted_outputs.txt", skip_header=1)

    break1 = int(len(data)/2)

    # Plotting
    plt.figure()
    plt.clf
    plt.xlabel(r"Trial", fontsize='x-large') 
    plt.ylabel(r"SNR", fontsize='x-large')
    plt.plot(np.arange(break1), data[:break1,1], '.', label='Whole grid')
    plt.plot(np.arange(break1), data[break1:,1], '.', label='Channel-wise')
    plt.legend()
    plt.savefig("plots/"+values.frb+"_normalise_" + str(DM) + ".pdf", bbox_inches='tight')
    plt.close()

#==============================================================================
"""
Function: plot_sd_test
Date: 28/02/2023
Create a plot of the SNR for using various sd values.

Imports:
    DM = Dispersion measure of the trial

Exports:
    FRB_sd_DM.pdf = SNR plot

"""

def plot_sd_test(DM):
 
    # Read data
    if DM == None:
        dirs = os.scandir("Dispersed_" + values.frb)
        for dir in dirs:
            if dir.name[:3] == "sd_":
                DM = dir.name[3:5]
                break
        dirs.close()
    
    if not os.path.exists("Dispersed_" + values.frb + "/sd_" + str(DM)):
        print("No directory found at Dispersed_" + values.frb + "/sd_" + str(DM))
        return
    elif not os.path.exists("Dispersed_" + values.frb + "/sd_" + str(DM) + "/extracted_outputs.txt"):
        print("No file found at Dispersed_" + values.frb + "/sd_" + str(DM) + "/extracted_outputs.txt")
        return
    else:
        data = np.genfromtxt("Dispersed_" + values.frb + "/sd_" + str(DM) + "/extracted_outputs.txt", skip_header=1)
    
    for i in range(len(data[:,1])-1):
        if data[i,1] >= data[i+1,1]:
            break1 = i+1

    # Plotting
    plt.figure()
    plt.clf
    plt.xlabel(r"SD", fontsize='x-large') 
    plt.ylabel(r"SNR", fontsize='x-large')
    plt.plot(data[:break1,1], data[:break1,2], '.', label='Whole grid')
    plt.plot(data[break1:,1], data[break1:,2], '.', label='Channel-wise')
    plt.legend()
    plt.savefig("plots/"+values.frb+"_sd_" + str(DM) + ".pdf", bbox_inches='tight')
    plt.close()

#==============================================================================

main()
