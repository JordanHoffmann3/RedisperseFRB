"""
Program: plot_HTR.py
Author: Jordan Hoffmann
Date: 15/10/2022
Purpose: Plot HTR data of the given FRBs

Inputs:
    FRB_list = List of FRB ID numbers
    
Outputs:
    FRB_HTR.png = Time series of the intensity of the HTR data
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize
from scipy import stats
from os.path import exists
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

plt.rcParams["figure.figsize"] = (8,6)

def main():

    # Parse command line parameters
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='frb', help='FRB name', type=str)
    parser.add_argument('--dt', dest='dt', type=float, default=None, help='Time resolution to integrate to (ms). If None, use full resolution')
    parser.add_argument('--buf', dest='buf', type=float, default=None, help='Duration to extract centred at the pulse (ms). If None the whole data length is used.')
    # parser.add_argument('-e', '--export', dest='export', default=False, action='store_true', help='Save shortened files centred on pulse')
    parser.add_argument('-p', '--pos', dest='pos', default=-1, type=float, help='Specify FRB position in seconds from the file start')
    parser.add_argument('-f', '--files', dest='f', default=None, nargs=3, type=str, help='Paths to x_t and y_t numpy files and param.dat file')
    parser.add_argument('-o', '--out_dir', dest='out_dir', default="", type=str, help='Save shortened files centred on pulse in given directory')
    parser.set_defaults(verbose=False, nxy="1,1")
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    global fignum
    global frb
    frb = values.frb

    dt = values.dt

    if values.f == None:
        xtf = "Data/" + str(frb) + "/" + str(frb) + "_X_t_" + str(DM) + "_full.npy"
        ytf = "Data/" + str(frb) + "/" + str(frb) + "_Y_t_" + str(DM) + "_full.npy"
        datf = "Data/" + str(frb) + "/" + str(frb) + "_param.dat"
    else:
        xtf = values.f[0]
        ytf = values.f[1]
        datf = values.f[2]

    if not exists(xtf):
        print("No data file found at: " + xtf)
        exit()
    if not exists(ytf):
        print("No data file found at: " + ytf)
        exit()
    if not exists(datf):
        print("No parameter file found at :" + datf)
        exit()

    x_t = np.load(xtf)
    y_t = np.load(ytf)
    data = np.genfromtxt(datf)

    DM = data[1]
    B = 336             # Bandwidth MHz

    I = np.abs(x_t)**2 + np.abs(y_t)**2
    
    if dt!=None:
        n_avg = int(dt*1e3*B)
    
        dt = 1e-3 / B * n_avg
        n_excess = len(I) % n_avg
        I = I[:len(I)-n_excess]
        I = [np.sum(I[i:i+n_avg]) for i in range(0, len(I), n_avg)]
    else:
        n_avg = 1
        dt = 1.0/B * 1e-6 * 1e3

    # Check if a buffer is specified, otherwise just save the whole thing
    # also save the whole thing if the buffer is longer than the  whole length...
    if values.buf != None and values.buf < (len(I)*dt):
        buf_idx = int(values.buf / dt)

        # If FRB position is specified centre it there, otherwise find the largest peak as the FRB position
        if values.pos != -1:
            left_idx = int((values.pos) / dt * 1e3)
            right_idx = left_idx
        else:
            # Normalise to determine burst position
            frbPos = np.argmax(I)
            if (frbPos > len(I)/2):
                noise = I[:frbPos-int(10/dt)]
            else:
                noise = I[frbPos+int(10/dt):]

            mean, sd = stats.norm.fit(noise)
            I = (I - mean) / sd

            # Define burst position by FWHM
            burst = np.argwhere(I > np.max(I)/2)
            left_idx = burst[0][0]
            right_idx = burst[-1][0]

            print("left_idx, right_idx:" , left_idx, right_idx)            
            print("left_idx, right_idx (ms):", str(left_idx*dt), str(right_idx*dt))
            print("fwhm (ms):", str((right_idx - left_idx)*dt))

        if left_idx - buf_idx//2 < 0:
            I = I[0:buf_idx]
            x_t = x_t[0:buf_idx*n_avg]
            y_t = y_t[0:buf_idx*n_avg]
            t_start = 0.0
        elif right_idx + buf_idx//2 > len(I):
            I = I[-buf_idx:]
            x_t = x_t[-buf_idx*n_avg:]
            y_t = y_t[-buf_idx*n_avg:]
            t_start = (len(I) - buf_idx) * dt
        else:
            I = I[left_idx - buf_idx//2: right_idx + buf_idx//2]
            x_t = x_t[(left_idx - buf_idx//2)*n_avg: (right_idx + buf_idx//2)*n_avg]
            y_t = y_t[(left_idx - buf_idx//2)*n_avg: (right_idx + buf_idx//2)*n_avg]
            t_start = (left_idx - buf_idx//2) * dt

    else:

        # For plotting purposes
        t_start = 0.0

    # Check if saving .npy files
    if values.out_dir != None:
        np.save(os.path.join(values.out_dir, str(frb) + "_X_t_" + str(DM) + ".npy"), x_t)
        np.save(os.path.join(values.out_dir, str(frb) + "_Y_t_" + str(DM) + ".npy"), y_t)

    # Plot HTR data
    t = dt * np.arange(len(I)) + t_start

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    ax.plot(t, I)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Intensity")
    plt.savefig("plots/" + str(frb) + "_HTR.png", format="png", bbox_inches='tight')

#==============================================================================

main()
