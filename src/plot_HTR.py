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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

plt.rcParams["figure.figsize"] = (8,6)

def main():

    # Parse command line parameters
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='frbs', help='FRB names e.g. 181112,210117,...', type=commasep)
    parser.add_argument('--dt', dest='dt', type=float, default=None, help='Time resolution to integrate to (ms). If None, use full resolution')
    parser.add_argument('--buf', dest='buf', type=commasep, default=None, help='Buffer time to trim around the pulse (ms). If one value is given a symmetrical buffer is taken, otherwise <left>,<right> buffer is taken. If None the whole data length is used.')
    parser.add_argument('-e', '--export', dest='export', default=False, action='store_true', help='Save shortened files centred on pulse')
    parser.add_argument('-p', '--pos', dest='pos', default=None, type=float, help='Specify FRB position in seconds from the file start')
    parser.set_defaults(verbose=False, nxy="1,1")
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    global fignum
    global frb

    fignum=0

    for frb in values.frbs:
        data = np.genfromtxt("Data/" + str(frb) + "/" + str(frb) + "_param.dat")
        DM = data[1]
        f_min = data[2]
        f_max = data[3]

        dt = values.dt

        if not exists("Data/" + str(frb) + "/" + str(frb) + "_x_t_" + str(DM) + "_full.npy"):
            print("No data files found \nExpected at: Data/" + str(frb) + "/" + str(frb) + "_x_t_" + str(DM) + "_full.npy")
            exit()

        x_t = np.load("Data/" + str(frb) + "/" + str(frb) + "_x_t_" + str(DM) + "_full.npy")
        y_t = np.load("Data/" + str(frb) + "/" + str(frb) + "_y_t_" + str(DM) + "_full.npy")

        I = np.abs(x_t)**2 + np.abs(y_t)**2
        
        if dt!=None:
            n_avg = int(dt*1e3*(f_max-f_min))
        
            dt = 1e-3 / (f_max - f_min) * n_avg
            n_excess = len(I) % n_avg
            I = I[:len(I)-n_excess]
            I = [np.sum(I[i:i+n_avg]) for i in range(0, len(I), n_avg)]
        else:
            n_avg = 1
            dt = 1/336 * 1e-6 * 1e3

        # frbPos = np.argmax(I)

        # if values.buf == None:
        #     noise_buf = 20
        # else:
        #     noise_buf = values.buf
        
        # if (frbPos > len(I)/2):
        #     noise = I[:frbPos-int(noise_buf/dt)]
        # else:
        #     noise = I[frbPos+int(noise_buf/dt):]

        # mean, sd = stats.norm.fit(noise)
        # I = (I - mean) / sd

        # print(np.max(I))
        # max_idx = np.argmax(I)
        # I = I[max_idx - buf:max_idx + buf]

        # Check if a buffer is specified, otherwise just save the whole thing
        if values.buf != None:
            buf = np.array(values.buf).astype(float)
            if len(buf) == 1:
                buf = np.append(buf, buf[0])
            
            buf_idx = (buf / dt).astype(int)

            # If FRB position is specified centre it there, otherwise find the largest peak as the FRB position
            if values.pos != None:
                left_idx = int((values.pos - 0.1) / dt * 1e3)
                right_idx = int((values.pos + 0.1) / dt * 1e3)
            else:
                burst = np.where(I > np.max(I)/2)[0]
                left_idx = burst[0]
                right_idx = burst[-1]

                print("left_idx, right_idx:" , left_idx, right_idx)            
                print("left_idx, right_idx (ms):", str(left_idx*dt), str(right_idx*dt))
                print("fwhm (ms):", str((right_idx - left_idx)*dt))

            I = I[left_idx - buf_idx[0]: right_idx + buf_idx[1]]

            # Check if saving .npy files
            if values.export:
                left_idx = left_idx * n_avg
                right_idx = right_idx * n_avg
                buf_idx = buf_idx * n_avg

                np.save("Data/" + str(frb) + "/" + str(frb) + "_x_t_" + str(DM) + ".npy", x_t[left_idx - buf_idx[0]: right_idx + buf_idx[1]])
                np.save("Data/" + str(frb) + "/" + str(frb) + "_y_t_" + str(DM) + ".npy", y_t[left_idx - buf_idx[0]: right_idx + buf_idx[1]])

            # For plotting purposes
            t_start = dt*(left_idx-buf_idx[0]) * n_avg
        else:

            # For plotting purposes
            t_start = 0.0

        # Plot HTR data
        t = dt * np.arange(len(I)) + t_start

        print("Saved")
        fignum = fignum+1
        plt.figure(fignum)
        plt.plot(t, I)
        plt.xlabel("Time (ms)")
        plt.ylabel("Intensity")
        plt.savefig("plots/" + str(frb) + "_HTR.png")

#==============================================================================
"""
Function: commasep
Date: 23/08/2022
Purpose:
    Turn a string of variables seperated by commas into a list

Imports:
    s = String of variables

Exports:
    List conversion of s
"""
def commasep(s):
    return list(map(str, s.split(',')))

#==============================================================================

main()
