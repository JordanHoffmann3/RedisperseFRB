"""
Program: plot_fof.py
Author: Jordan Hoffmann
Date: 05/04/2023
Purpose: Plot things from fredda cand and cand.fof files

Inputs:
    FRB = FRB number
    DM = DM of FRB
    
Outputs:
    Requested plots
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
    parser.add_argument(dest='frb', help='FRB name e.g. 181112', type=str)
    parser.add_argument(dest='dm', help='DM of candidate file', type=float)
    parser.set_defaults(verbose=False, nxy="1,1")
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    #data = np.genfromtxt("Data/" + str(values.frb) + "/" + str(values.frb) + "_param.dat")
    data = np.genfromtxt("Dispersed_" + str(values.frb) + "/fredda_outputs/" + str(values.frb) + "_DM_" + str(values.dm) + ".fil.cand")

    plt.figure()
    plt.plot(data[:,5], data[:,0], 'rx')
    plt.ylabel("SNR")
    plt.xlabel("FREDDA DM")
    plt.vlines(values.dm, 0, np.max(data[:,0])*2, 'b')
    plt.ylim(0,np.max(data[:,0])*1.3)
    plt.savefig("plots/" + str(values.frb) + "_cand_plot_DM_" + str(values.dm) + ".png")
    plt.close()

    

main()
