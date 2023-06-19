"""
Program: incoherent_dedispersion.py
Author: Jordan Hoffmann
Date: 07/03/2023
Purpose: Incoherently dedispersed an FRB and obtains the SNR for it
Last update: Initial creation

Inputs:
    frb = FRB ID number
    DM = Dispersion measure to redisperse to
    f_low = Lowest observational frequency (MHz)
    f_high = Highest observational frequency (MHz)
    int_t = Integration time (ms)
    
    Note: One input file is expected
        ../outputs/<frb>_DM_<DM>.fil

Outputs to Dispersed_<frb>/incoherent_dedispersion/<frb>_incoherent_SNR.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sigproc
import os
import glob
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

fignum=0

def main():

    # Parse command line parameters
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='frb', help='FRB name e.g. 181112', type=str)
    parser.add_argument(dest='DM', help='Redispersed DM', type=float)
    parser.add_argument(dest='f_low', help='Lowest observational frequency', type=float)
    parser.add_argument(dest='f_high', help='Highest observational frequency', type=float)
    parser.add_argument(dest='int_t', help='Integration time', type=float)
    parser.add_argument('-n', default='10', dest='n', help='Max number of time bins integrated over', type=int)
    parser.add_argument('-s', '--start', default=5, help='Starting position to read the filterbank from (seconds)', type=int)
    parser.add_argument('-p', '--plot', default=False, action='store_true', help='Turn on to produce and save pdfs of the outputs')
    parser.add_argument('-w', '--binwidth', dest='width', nargs='?', default=1.0, help='Width of the frequency bins', type=float)
    parser.set_defaults(verbose=False, nxy="1,1")
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    D = 4.148808e-3 # Dispersion constant in GHz^2 pc^-1 cm^3 s
    max_dt = D * values.DM * ((values.f_low/1e3)**(-2) - (values.f_high/1e3)**(-2))

    seconds = []
    seconds.append(values.start)
    seconds.append(int(max_dt) + 10)
    # print("Data length = " + str(seconds[1]) + "s")

    t_start = int(seconds[0] / values.int_t * 1e3)
    n_times = int(seconds[1] / values.int_t * 1e3)

    N = int((values.f_high - values.f_low)/values.width)      # Number of frequency bins

    #filename = 'Dispersed_' + str(values.frb) + '/outputs/' + str(values.frb) + '_DM_' + str(values.DM) + '.fil'
    filename = '../outputs/' + str(values.frb) + '_DM_' + str(values.DM) + '.fil'
    spec, f_off = load_beams(filename, t_start, n_times)
    spec = spec.reshape([n_times, N])

    if f_off[0] < 0:
        spec = np.fliplr(spec)

    t_spec = np.arange(n_times) * values.int_t + seconds[0]
    f_spec = np.arange(N) * values.width + values.f_low

    #showSpec(np.transpose(spec), t_spec, f_spec)
    spec_dedispersed = applyDedispersion(spec, values, f_spec)
    SNR_array, SNR_pos = getSNR(spec_dedispersed, values.n)
    print("DM: " + str(values.DM) + " Width: " + str(np.argmax(SNR_array)+1), flush=True)

    if values.plot:
        saveSpec(spec.T, t_spec, f_spec, values.frb + "_DM_" + str(values.DM))
        saveSpec(np.transpose(spec_dedispersed), np.arange(spec_dedispersed.shape[0])*values.int_t*1e-3 + seconds[0], f_spec, values.frb + "_DM_" + str(values.DM) + "_dedispersed")
        w = np.argmax(SNR_array) + 1
        saveSpec(np.transpose(spec_dedispersed[SNR_pos[w-1]-5:SNR_pos[w-1]+5,:]), np.arange(w+10)*values.int_t*1e-3, f_spec, values.frb + "_DM_" + str(values.DM) + "_zoomed", vlines=[SNR_pos[w-1]-w/2.0,SNR_pos[w-1]+w/2.0])

    file_out = open(values.frb+'_incoherent_SNR.txt', 'a')
    file_out.write(str(values.DM) + "\t" + str(np.max(SNR_array)) + "\n")
    file_out.close()

#==============================================================================
"""
Function: load_beams
Date: 07/03/2023
Author: Keith Bannister (taken from github.com/askap-craco/craft/)
Purpose:
Loads a filterbank file (or directory of .fil files) into a numpy array format

Imports:
    path = Path to the .fil file or directory of .fil files
    tstart = Index of extraction start from the beginning of the file
    ntimes = Number of time bins to extract
    pattern = File format
    return_files = Returns the files used for extraction

Exports:
    data = Specified data from the file as a numpy array
    sigfiles = List of files extracted from
"""
def load_beams(path, tstart, ntimes, pattern='*.fil', return_files=False):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, pattern)))
    elif isinstance(path, str):
        files = [path]
    else:
        files = path
       
    if len(files) == 0:
        raise ValueError('No files in path  %s' % path)

    data = None
    sigfiles = []
    f_off = []

    for ifname, fname in enumerate(files):
        f = sigproc.SigprocFile(fname)
        sigfiles.append(f)
        tend = tstart + ntimes
        nelements = ntimes*f.nifs*f.nchans
        
        f.seek_data(int(f.bytes_per_element)*tstart)
        if (f.nbits == 8):
            dtype = np.uint8
        elif (f.nbits == 32):
            dtype = np.float32

        v = np.fromfile(f.fin, dtype=dtype, count=nelements )
        v.shape = (-1, f.nifs, f.nchans)
        

        if f.nifs == 1:
            nifs = len(files)
            ifslice = slice(0, nifs)
            if data is None:
                data = np.zeros((ntimes, nifs, f.nchans))

            #ifnum = int(fname.split('.')[-2])
            ifnum = ifname
            data[0:v.shape[0], ifnum, :] = v[:, 0, :]

            '''
            print 'load beams', v.shape, data.shape, ifnum, ifname
            print 'WARNING! WHY IS THAT DEAD CHANNEL IN THERE< EVEN WITH ONES?'
            import pylab
            pylab.figure()
            pylab.imshow(v[:, 0, :])
            pylab.title('v')
            pylab.figure()
            pylab.imshow(data[:, ifnum, :])
            pylab.title('data')
            '''

        else:
            data = v
        
        f_off.append(f.foff)

    if return_files:
        return data, sigfiles
    else:
        return data, f_off

#==============================================================================
"""
Function: commasep
Date: 07/03/2023
Author: Keith Bannister (taken from github.com/askap-craco/craft/)
Purpose:
Separates a string of comma separated values into a list

Imports:
    s = String of the CSVs

Exports:
    list of the CSVs
"""
def commasep(s):
    return list(map(int, s.split(',')))

#==============================================================================
"""
Function: saveSpec
Date: 02/05/2022
Purpose:
Saves spectrogram as png file

Imports:
    spec = Spectrogram to be saved
    t = Time vector in ms
    f = Frequency vector in MHz
    file_name = Name of png file
    vlines = Position of vertical lines

Exports:
    spec as a pdf to file_name
"""
def saveSpec(spec, t, f, file_name, vlines=None):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    #plt.pcolormesh(t, f, spec, shading='auto')
    ax.imshow(spec, aspect="auto", extent=[t[0], t[-1], f[0], f[-1]], origin='lower')
    fig.colorbar()
    if vlines != None:
        ax.axvline(vlines)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    plt.savefig(file_name + ".png", format="png", bbox_inches='tight')

#==============================================================================
"""
Function: applyDedispersion
Date: 07/03/2023
Purpose:
Incoherently dedisperse a spectrogram

Imports:
    spec = Spectrogram to be dedispersed
    values = List of command line inputs. Of interest here are:
            values.f_low: Lowest frequency of the observing band
            values.int_t: Time resolution of the integrated spectrogram
    f_spec = Vector of frequencies corresponding to the spectrogram

Exports:
    spec_dedispersed = Central region of the dedispersed spectrogram
"""
def applyDedispersion(spec, values, f_spec):
    D = 4.148808 # Dispersion constant in GHz^2 pc^-1 cm^3 ms
    dt = D * values.DM * ((values.f_low/1e3)**(-2) - (f_spec/1e3)**(-2))
    dn = np.floor(dt/values.int_t).astype(int)

    spec_dedispersed = np.zeros([spec.shape[0] + np.max(dn) + 1, spec.shape[1]])

    for i in range(len(dn)):
        spec_dedispersed[dn[i]:dn[i]+spec.shape[0],i] = spec[:,i]
    
    return spec_dedispersed[dn[-1]:spec.shape[0],:]

#==============================================================================
"""
Function: getSNR
Date: 07/03/2023
Purpose:
Obtain the SNR for a dedispersed pulse

Imports:
    spec_dedispersed = Spectrogram of the dedispersed pulse
    n = Maximum number of time bins to average over in pulse search

Exports:
    SNR_array = Array of SNR values for each pulse width searched for
"""
def getSNR(spec_dedispersed, n):
    spec_array = np.zeros(n, dtype=object)
    SNR_array = np.zeros(n, dtype=float)
    SNR_pos = np.zeros(n, dtype=int)
    signal_array = np.zeros(n, dtype=object)

    # global fignum
    # fignum=fignum+1
    # plt.figure(fignum)
    # plt.clf()
    # plt.xlabel("Time (s)")
    # plt.ylabel("SNR")
    
    # i=0
    # while True:
    for i in range(n):
        spec_array[i] = np.zeros([spec_dedispersed.shape[0]-i, spec_dedispersed.shape[1]])

        for j in range(spec_array[i].shape[0]):
            spec_array[i][j,:] = np.sum(spec_dedispersed[j:j+i+1,:], axis=0) / (i+1)

        signal = np.sum(spec_array[i], axis=1) / spec_array[i].shape[1]

        buf = 100
        if np.argmax(signal) > len(signal)/2: 
            noise = signal[:int(np.argmax(signal)-buf)]
        else:
            noise = signal[int(np.argmax(signal)+buf):]

        # if np.argmax(signal) > len(signal)/2: 
        #     noise = signal[:len(signal)//2-buf]
        # else:
        #     noise = signal[len(signal)//2+buf:]

        mean, sd = stats.norm.fit(noise)
        # print(mean, sd)

        spec_array[i] = (spec_array[i] - mean) / sd
        signal_array[i] = (signal - mean) / sd

        SNR_array[i] = np.max(signal_array[i])
        SNR_pos[i] = int((np.argmax(signal_array[i]) + 0.5) * (i+1))

        # plt.plot(np.arange(len(signal_array[i])), signal_array[i], label=i)

        # i = i+1

        # if i>=n: #or (i>1 and SNR_array[i] < SNR_array[i-1]):
        #     break

    # plt.legend()
    # plt.savefig("SNR.png", format="png", bbox_inches='tight')

    return SNR_array, SNR_pos #, signal_array, spec_array

#==============================================================================

main()
