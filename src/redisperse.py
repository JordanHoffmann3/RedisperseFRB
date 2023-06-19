"""
Program: redisperse.py
Author: Jordan Hoffmann
Date: 16/06/2023
Purpose: Redisperses a dedispersed FRB.
Last update: Added masking and changed plotting

Inputs:
    FRB = FRB ID number
    real_DM = Real DM of the FRB
    DM = Dispersion measure to redisperse to
    f_low = Lowest observational frequency (MHz)
    f_high = Highest observational frequency (MHz)
    int_t = Integration time (ms)
    
    Note: Four input files are expected
        x,y for frequency,time.
        They are assumed to be named <frb>_<x or y>_<f or t>_<real_DM>.npy placed in a folder called Data/<frb>
        e.g. Data/181112/181112_x_f_589.265.npy

Outputs to Dispersed_<frb> folder
    <frb>_DM<dm>.npy = Frequency-time numpy matrix for redispersed and reconstructed FRB
        e.g. Dispersed_181112/181112_DM0.npy

    <frb>_DM<dm>.png = Frequency-time spectrogram image for redispersed FRB
        e.g. Dispersed_181112/181112_DM0.png

    <frb>_DM<dm>_reconstructed.png = Frequency-time spectrogram image for redispersed and reconstructed FRB
        e.g. Dispersed_181112/181112_DM0_reconstructed.png
"""

import numpy as np
import sys
from os.path import exists
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy import stats
import write_filterbank as wf
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time

def main():

    # Parse command line parameters
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='frb', help='FRB name e.g. 181112', type=str)
    parser.add_argument(dest='real_DM', help='Original dispersion measure', type=str)
    parser.add_argument(dest='DM', help='Dispersion measure to redisperse to', type=float)
    parser.add_argument(dest='f_low', help='Lowest observational frequency', type=float)
    parser.add_argument(dest='f_high', help='Highest observational frequency', type=float)
    parser.add_argument(dest='int_t', help='Integration time', type=float)
    parser.add_argument('-w', '--binwidth', dest='width', nargs='?', default=1.0, help='Width of the frequency bins', type=float)
    parser.add_argument('-r', '--reverse_fq', default=False, action='store_true', help='Start from the highest frequency when saving to filterbank')
    parser.add_argument('-o', '--offset', dest='offset', nargs='?', default=0.0, help='Offset to consider to change integration start position as a percentage of one integration bin', type=float)
    parser.add_argument('-f', '--files', dest='files', nargs='+', help='File location of: x_time_data y_time_data [x_freq_data y_freq_data]. If not specified expected to be in "Data" folder', type=commasep)
    parser.add_argument('--sd_f', default=8, dest='sd_f', help='Standard deviation for the final filterbank file (converetd to 8-bit integer)', type=int)
    parser.add_argument('-n', '--normalisation', default=0, dest='normalisation', help='Normalisation by: 0 = whole grid, 1 = frequency channels', type=int)
    parser.add_argument('-s', '--save', default=False, action='store_true', help='Turn on to produce and save pdfs of the outputs')
    parser.add_argument('-p', '--pfb', default=False, action='store_true', help='Turn on to use a simple pfb')
    parser.set_defaults(verbose=False, nxy="1,1")
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read data files
    if values.files == None:
        if exists("Data/"+values.frb+"/"+values.frb+"_x_t_"+values.real_DM+".npy"):
            print("Loading x time data", flush=True)
            x_t = np.load("Data/"+values.frb+"/"+values.frb+"_x_t_"+values.real_DM+".npy")
        else:
            print("Filepath to x time data is expected to be specified with -f or located at Data/"+values.frb+"/"+ values.frb + "_x_t_" + values.real_DM + ".npy", flush=True)
            sys.exit()

        if exists("Data/"+values.frb+"/"+values.frb+"_y_t_"+values.real_DM+".npy"):
            print("Loading y time data", flush=True)
            y_t = np.load("Data/"+values.frb+"/"+values.frb+"_y_t_"+values.real_DM+".npy")
        else:
            print("Filepath to y time data is expected to be specified with -f or located at Data/"+values.frb+"/" + values.frb + "_y_t_" + values.real_DM + ".npy", flush=True)
            sys.exit()
    else:
        print("Loading x time data", flush=True)
        x_t = np.load(values.files(0))
        print("Loading y time data", flush=True)
        y_t = np.load(values.files(1))

    # If the frequency files exist read them, otherwise create them
    if values.files != None and len(values.files) == 3:
        print("Loading x fq data", flush=True)
        x_f = np.load(values.files(2))
    elif exists("Data/"+values.frb+"/"+values.frb+"_x_f_"+values.real_DM+".npy"):
        print("Loading x fq data", flush=True)
        x_f = np.load("Data/"+values.frb+"/"+values.frb+"_x_f_"+values.real_DM+".npy")
    else:
        x_f = fft(x_t)
        print("Saving x fq data", flush=True)
        np.save("Data/"+values.frb+"/"+values.frb+"_x_f_"+values.real_DM+".npy", x_f)

    if values.files != None and len(values.files) == 4:
        print("Loading y fq data", flush=True)
        y_f = np.load(values.files(3))
    elif exists("Data/"+values.frb+"/"+values.frb+"_y_f_"+values.real_DM+".npy"):
        print("Loading y fq data", flush=True)
        y_f = np.load("Data/"+values.frb+"/"+values.frb+"_y_f_"+values.real_DM+".npy")
    else:
        y_f = fft(y_t)
        print("Saving y fq data", flush=True)
        np.save("Data/"+values.frb+"/"+values.frb+"_y_f_"+values.real_DM+".npy", y_f)

    # Set parameters
    N = int((values.f_high - values.f_low)/values.width)      # Number of frequency bins

    L = len(x_f)
    t_space = 1/(N*1e6)
    M = int(np.floor((values.int_t*1e-3)/t_space/N))     # Average over M time bins
    spec_space = t_space * N * M
    values.offset = int(np.round(values.offset * M / 100.0))

    f_total = fftfreq(L, t_space*1e6) + values.f_low
    f_total[int(np.ceil((L)/2))::] = f_total[int(np.ceil((L)/2))::] + N

    f = fftfreq(N, values.width/N) + values.f_low
    f[int(np.ceil(N/2))::] = f[int(np.ceil(N/2))::] + N

    # Check if a DM0 baseline exists already then read / create it
    if not exists("Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_0.0.npy"):
        orig_spec = getSpec(x_t, y_t, N, M, values)
        np.save("Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_0.0",orig_spec)
        print("DM0 baseline saved", flush=True)
    else:
        orig_spec = np.load("Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_0.0.npy")
        print("DM0 baseline loaded", flush=True)

    # Apply a dispersion to frequency data
    x_f_dispersed, y_f_dispersed = applyDispersion(values.DM, values.f_low, values.f_high, f_total, x_f, y_f)
    print("Dispersion to DM " + str(values.DM) + " applied", flush=True)

    # IFFT to get 'dispersed time data'
    x_t_dispersed = ifft(x_f_dispersed)
    y_t_dispersed = ifft(y_f_dispersed)

    # Convert dispersed data into a spectrogram
    dispersed_spec = getSpec(x_t_dispersed, y_t_dispersed, N, M, values)
    print("Dispersed spectrogram created", flush=True)

    # Reconstruct the spectrogram to eliminate wrapping of the signal
    intercept_idx = findFRBWrapPos(orig_spec, dispersed_spec, values, f, spec_space)
    mean, sd = getNoise(orig_spec)
    dispersed_spec_reconstructed = reconstructSpec(dispersed_spec, intercept_idx, mean, sd, N)
    print("Dispersed spectrogram reconstructed to avoid wrapping", flush=True)

    # Create filterbank appropriate spectrogram
    noise = getFilterbankSpec(spec_space, dispersed_spec_reconstructed, mean, sd, sd_f=values.sd_f, normalisation=values.normalisation)
    print("Noise added", flush=True)

    # Save to filterbank file
    if values.reverse_fq:
        header = header = wf.makeHeader(values.f_high, -values.width, N, spec_space)
        wf.inject(header,"Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_"+str(values.DM),np.flipud(np.transpose(noise)))
    else:
        header = wf.makeHeader(values.f_low, values.width, N, spec_space)
        wf.inject(header,"Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_"+str(values.DM),np.transpose(noise))
    print("Saved filterbank file", flush=True)

    if values.save:
        # Save the final spectrogram
        np.save("Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_"+str(values.DM),dispersed_spec_reconstructed)
        print("Reconstructed spectrogram saved", flush=True)
        
        # Save pngs of each spectrogram
        t = np.arange(orig_spec.shape[0]) * spec_space
        t_expanded = np.arange(dispersed_spec_reconstructed.shape[0]) * spec_space
        t_noise = np.arange(noise.shape[0]) * spec_space
        saveSpec(np.transpose(noise), t_noise, f, "Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_"+str(values.DM)+"_noisy")
        dispersed_spec = normaliseSpec(dispersed_spec, mean, sd, sd_f=values.sd_f, normalisation=values.normalisation)
        saveSpec(np.transpose(dispersed_spec), t, f, "Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_"+str(values.DM))
        dispersed_spec_reconstructed = normaliseSpec(dispersed_spec_reconstructed, mean, sd, sd_f=values.sd_f, normalisation=values.normalisation)
        saveSpec(np.transpose(dispersed_spec_reconstructed), t_expanded, f, "Dispersed_"+values.frb+"/outputs/"+values.frb+"_DM_"+str(values.DM)+"_reconstructed")
        print("Saved pngs", flush=True)

#==============================================================================
"""
Function: getSpec
Date: 28/07/2022
Purpose:
FFT in lots of N to produce spectrogram

Imports:
    x = x data
    y = y data
    values = Command line parameters
    N = Length of sections to be FFT'd
    M = Number of bins over which to average results
    offset = Number of bins which are ignored at the beginning in order to
             have averaging starting at different points

Exports:
    f = frequency array
    mat_mag = 2D array spectrogram

Last update:
    Fixed PFB implementation to do real and imaginary components separately
"""
def getSpec(x, y, N, M, values, taps=8):

    if values.pfb:
        # Do pfb averaging.
        window = make_pfb_window(N,taps,plot=False)
        window[:] = 0
        window[:N] = 5

        # Implement PFB on real component
        x_real = implement_PFB(np.real(x), window, N, taps)
        y_real = implement_PFB(np.real(y), window, N, taps)
    
        # Implement PFB on imaginary component
        x_imag = implement_PFB(np.imag(x), window, N, taps)
        y_imag = implement_PFB(np.imag(y), window, N, taps)

        # Combine real and imaginary components
        x = x_real + 1j*x_imag
        y = y_real + 1j*y_imag

    # Ignore the first 'offset' lots of fft lines to change where averaging
    # begins. One fft line is N data points.
    x = x[values.offset*N:]
    y = y[values.offset*N:]
    L = len(x)

    array_mag = np.zeros(int(L/(N*M)), dtype=object)
    
    for i in range(int(L/(N*M))):
        avg_mag = np.zeros(round(N), dtype=float)

        # Averaging over M bins
        for j in range(M):
            k = i*M+j
            start_idx = N*k
            end_idx = N*k + N

            x_f = fft(x[start_idx:end_idx])
            y_f = fft(y[start_idx:end_idx])
            avg_mag = avg_mag + abs(x_f)**2 + abs(y_f)**2
        
        array_mag[i] = avg_mag / M

    mat_mag = np.stack(array_mag)

    return mat_mag

#==============================================================================
"""
Function: applyDispersion
Date: 08/03/2023
Purpose:
Applies a DM to the frequency data

Imports:
    DM = Dispersion measure
    f_low = Lowest frequency
    f_high = Highest frequency
    f_total = Array of frequencies corresponding to the data
    x_f = Frequency data for x
    y_f = Frequency data for y

Exports:
    x_t_dispersed = Dispersed time data for x
    y_t_dispersed = Dispersed time data for y
"""
def applyDispersion(DM, f_low, f_high, f_total, x_f, y_f):
    # Apply dispersion
    D = 4.148808e3 # Dispersion constant in MHz^2 pc^-1 cm^3 s
    f_0 = f_high # Frequency dispersion is done relative to in MHz
    dphi = 2*np.pi * D*1e6 * DM * (f_total - f_0)**2 / ((f_0)**2 * f_total)
    x_f_dispersed = x_f * np.exp(1j*dphi)
    y_f_dispersed = y_f * np.exp(1j*dphi)

    return x_f_dispersed, y_f_dispersed

#==============================================================================
"""
Function: findFRBWrapPos
Date: 02/05/2022
Purpose:
Finds where the FRB intercepts any boundaries of the array. Essentially does
a sum over all bins corresponding to the correct frequency and finds which
starting bin has the highest S/N.

Imports:
    orig_spec = Original dedispersed spectrogram
    dispersed_spec = Spectrogram of dispersed FRB as a 2D numpy array
    DM = Dispersion measure
    values = Command line parameters
    f = Frequency vector
    spacing = Time spacing of the data

Exports:
    intercept_idx = List of indexes (corr. to frequencies) of where the FRB 
                    intercepts the array bounds and wraps.
"""
def findFRBWrapPos(orig_spec, dispersed_spec, values, f, spacing):

    t_len = dispersed_spec.shape[0]

    # Determine time delay for each frequency
    D = 4.148808e3 # Dispersion constant in MHz^2 pc^-1 cm^3 s
    dt = D * values.DM * (f**(-2) - values.f_high**(-2))

    start = np.argmax(np.sum(orig_spec, axis=1)) #- int(dt[-1] / values.int_t)

    # Construct a tuple of frequencies where crossing occurs to give boundaries for array restructuring
    intercept_idx = []

    while(np.round(max(dt)/spacing) + start > t_len):
        intercept_idx.append(np.where(dt > (t_len - start)*spacing)[0][-1])
        dt = dt - t_len*spacing

    print('Start: ', str(start), flush=True)

    return intercept_idx

#==============================================================================
"""
Function: reconstructSpec
Date: 19/05/2022
Purpose:
Creates a new spectrogram without any wrapping of the FRB. Pads extra space with noise.

Imports:
    dispersed_spec = Spectrogram of dispersed FRB as a 2D numpy array
    intercept_idx = List of indexes (corr. to frequencies) of where the FRB intercepts the array bounds and wraps
    mean = Vector of mean noise values for each frequency channel 
    sd = Vector of standard deviation of noise for each frequency channel
    N = Number of frequency bins
    spacing = Time spacing of the data
    buffer = Leeway for where the burst starts compared to the detection height
        - Larger ensures the entire burst is captured
        - Smaller ensures the next wrap is not captured
    

Exports:
    dispersed_spec_reconstructed = Reconstructed spectrogram
"""
def reconstructSpec(dispersed_spec, intercept_idx, mean, sd, N, buffer=10):

    numIntercepts = len(intercept_idx)
    if (numIntercepts > 0):

        # Create new padded array of noise
        padded = np.zeros(((numIntercepts + 1)*dispersed_spec.shape[0],dispersed_spec.shape[1]))
        for i in range(N):
            padded[:,i] = np.random.normal(mean[i],sd[i],((numIntercepts + 1)*dispersed_spec.shape[0]))
        
        padded_unordered = padded.copy()
        padded_unordered[:dispersed_spec.shape[0],:] = dispersed_spec

        # Copy in pieces as necessary
        fin = dispersed_spec.shape[0]
        mid = round(fin/2)

        # Start at top
        left_idx = N
        right_idx = N

        # Reconstruct array -> Takes blocks between each intercept frequency and accross 
        # half of the times and shifts them appropriately.
        for i in range(numIntercepts):
            # Odd blocks left side
            next_left_idx = intercept_idx[i] + buffer
            if (next_left_idx > N):
                next_left_idx = N

            padded[(i)*fin:(i)*fin+mid,next_left_idx:left_idx] = dispersed_spec[:mid,next_left_idx:left_idx]

            left_idx = next_left_idx

            # Even blocks right side
            next_right_idx = intercept_idx[i] - buffer
            if (next_right_idx < 0):
                next_right_idx = 0

            padded[(i)*fin+mid:(i)*fin+fin,next_right_idx:right_idx] = dispersed_spec[mid:fin,next_right_idx:right_idx]

            right_idx = next_right_idx

        # Do final blocks to the end
        next_left_idx = 0
        next_right_idx = 0
        i = numIntercepts
        padded[(i)*fin:(i)*fin+mid,next_left_idx:left_idx] = dispersed_spec[:mid,next_left_idx:left_idx]
        padded[(i)*fin+mid:(i)*fin+fin,next_right_idx:right_idx] = dispersed_spec[mid:fin,next_right_idx:right_idx]

    else:
        padded = dispersed_spec

    return padded

#==============================================================================
"""
Function: getNoise
Date: 19/05/2022
Purpose:
Get noise parameters

Imports:
    spec = Spectrogram to get noise from 

Exports:
    mean = Vector of mean noise in each frequency channel
    sd = Vector of standard deviation of noise in each frequency channel
"""
def getNoise(spec):

    # Get a sample of the noise
    summed = np.sum(spec,1)
    frbPos = np.where(summed == np.max(summed))[0][0]
    if (frbPos > spec.shape[0]/2):
        noise = spec[:frbPos-15,:]
    else:
        noise = spec[frbPos+15:,:]

    # Get noise parameters
    N = spec.shape[1]
    mean = np.zeros(N)
    sd = np.zeros(N)

    for i in range(N):
        mean[i],sd[i] = stats.norm.fit(noise[:,i])

    return mean, sd

#==============================================================================
"""
Function: getFilterbankSpec
Date: 28/02/2023
Purpose:
Makes a spectrogram suitable for a filterbank file. Adds noise on both
sides of the signal and normalises each band to mean_f and sd_f.

Imports:
    spec_space = Spacing of time signals in the spectrogram
    spec = Spectrogram containing signal
    mean = Vector of mean noise in each frequency channel
    sd = Vector of standard deviation of noise in each frequency channel
    mean_f = Final mean expected
    sd_f = Final standard deviation expected
    normalisation = Method of normalisation (0 = whole grid, 1 = by frequency 
                                            channels, 2 = by time channels)

Exports:
    spec_out = Final spectrogram with added noise and after normalisation
"""
def getFilterbankSpec(spec_space,spec,mean,sd,mean_f=96,sd_f=8,normalisation=0):

    # Determine amount of padding
    t_pad = spec.shape[0] * spec_space
    if t_pad < 5.0:
        t_pad = 5.0

    # Create large array of random noise and set the centre to be the redispersed data
    nsamp = int(np.round(t_pad / spec_space))
    spec_out = np.empty((2*nsamp + spec.shape[0], spec.shape[1]))
    for i in range(len(mean)):
        spec_out[:,i]=np.random.normal(mean[i], sd[i], spec_out.shape[0])
    
    spec_out[nsamp:nsamp + spec.shape[0],:] = spec

    spec_out = normaliseSpec(spec_out,mean,sd,mean_f,sd_f,normalisation)

    return spec_out

#==============================================================================
"""
Function: normaliseSpec
Date: 28/03/2023
Purpose:
Normalise a spectrogram to mean_f and sd_f.

Imports:
    spec = Spectrogram containing signal
    mean = Vector of mean noise in each frequency channel
    sd = Vector of standard deviation of noise in each frequency channel
    mean_f = Final mean expected
    sd_f = Final standard deviation expected
    normalisation = Method of normalisation (0 = whole grid, 
                                            1 = by frequency channels)

Exports:
    spec = Final spectrogram after normalisation
"""
def normaliseSpec(spec,mean,sd,mean_f=128,sd_f=18,normalisation=0):

    # Flag channels which are too large to ensure everything else is not rounded to 0
    means_mean = np.mean(mean)
    means_sd = np.std(mean)

    sds_mean = np.mean(sd)
    sds_sd = np.std(sd)

    for i in range(len(mean)):
        mean_z = (mean[i] - means_mean)/means_sd
        sd_z = (sd[i] - sds_mean) / sds_sd
        if (mean_z > 10) or (sd_z > 10):
            print("Channel " + str(i) + " masked")
            print("Mean z value: " + str(mean_z))
            print("SD z value: " + str(sd_z), flush=True)
            spec[:,i] = 0

    # Normalise by whole grid / frequency channels
    if normalisation==0:
        mean_t, sd_t = stats.norm.fit(spec[:,:])
        spec = (spec - mean_t) / sd_t
        spec = spec * sd_f + mean_f
    elif normalisation==1:
        for i in range(len(mean)):
            spec[:,i] = (spec[:,i] - mean[i]) / sd[i]
            spec[:,i] = spec[:,i] * sd_f + mean_f

    return spec

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

Exports:
    spec as a pdf to file_name
"""
def saveSpec(spec, t, f, file_name):
    # plt.figure()
    # #plt.pcolormesh(t, f, spec, shading='auto')
    # plt.imshow(spec, aspect="auto", extent=[t[0], t[-1], f[0], f[-1]], origin='lower', vmin=np.min(spec), vmax=np.max(spec))
    # plt.colorbar()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (MHz)")
    # plt.savefig(file_name + ".png", format="png", bbox_inches='tight')
    # plt.close()

    fig = plt.figure()
    ax = plt.subplot(1,1,1)

    #plt.pcolormesh(t, f, spec, shading='auto')
    ax.imshow(spec, aspect="auto", extent=[t[0], t[-1], f[0], f[-1]], origin='lower', vmin=np.min(spec), vmax=np.max(spec))
    fig.colorbar()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    plt.savefig(file_name + ".png", format="png", bbox_inches='tight')

#==============================================================================
"""
Function: make_pfb_window
Date: 05/07/2022
Author: Clancy James
Purpose:
Creates a window for a pfb

Imports:
    N = Length of FFT to later be performed on the data
    M = Number of taps (i.e. sums)
    plot = Boolean to save the window to a file

Exports:
    coeffs = Vector of coefficients corresponding to the pfb window
"""
def make_pfb_window(N,M,plot=True):
    # sine envelope
    # sets the phase to go from 0 to pi over length N*M
    # the end points are non-zero, hence arange from 1 to L
    # division by L+1
    L=N*M
    sin_phase = (np.arange(L)+1)*np.pi/(L+1)
    sine_term = np.sin(sin_phase)
    
    # sinc function, period of N
    dsinc=np.pi/N #so zeroes at +- N
    themin=-L/2+0.5
    themax=L/2-0.5
    sinc_input = np.linspace(themin,themax,L)/L * np.pi * M
    sfunction=np.sin(sinc_input)/sinc_input
    
    # coeffs
    coeffs=sfunction*sine_term
    
    if plot:
        plt.figure()
        plt.xlabel('Index of PFB')
        plt.ylabel('coefficient')
        plt.plot(coeffs)
        
        themin=np.min(coeffs)
        themax=np.max(coeffs)
        for i in (np.arange(M-1)+1):
            plt.plot([i*N,i*N],[themin,themax],linestyle='--',color='black')
        plt.tight_layout()
        plt.savefig('pfb_coeffs.pdf', bbox_inches='tight')
        plt.close()
    
    return coeffs

#==============================================================================
"""
Function: implement_PFB
Date: 05/07/2022
Author: Clancy James
Purpose:
    Implement a pfb on a data set ready to be FFTd

Imports:
    data = Vector of complex time data with which to apply the pfb window
    window = Vector of coefficients for the window
    N = Length of FFT to later be performed on the data
    M = Number of taps (i.e. sums)

Exports:
    opdata = Vector of data after the pfb averaging has been performed
"""
def implement_PFB(data,window,N,M):
    # requires the data to be some multiple of N
    nIPblocks=int(data.size/N)
    nOPblocks=int(nIPblocks-M+1)
    
    # remove last parts of data
    data=data[:nIPblocks*N]
    
    # reshape everything to have last dimension N (speedup)
    data=data.reshape([nIPblocks,N])
    window=window.reshape(M,N)
    
    # output array
    opdata=np.zeros([nOPblocks,N])
    
    for i in np.arange(N):
        opdata[:,i]=np.convolve(data[:,i],window[:,i],mode='valid')
    
    opdata=opdata.reshape([nOPblocks*N])
    
    return opdata

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

t0 = time.time()
main()
print("Total time taken: " + str(time.time() - t0) + " secs = " + str((time.time() - t0) / 60.0) + " mins")