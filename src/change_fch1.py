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
import sigproc
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def main():

    # Parse command line parameters
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='filename', help='Filterbank file name', type=str)
    parser.add_argument(dest='outfile', help='Outpute file name', type=str)
    parser.set_defaults(verbose=False, nxy="1,1")
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    f = sigproc.SigprocFile(values.filename)

    # Reverse header
    header = f.header
    
    header['fch1'] = header['fch1'] + header['foff']/2.

    # Reverse frequency ordering in data
    f.seek_data()
    v = np.fromfile(f.fin, dtype=np.uint8, count=-1)
    v.shape = (-1, f.nchans)

    # Write to new filterbank file
    f_out = sigproc.SigprocFile(values.outfile,'wb',header)
    f_out.seek_data()
    v.tofile(f_out.fin)

    f.fin.flush()
    f.fin.close()
    f_out.fin.flush()
    f_out.fin.close()

main()
