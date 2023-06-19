"""
File: write_filterbank.py
Author: Jordan Hoffmann (adapted from Harry Qiu's work)
Date: 05/05/2022
Purpose:
Contains functions used to create a filterbank file for simulated FRBs.
"""

import numpy as np
import matplotlib.pyplot as plt
import fbio

#==============================================================================
"""
Function: makeHeader
Date: 05/05/2022
Credits: Adapted from Harry Qiu's makeheader function from his simfred 
         directory.
Purpose:
Create a header for a filterbank file.

Imports:
    fch1 = First frequency channel (MHz)
    foff = Frequency offset between channels (ie fch2 - fch1) (MHz)
    nchan = Number of frequency channels
    tsamp = Sampling time (s)

Exports:
    header = Header for filterbank file as a dictionary.
"""
def makeHeader(fch1,foff,nchan,tsamp):
    header={'az_start': 0.0,
        'barycentric': None,
        'data_type': 1,
        'fch1': fch1,
        'fchannel': None,
        'foff': foff,
        'machine_id': 0,
        'nbits': 8,
        'nchans': nchan,
        'nifs': 1,
        'nsamples': None,
        'period': None,
        'pulsarcentric': None,
        'rawdatafile': None,
        'refdm': None,
        'source_name': 'Fake FRB',
        'src_raj':174540.1662,
        'src_dej':-290029.896,
        'telescope_id': 7,
        'tsamp': tsamp,
        'tstart': 57946.52703893818,
        'za_start': 0.0}
    
    return header

#==============================================================================
"""
Function: inject
Date: 05/05/2022
Credits: Adapted from Harry Qiu's inject function from his simfred directory.
Purpose:
Write a filterbank file for an FRB.

Imports:
    mockheader = Header for the filterbank file (see makeHeader above)
    output = Output file name
    burst = FRB burst as a numpy 2D array

Exports:
    filterbank file named fileOut.fil
"""

def inject(mockheader,output,burst):
    filterbank=fbio.makefilterbank(output+".fil",header=mockheader)
    filterbank.writeblock(burst.astype(np.uint8))
    filterbank.closefile()

#==============================================================================

