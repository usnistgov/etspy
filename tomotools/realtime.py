# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Realtime module for TomoTools package.

@author: Andrew Herzing
"""

import tomotools.datasets as ds
import time
import glob
import hyperspy.api as hs
from tomotools import io


def simulate_acquisition(delay=20, outpath='/tmp/tomo/', ext='tif'):
    """
    Simulate a real time acquisition of a needle-shaped test specimen.

    Parameters
    ----------
    delay : int
        Length of time in seconds between saving frames.

    outpath : str
        Location where the frames will be saved.  Default is /tmp/tomo

    ext : str
        Format for saving the frames. Default is tif.
    """
    stack = ds.get_needle_data()
    tilts = stack.metadata.Tomography.tilts
    nimages = stack.data.shape[0]

    for i in range(0, nimages):
        outfile = outpath + 'TiltSeries_Slice%s_Angle%s.%s' % (i + 1, int(tilts[i]), ext)
        stack.inav[i].save(outfile, overwrite=True)
        time.sleep(delay)
    return


def get_stack(datapath, ext):
    """
    Load an acquisition simulation dataset.

    Parameters
    ----------
    datapath : str
        Location where the frames are stored.

    ext : str
        File extension to look for.
    """
    imfiles = glob.glob(datapath + '/*.' + ext)
    stack = hs.load(imfiles, stack=True)
    nimages = stack.data.shape[0]
    tilts = stack.metadata.Tomography.tilts[0:nimages]
    stack = io.convert_to_tomo_stack(stack, tilts=tilts)
    return stack
