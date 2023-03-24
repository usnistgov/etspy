# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Utility module for TomoTools package.

@author: Andrew Herzing
"""

import numpy as np


def parse_mrc_header(filename):
    """
    Read the mrc header and return as dictionary.

    Args
    ----------
    filename : str
        Name of the MRC file to parse

    Returns
    ----------
    headar : dict
        Dictionary with header values

    """
    header = {}
    with open(filename, 'r') as h:
        header['nx'], header['ny'], header['nz'] = np.fromfile(h, np.uint32, 3)
        header['mode'] = np.fromfile(h, np.uint32, 1)[0]
        header['nxstart'], header['nystart'], header['nzstart'] = np.fromfile(h, np.uint32, 3)
        header['mx'], header['my'], header['mz'] = np.fromfile(h, np.uint32, 3)
        header['xlen'], header['ylen'], header['zlen'] = np.fromfile(h, np.uint32, 3)
        _ = np.fromfile(h, np.uint32, 6)
        header['amin'], header['amax'], header['amean'] = np.fromfile(h, np.uint32, 3)
        _ = np.fromfile(h, np.uint32, 1)
        header['nextra'] = np.fromfile(h, np.uint32, 1)[0]
        _ = np.fromfile(h, np.uint16, 1)[0]
        _ = np.fromfile(h, np.uint8, 6)
        strbits = np.fromfile(h, np.int8, 4)
        header['ext_type'] = ''.join([chr(item) for item in strbits])
        header['nversion'] = np.fromfile(h, np.uint32, 1)[0]
        _ = np.fromfile(h, np.uint8, 16)
        header['nint'] = np.fromfile(h, np.uint16, 1)[0]
        header['nreal'] = np.fromfile(h, np.uint16, 1)[0]
        _ = np.fromfile(h, np.int8, 20)
        header['imodStamp'] = np.fromfile(h, np.uint32, 1)[0]
        header['imodFlags'] = np.fromfile(h, np.uint32, 1)[0]
        header['idtype'] = np.fromfile(h, np.uint16, 1)[0]
        header['lens'] = np.fromfile(h, np.uint16, 1)[0]
        header['nd1'], header['nd2'], header['vd1'], header['vd2'] = np.fromfile(h, np.uint16, 4)
        _ = np.fromfile(h, np.float32, 6)
        header['xorg'], header['yorg'], header['zorg'] = np.fromfile(h, np.float32, 3)
        strbits = np.fromfile(h, np.int8, 4)
        header['cmap'] = ''.join([chr(item) for item in strbits])
        header['stamp'] = np.fromfile(h, np.int8, 4)
        header['rms'] = np.fromfile(h, np.float32, 1)[0]
        header['nlabl'] = np.fromfile(h, np.uint32, 1)[0]
        strbits = np.fromfile(h, np.int8, 800)
        header['text'] = ''.join([chr(item) for item in strbits])
        header['ext_header'] = np.fromfile(h, np.int16, int(header['nextra'] / 2))
    return header
