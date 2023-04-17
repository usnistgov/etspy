# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Utility module for TomoTools package.

@author: Andrew Herzing
"""

import numpy as np
import hyperspy.api as hspy
from tomotools.io import numpy_to_tomo_stack
from tomotools.base import TomoStack


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


def parse_mdoc(mdoc_file):
    """
    Parse a SerialEM mdoc file.

    Args
    ----------
    mdoc_file : str
        Name of the mdoc file to parse

    Returns
    ----------
    metadata : dict
        Dictionary with values parsed from mdoc file

    """
    keys = ['PixelSpacing', 'Voltage', 'ImageFile', 'Image Size', 'DataMode',
            'TiltAngle', 'Magnification', 'ExposureTime', 'SpotSize', 'Defocus']
    metadata = {}
    with open(mdoc_file, 'r') as f:
        for i in range(0, 35):
            line = f.readline()
            for k in keys:
                if k in line:
                    if k == 'ImageFile':
                        metadata[k] = line.split('=')[1].strip()
                    else:
                        metadata[k] = float(line.split('=')[1].strip())
    return metadata


def read_serialem_series(mrcfiles, mdocfiles):
    """
    Load a multi-frame series collected by SerialEM.

    Parameters
    ----------
    mrc_files : list of files
        MRC files containing multi-frame tilt series data.

    mdoc_files : list of files
        SerialEM metadata files for multi-frame tilt series data.

    align : bool
        If True, align the frames using PyStackReg at each tilt prior to summing.

    Returns
    ----------
    stack : TomoStack object
        Tilt series resulting by summing frames at each tilt

    """

    stack = [None] * len(mrcfiles)
    meta = [None] * len(mdocfiles)
    for i in range(0, len(mdocfiles)):
        meta[i] = parse_mdoc(mdocfiles[i])

    tilts = np.array([meta[i]['TiltAngle'] for i in range(0, len(meta))])
    tilts_sort = np.argsort(tilts)
    tilts.sort()

    for i in range(0, len(mrcfiles)):
        fn = mdocfiles[tilts_sort[i]][:-5]
        stack[i] = hspy.load(fn)

    images_per_tilt = stack[0].data.shape[0]
    stack = hspy.stack(stack)
    stack.axes_manager[1].scale = tilts[1] - tilts[0]
    stack.axes_manager[1].offset = tilts[0]
    stack.axes_manager[1].units = 'degrees'
    stack.axes_manager[1].units = 'Tilt'
    stack.axes_manager[2].scale = meta[0]['PixelSpacing'] / 10
    stack.axes_manager[3].scale = meta[0]['PixelSpacing'] / 10
    stack.axes_manager[2].units = 'nm'
    stack.axes_manager[3].units = 'nm'
    stack.axes_manager[2].name = 'y'
    stack.axes_manager[3].name = 'x'

    if not stack.metadata.has_item('Acquisition_instrument.TEM'):
        stack.metadata.add_node('Acquisition_instrument.TEM')
    stack.metadata.Acquisition_instrument.TEM.magnification = meta[0]['Magnification']
    stack.metadata.Acquisition_instrument.TEM.beam_energy = meta[0]['Voltage']
    stack.metadata.Acquisition_instrument.TEM.dwell_time = meta[0]['ExposureTime'] * images_per_tilt * 1e-6
    stack.metadata.Acquisition_instrument.TEM.spot_size = meta[0]['SpotSize']
    stack.metadata.Acquisition_instrument.TEM.defocus = meta[0]['Defocus']
    stack.metadata.General.original_filename = meta[0]['ImageFile']
    return stack


def register_serialem_stack(stack, method='ECC'):
    """
    Register a multi-frame series collected by SerialEM.

    Parameters
    ----------
    stack : Hyperspy Signal2D
        Signal of shape [ntilts, nframes, ny, nx].

    method : string
        Stack registration method to use.

    Returns
    ----------
    reg : TomoStack object
        Result of aligning and integrating frames at each tilt with shape [ntilts, ny, nx]

    """

    reg = np.zeros([stack.data.shape[0], stack.data.shape[2], stack.data.shape[3]], stack.data.dtype)
    for i in range(0, stack.data.shape[0]):
        temp = TomoStack(stack.data[i])
        reg[i, :, :] = temp.stack_register(method=method).data.sum(0)
    reg = numpy_to_tomo_stack(reg)
    return reg
