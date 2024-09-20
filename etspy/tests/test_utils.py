import glob
import os

import hyperspy.api as hs
import numpy
import pytest

import etspy
from etspy import datasets as ds
from etspy import io, utils
from etspy.base import TomoStack

etspy_path = os.path.dirname(etspy.__file__)


def hspy_mrc_reader_check():
    dirname = os.path.join(etspy_path, "tests", "test_data", "SerialEM_Multiframe_Test")
    files = glob.glob(dirname + "/*.mrc")
    file = files[0]
    s = hs.load(file)
    return s


try:
    hspy_mrc_reader_check()
except TypeError:
    hspy_mrc_broken = True
else:
    hspy_mrc_broken = False


@pytest.mark.skipif(hspy_mrc_broken is True, reason="Hyperspy MRC reader broken")
class TestMultiframeAverage:
    def test_register_serialem_stack(self):
        dirname = os.path.join(
            etspy_path, "tests", "test_data", "SerialEM_Multiframe_Test"
        )
        files = glob.glob(dirname + "/*.mrc")
        stack = io.load(files)
        stack_avg = utils.register_serialem_stack(stack, ncpus=1)
        assert type(stack_avg) is TomoStack
        assert stack_avg.data.shape[0] == 3

    def test_register_serialem_stack_multicpu(self):
        dirname = os.path.join(
            etspy_path, "tests", "test_data", "SerialEM_Multiframe_Test"
        )
        files = glob.glob(dirname + "/*.mrc")
        stack = io.load(files)
        stack_avg = utils.register_serialem_stack(stack, ncpus=2)
        assert type(stack_avg) is TomoStack
        assert stack_avg.data.shape[0] == 3

    def test_multiaverage(self):
        dirname = os.path.join(
            etspy_path, "tests", "test_data", "SerialEM_Multiframe_Test"
        )
        files = glob.glob(dirname + "/*.mrc")
        stack = io.load(files)
        ntilts, nframes, ny, nx = stack.data.shape
        stack_avg = utils.multiaverage(stack.data[0], nframes, ny, nx)
        assert type(stack_avg) is numpy.ndarray
        assert stack_avg.shape == (ny, nx)


class TestWeightStack:
    def test_weight_stack_low(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        reg = utils.weight_stack(stack, accuracy="low")
        assert type(reg) is TomoStack

    def test_weight_stack_medium(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        reg = utils.weight_stack(stack, accuracy="medium")
        assert type(reg) is TomoStack

    def test_weight_stack_high(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        reg = utils.weight_stack(stack, accuracy="high")
        assert type(reg) is TomoStack

    def test_weight_stack_bad_accuracy(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        with pytest.raises(ValueError):
            utils.weight_stack(stack, accuracy="wrong")


class TestHelperUtils:
    def test_est_angles(self):
        est = utils.calc_EST_angles(10)
        assert type(est) is numpy.ndarray
        assert est.shape[0] == 20

    def test_est_angles_error(self):
        with pytest.raises(ValueError):
            utils.calc_EST_angles(11)

    def test_golden_ratio_angles(self):
        gr = utils.calc_golden_ratio_angles(10, 5)
        assert type(gr) is numpy.ndarray
        assert gr.shape[0] == 5

    def test_radial_mask_no_center(self):
        mask = utils.get_radial_mask([100, 100], None)
        assert type(mask) is numpy.ndarray
        assert mask.shape == (100, 100)

    def test_radial_mask_with_center(self):
        mask = utils.get_radial_mask([100, 100], [50, 50])
        assert type(mask) is numpy.ndarray
        assert mask.shape == (100, 100)


class TestWeightingFilter:
    def test_weighting_filter_shepp_logan(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        filtered = utils.filter_stack(stack, filter_name="shepp-logan", cutoff=0.5)
        assert type(filtered) is TomoStack

    def test_weighting_filter_ram_lak(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        filtered = utils.filter_stack(stack, filter_name="ram-lak", cutoff=0.5)
        assert type(filtered) is TomoStack

    def test_weighting_filter_cosine(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        filtered = utils.filter_stack(stack, filter_name="cosine", cutoff=0.5)
        assert type(filtered) is TomoStack

    def test_weighting_filter_shepp_hanning(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        filtered = utils.filter_stack(stack, filter_name="hanning", cutoff=0.5)
        assert type(filtered) is TomoStack

    def test_weighting_filter_two_dimensional_data(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0]
        filtered = utils.filter_stack(stack, filter_name="hanning", cutoff=0.5)
        assert type(filtered) is TomoStack

    def test_weighting_filter_bad_filter(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:3]
        with pytest.raises(ValueError):
            utils.filter_stack(stack, filter_name="wrong", cutoff=0.5)
