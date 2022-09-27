#!/usr/bin/env python3

#################
# Fit a gaussian distribution to the histogram of residuals in rad
# Determine the residual threshold for further ifgs to be removed
# Written by Qi Ou, University Leeds, 22 Sep 2022
#################

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from scipy.optimize import curve_fit


# Define the Gaussian function
def Gauss(x, a, b, c):
    """ a = height, b = mean (mu), c = sigma """
    y = a*np.exp(-(x-b)**2/(2*c**2))
    return y


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect coregistration error")
    parser.add_argument('-f', "--frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-g', '--GEOCml_dir', dest="unw_dir", default="GEOCml10GACOS", help="folder containing unw input")
    parser.add_argument('-t', '--ts_dir', dest="ts_dir", default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-r', '--rms_thresh', dest="thresh", default=0.3, help="threshold RMS residual per ifg as a fraction of 2 pi radian")
    args = parser.parse_args()

    speed_of_light = 299792458  # m/s
    radar_frequency = 5405000000.0  # Hz
    wavelength = speed_of_light / radar_frequency
    coef_r2m = -wavelength / 4 / np.pi * 1000

    unwdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    resultsdir = os.path.join(tsadir, 'results')
    infodir = os.path.join(tsadir, 'info')
    netdir = os.path.join(tsadir, 'network')
    resdir = os.path.join(tsadir, '13resid')

    restxtfile = os.path.join(infodir,'13resid_2pi.txt')
    if os.path.exists(restxtfile): os.remove(restxtfile)
    with open(restxtfile, "w") as f:
        print('# RMS of residual (in number of 2pi)', file=f)
        res_rms_list = []

        for i in glob.glob(os.path.join(resdir, '*.res')):
            pair = os.path.basename(i).split('.')[0][-17:]
            print(pair)
            res_mm = np.fromfile(i, dtype=np.float32)
            res_rad = res_mm / coef_r2m
            res_num_2pi = res_rad / 2 / np.pi
            counts, bins = np.histogram(res_num_2pi, np.arange(-2.5, 2.6, 0.1))
            peak = bins[counts.argmax()]+0.05
            res_num_2pi = res_num_2pi - peak
            res_rms = np.sqrt(np.nanmean(res_num_2pi**2))
            res_rms_list.append(res_rms)

            print('{} {:5.2f}'.format(pair, res_rms), file=f)

        count_ifg_res_rms, bin_edges, patches = plt.hist(res_rms_list, np.arange(0, 2, 0.1))
        bin_centers = bin_edges[1:] - bin_edges[:-1]
        parameters, covariance = curve_fit(Gauss, bin_centers, count_ifg_res_rms)
        fit_height = parameters[0]
        fit_mu = parameters[1]
        fit_sigma = parameters[2]
        fit_y = Gauss(bin_centers, fit_height, fit_mu, fit_sigma)
        threshold = fit_mu+fit_sigma
        plt.plot(bin_centers, fit_y, '-')
        plt.vline(x=threshold, color='r')
        plt.title("Residual, thresh = {:2f}".format(threshold))
        plt.savefig(resdir+"RMS_ifg_res_hist.png", dpi=300)

        print('RMS_threshold: {:5.2f}'.format(threshold), file=f)


