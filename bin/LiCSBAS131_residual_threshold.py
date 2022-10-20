#!/usr/bin/env python3
"""
v1.0 20220928 Qi Ou, Leeds Uni

This script
(1) calculates a histogram of each residual map converted into factors of 2pi radian,
(2) offsets the residual map by the histogram peak to remove any bias from referencing effect
(3) calculate and saves the RMS of the de-peaked residual as multiples of 2pi [.txt]
(4) plots and saves a histogram[.png] of all RMS of all ifgs
(5) sets a threshold [.txt] for automatic correction with 132_3D_correction.py

===============
Input & output files
===============

Inputs in GEOCml*/ :
 - slc.mli.par

Inputs in TS_GEOCml*/ :
 - 13resid/
   - yyyymmdd_yyyymmdd.res

Outputs in TS_GEOCml*/ :
 - info/
   - 131resid_2pi{suffix}.txt        : RMS of the de-peaked residuals as factors of 2pi radian
   - 131RMS_ifg_res_hist{suffix}.png : plot of histogram with a vertical bar indicating threshold
 """
# - 131ref_de-peaked.txt    : reference point chosen as the pixel with minimum residual


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import LiCSBAS_io_lib as io_lib


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect coregistration error")
    parser.add_argument('-f', "--frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-g', '--unw_dir', default="GEOCml10GACOS", help="folder containing slc.mli.par")
    parser.add_argument('-t', '--ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-p', '--percentile', default="80", type=float, help="percentile RMS for thresholding")
    parser.add_argument('--input_suffix', default="", type=str, help="suffix of the input")
    parser.add_argument('--output_suffix', default="", type=str, help="suffix of the output")
    args = parser.parse_args()

    # define input directories
    unwdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    resdir = os.path.join(tsadir, '13resid'+args.input_suffix)

    # define output directories
    infodir = os.path.join(tsadir, 'info')

    # read ifg size and satellite frequency
    mlipar = os.path.join(unwdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency'))  # 5405000000.0 Hz for C-band
    speed_of_light = 299792458  # m/s
    wavelength = speed_of_light/radar_frequency
    coef_r2m = -wavelength/4/np.pi*1000


    # resultsdir = os.path.join(tsadir, 'results'+args.out_suffix)
    # if os.path.exists(resultsdir): os.remove(resultsdir)

    # netdir = os.path.join(tsadir, 'network')

    # with open(os.path.join(infodir, '13parameters.txt'), 'r') as f:
    #     for line in f.readlines():
    #         if 'range_samples' in line:
    #             range_samples = int(line.split()[1])
    #         if 'azimuth_lines' in line:
    #             azimuth_lines = int(line.split()[1])

    # #%% Start finding low residual ref point
    # sumsq_de_peaked_res = np.zeros((length, width), dtype=np.float32)

    print('Reading residual maps from {}'.format(resdir))
    restxtfile = os.path.join(infodir, '131resid_2pi{}.txt'.format(args.output_suffix))
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

        count_ifg_res_rms, bin_edges, patches = plt.hist(res_rms_list, np.arange(0, 3, 0.1))
        peak_ifg_res_rms = bin_edges[count_ifg_res_rms.argmax()]+0.05
        threshold = np.nanpercentile(res_rms_list, args.percentile)
        plt.axvline(x=peak_ifg_res_rms, color='r', linestyle='--')
        plt.axvline(x=threshold, color='r')
        plt.title("Residual, peak = {:2f}, {}% = {:2f}".format(peak_ifg_res_rms, int(args.percentile), threshold))
        plt.savefig(infodir+"/131RMS_ifg_res_hist{}.png".format(args.output_suffix), dpi=300)
        
        print('RMS_peak: {:5.2f}'.format(peak_ifg_res_rms), file=f)
        print('RMS_percentile: {}'.format(int(args.percentile), ), file=f)
        print('RMS_thresh: {:5.2f}'.format(threshold), file=f)

        print('IFG RMS res, peak = {:2f}, {}% = {:2f}'.format(peak_ifg_res_rms, int(args.percentile), threshold))

        # # calculate rms de-peaked residuals
        # for i in range(len(res_rms_list)):
        #     sumsq_de_peaked_res += abs(res_num_2pi.reshape((length, width)))
        #     # print(res_num_2pi.reshape((azimuth_lines, range_samples))[500:505, 249:253])
        #     # print(sumsq_de_peaked_res[500:505, 249:253])
    # fig, ax = plt.subplots(1, 3)
    # calculate rms de-peaked residuals
    # rms_de_peaked_res = np.sqrt(sumsq_de_peaked_res/len(res_rms_list))
    # vmin = np.nanpercentile(rms_de_peaked_res, 1)
    # vmax = np.nanpercentile(rms_de_peaked_res, 95)
    # im = ax[0].imshow(rms_de_peaked_res, vmin=vmin, vmax=vmax)
    # ax[0].set_title("RMS_de-peaked_Res")
    # ### Mask residual by minimum n_gap
    # n_gap = io_lib.read_img(os.path.join(resultsdir, 'n_gap'), length, width)
    # min_n_gap = np.nanmin(n_gap)
    # mask_n_gap = np.float32(n_gap==min_n_gap)
    # mask_n_gap[mask_n_gap == 0] = np.nan
    # rms_de_peaked_res = rms_de_peaked_res*mask_n_gap
    # mask=ax[1].imshow(mask_n_gap)
    # ax[1].set_title("n_gap")
    # ax[2].imshow(rms_de_peaked_res, vmin=vmin, vmax=vmax)
    # ax[2].set_title("Masked RMS")
    # plt.colorbar(im, ax=ax, orientation='horizontal')
    # plt.colorbar(mask, ax=ax[1], orientation='horizontal')
    # print(min_n_gap)

    # ### Find stable reference
    # min_rms = np.nanmin(rms_de_peaked_res)
    # refy1s, refx1s = np.where(rms_de_peaked_res == min_rms)
    # print(refy1s, refx1s, min_rms)
    # # print(rms_de_peaked_res[refy1s[0]-5:refy1s[0]+5, refx1s[0]-5:refx1s[0]+5])
    # ax[2].scatter(refx1s, refy1s)
    # ax[2].scatter(refx1s[0], refy1s[0])
    # fig.savefig(infodir+"/131RMS_ifg_res_ref.png", dpi=300)
    # refy1s, refx1s = refy1s[0], refx1s[0]  ## Only first index
    # refy2s, refx2s = refy1s+1, refx1s+1
    # print('Selected ref: {}:{}/{}:{}'.format(refx1s, refx2s, refy1s, refy2s), flush=True)
    #
    # ### Save ref
    # refsfile = os.path.join(infodir, '131ref_de-peaked.txt')
    # with open(refsfile, 'w') as f:
    #     print('{}:{}/{}:{}'.format(refx1s, refx2s, refy1s, refy2s), file=f)
