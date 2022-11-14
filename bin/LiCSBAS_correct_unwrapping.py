#!/usr/bin/env python3

#################
# plot unw with non-cyclic linear colour bar,
# plot connected components output by SNAPHU,
# plot residual in radian divided by 2pi and rounded to the nearest integer,
# plot a histogram of residual in radian divided by 2pi
# correct each component by the mode of nearest integer in that component
# run from inside the 13resid folder of LiCSBAS output with "../info/slc.mli.par" pointing to a text file containing range samples and azimuth lines
# Written by Qi Ou, University Leeds, 15 Aug 2022
#################

from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
import glob
import argparse
from pathlib import Path
#import cmcrameri as cm
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


parser = argparse.ArgumentParser(description="Detect coregistration error")
parser.add_argument('-f', "--frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
parser.add_argument('-g', '--GEOCml_dir', dest="unw_dir", default="GEOCml10GACOS", help="folder containing unw input to time series")
parser.add_argument('-t', '--ts_dir', dest="ts_dir", default="TS_GEOCml10GACOS", help="folder containing unw input to time series")
args = parser.parse_args()

speed_of_light = 299792458  # m/s
radar_frequency = 5405000000.0  # Hz
wavelength = speed_of_light/radar_frequency
coef_r2m = -wavelength/4/np.pi*1000

with open('{}/{}/info/13parameters.txt'.format(args.frame_dir, args.ts_dir), 'r') as f:
    for line in f.readlines():
        if 'range_samples' in line:
            range_samples = int(line.split()[1])
        if 'azimuth_lines' in line:
            azimuth_lines = int(line.split()[1])

with open('{}/{}/info/12ref.txt'.format(args.frame_dir, args.ts_dir), 'r') as f:
    for line in f.readlines():
        ref_x_step12 = int(line.split(":")[0])
        ref_y_step12 = int(line.split("/")[1].split(":")[0])

stats_file = '{}/{}/info/13_residual_component_mode_stats.txt'.format(args.frame_dir, args.ts_dir)
if os.path.exists(stats_file):
    os.remove(stats_file)

with open(stats_file, 'w') as f:

    print("IFG,  RMS of residual in number of 2pi,  % pixels to correct, % pixels needing multiple corrections")
    f.write("IFG,  RMS of residual in number of 2pi,  % pixels to correct, % pixels needing multiple corrections \n")

    for i in glob.glob('{}/{}/13resid/*.res'.format(args.frame_dir, args.ts_dir)) :
        pair = os.path.basename(i).split('.')[0][-17:]
        print(pair)
        con = np.fromfile(os.path.join(args.frame_dir, args.unw_dir, pair, pair+'.conncomp'), dtype=np.int8).reshape((azimuth_lines, range_samples))
        unw = np.fromfile(os.path.join(args.frame_dir, args.unw_dir, pair, pair+'.unw'), dtype=np.float32).reshape((azimuth_lines, range_samples))

        res_mm = np.fromfile(i, dtype=np.float32).reshape((azimuth_lines, range_samples))
        res_rad = res_mm / coef_r2m
        res_num_2pi = res_rad / 2 / np.pi
        res_integer = np.round(res_num_2pi)
        res_rms = np.sqrt(np.nanmean(res_num_2pi**2))

        res_mode = copy.copy(res_integer)
        for j in np.unique(con.flatten())[1:]:
            component_values = res_integer[con == j]
            int_values = component_values[~np.isnan(component_values)].astype(int)
            mode = stats.mode(int_values)[0][0]
            res_mode[con == j] = mode

        res_mode[np.isnan(res_integer)] = np.nan

        unw_corrected = unw - res_mode * 2 * np.pi
        #unw_corrected_by_int = unw - res_integer * 2 * np.pi
        unw_masked = copy.copy(unw)
        unw_masked[res_integer != 0] = np.nan

        fig, ax = plt.subplots(3, 3, figsize=(9, 6))
        fig.suptitle(pair)
        for x in ax[:2, :].flatten():
            x.axes.xaxis.set_ticklabels([])
            x.axes.yaxis.set_ticklabels([])
        ax[2, 0].axes.xaxis.set_ticklabels([])
        ax[2, 0].axes.yaxis.set_ticklabels([])
        ax[2, 2].set_axis_off()

        unw_vmin = np.nanpercentile(unw, 0.5)
        unw_vmax = np.nanpercentile(unw, 99.5)

        im_unw = ax[0,0].imshow(unw, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu)
        im_unw = ax[0,1].imshow(unw_masked, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu)
        im_unw = ax[0,2].imshow(unw_corrected, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu)
        im_res = ax[1,0].imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu)
        im_res = ax[1,1].imshow(res_integer, vmin=-2, vmax=2, cmap=cm.RdBu)
        im_res = ax[1,2].imshow(res_mode, vmin=-2, vmax=2, cmap=cm.RdBu)
        ax[1,0].scatter(ref_x_step12, ref_y_step12, c='r', s=10)
        im_con = ax[2,0].imshow(con, cmap=cm.tab10, interpolation='none')

        ax[0,0].set_title("Unw (rad)")
        ax[0,1].set_title("Unw_masked")
        ax[0,2].set_title("Unw_corrected")
        ax[1,0].set_title("Residual/2pi (RMS={:.2f})".format(res_rms))
        ax[1,1].set_title("Nearest integer")
        ax[1,2].set_title("Component mode")
        ax[2,0].set_title("Components")

        fig.colorbar(im_unw, ax=ax[0,:], location='right', shrink=0.8)
        fig.colorbar(im_res, ax=ax[1,:], location='right', shrink=0.8)
        fig.colorbar(im_con, ax=ax[2,0], location='right', shrink=0.8)

        ax[2,1].hist(res_num_2pi.flatten(), np.arange(-2.5, 3.5, 1))
        ax[2,1].hist(res_num_2pi.flatten(), np.arange(-2.5, 2.6, 0.1))
        ax[2,1].set_yscale("log")
        ax[2,1].set_xlabel("Residual (rad) / 2 pi")
        ax[2,1].set_ylabel("Pixel Count")
        ax[2,1].yaxis.set_label_position("right")
        ax[2,1].yaxis.tick_right()

        nonnan_res_integer = res_mode[~np.isnan(res_mode)]
        nonnan_pixel_number = len(nonnan_res_integer)
        number_pixels_to_correct = len(nonnan_res_integer[abs(nonnan_res_integer)>0.9])
        number_pixels_to_correct_by_2cycles = len(nonnan_res_integer[abs(nonnan_res_integer)>1.9])
        percentage_pixel_to_correct = number_pixels_to_correct / nonnan_pixel_number * 100
        percentage_pixel_to_correct_by_2cycles = number_pixels_to_correct_by_2cycles / nonnan_pixel_number * 100
        ax[2, 1].set_title("{}% to correct, {}% >1 cycle".format(int(percentage_pixel_to_correct), int(percentage_pixel_to_correct_by_2cycles)))

        plt.savefig('{}/{}/13resid/{}_unw_mask_mode.png'.format(args.frame_dir, args.ts_dir, pair), dpi=300)
        plt.close()

        Path("{}/{}_corrected/{}/".format(args.frame_dir, args.unw_dir, pair)).mkdir(parents=True, exist_ok=True)
        unw_corrected.flatten().tofile("{}/{}_corrected/{}/{}.unw".format(args.frame_dir, args.unw_dir, pair, pair))

        print(pair,res_rms, "{:.2f}".format(percentage_pixel_to_correct), "{:.2f}".format(percentage_pixel_to_correct_by_2cycles) )
        f.write("{} {:.2f} {:.2f} {:.2f} \n".format(pair, res_rms, percentage_pixel_to_correct, percentage_pixel_to_correct_by_2cycles))

        del con, unw, unw_corrected, unw_masked, res_num_2pi, res_integer, nonnan_res_integer
