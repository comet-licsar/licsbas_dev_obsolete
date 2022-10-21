#!/usr/bin/env python3
"""
v1.0 20220928 Qi Ou, Leeds Uni

plot unw with non-cyclic linear colour bar,
plot connected components output by SNAPHU,
plot residual in radian divided by 2pi and rounded to the nearest integer,
plot a histogram of residual in radian divided by 2pi
correct each component by the mode of nearest integer in that component
run from inside the 13resid folder of LiCSBAS output with "../info/slc.mli.par" pointing to a text file containing range samples and azimuth lines

===============
Input & output files
===============

Inputs in GEOCml*/ (--comp_cc_dir):
 - baselines
 - slc.mli.par
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.conncomp
   - yyyymmdd_yyyymmdd.cc

Inputs in GEOCml*/ (--unw_dir):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw

Inputs in TS_GEOCml*/ :
 - 13resid*/ (--resid_dir)
   - yyyymmdd_yyyymmdd.res
 - info/
   - 131resid_2pi*.txt     : RMS residuals per IFG computed in radian and as a factor of 2pi

Outputs in TS_GEOCml*/ :
 - 13resid*/ (--resid_dir)
   - good_ifg_no_correction/*png : residuals of uncorrected ifgs
   - bad_ifg_no_correction/*png  : residuals and nearest integers showing why ifgs can't be corrected
   - integer_correction/*png     : ifgs corrected by nearest residual integer (when mode doesn't work)
   - mode_correction/*png        : ifgs corrected by component mode (preferred)
 - info/
   - 132*.txt          : Lists of ifgs corrected by each category (good, bad, integer_corrected, mode_corrected)
 - network/
   - network132*.png   : Figures of the network with/without corrected ifgs, with bad ifgs removed
 """

from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
import glob
import argparse
import sys
import time
from pathlib import Path
#import cmcrameri as cm
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
import shutil


def init_args():
    global args

    parser = argparse.ArgumentParser(description="Detect coregistration error")
    parser.add_argument('-f', "--frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-c', '--comp_cc_dir', default="GEOCml10GACOS", help="folder containing connected components and cc files")
    parser.add_argument('-g', '--unw_dir', default="GEOCml10GACOS", help="folder containing unw input to be corrected")
    parser.add_argument('-r', '--correct_dir', default="GEOCml10GACOS_corrected", help="folder containing corrected unw input")
    parser.add_argument('-t', '--ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('--thresh', default=0.5, help="threshold RMS residual per ifg as a fraction of 2 pi radian, used if info/131resid_2pi.txt doesn't exist")
    parser.add_argument('--suffix', default="", type=str, help="suffix of the input 131resid_2pi*.txt and outputs")
    args = parser.parse_args()


def start():
    # intialise and print info on screen
    start = time.time()
    ver="1.0"; date=20221020; author="Qi Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    return start


def finish(start_time):
    # %% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time / 3600)
    minite = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minite, sec))
    print('\n{} {} Successfully finished!!\n'.format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])))
    print('Output directory: {}\n'.format(os.path.relpath(correct_dir)))


def set_input_output():
    global ccdir, unwdir, tsadir, resdir, infodir, netdir, correct_dir, good_png_dir, bad_png_dir, integer_png_dir, mode_png_dir

    # define input directories
    ccdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir))
    unwdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    resdir = os.path.join(tsadir, '130resid{}'.format(args.suffix))
    infodir = os.path.join(tsadir, 'info')

    # define output directories
    netdir = os.path.join(tsadir, 'network')

    correct_dir = os.path.abspath(os.path.join(args.frame_dir, args.correct_dir))
    if os.path.exists(correct_dir): shutil.rmtree(correct_dir)
    Path(correct_dir).mkdir(parents=True, exist_ok=True)

    good_png_dir = os.path.join(resdir, 'good_ifg_no_correction/')
    if os.path.exists(good_png_dir): shutil.rmtree(good_png_dir)
    Path(good_png_dir).mkdir(parents=True, exist_ok=True)

    bad_png_dir = os.path.join(resdir, 'bad_ifg_no_correction/')
    if os.path.exists(bad_png_dir): shutil.rmtree(bad_png_dir)
    Path(bad_png_dir).mkdir(parents=True, exist_ok=True)

    integer_png_dir = os.path.join(resdir, 'integer_correction/')
    if os.path.exists(integer_png_dir): shutil.rmtree(integer_png_dir)
    Path(integer_png_dir).mkdir(parents=True, exist_ok=True)

    mode_png_dir = os.path.join(resdir, 'mode_correction/')
    if os.path.exists(mode_png_dir): shutil.rmtree(mode_png_dir)
    Path(mode_png_dir).mkdir(parents=True, exist_ok=True)



def get_para():
    global width, length, coef_r2m, thresh, ref_x, ref_y

    # read ifg size and satellite frequency
    mlipar = os.path.join(ccdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency'))  # 5405000000.0 Hz for C-band
    speed_of_light = 299792458  # m/s
    wavelength = speed_of_light/radar_frequency
    coef_r2m = -wavelength/4/np.pi*1000

    # read threshold value
    resid_threshold_file = os.path.join(infodir, '131resid_2pi{}.txt'.format(args.suffix))
    if os.path.exists(resid_threshold_file):
        thresh = float(io_lib.get_param_par(resid_threshold_file, 'RMS_thresh'))
    else:
        thresh = args.thresh
    print("Correction threshold = {:2f}".format(thresh))

    # read reference for plotting purpose
    with open(os.path.join(infodir, '120ref.txt'), 'r') as f:
        for line in f.readlines():
            ref_x = int(line.split(":")[0])
            ref_y = int(line.split("/")[1].split(":")[0])


def correction_decision():
    global bad_ifg_not_corrected, ifg_corrected_by_mode, ifg_corrected_by_integer, good_ifg

    # set up empty ifg lists
    good_ifg = []
    ifg_corrected_by_mode = []
    ifg_corrected_by_integer = []
    bad_ifg_not_corrected = []

    # automatic correction
    for i in glob.glob(os.path.join(resdir, '*.res')):
        # read input res
        pair = os.path.basename(i).split('.')[0][-17:]
        print(pair)
        res_mm = np.fromfile(i, dtype=np.float32).reshape((length, width))
        res_rad = res_mm / coef_r2m
        res_num_2pi = res_rad / 2 / np.pi
        counts, bins = np.histogram(res_num_2pi.flatten(), np.arange(-2.5, 2.6, 0.1))
        peak = bins[counts.argmax()] + 0.05
        res_num_2pi = res_num_2pi - peak
        res_rms = np.sqrt(np.nanmean(res_num_2pi ** 2))

        if res_rms < thresh:
            good_ifg.append(pair)
            print("RMS residual = {:.2f}, good...".format(res_rms))

            # define output dir
            correct_pair_dir = os.path.join(correct_dir, pair)
            Path(correct_pair_dir).mkdir(parents=True, exist_ok=True)

            # Link unw
            unwfile = os.path.join(unwdir, pair, pair + '.unw')
            linkfile = os.path.join(correct_pair_dir, pair + '.unw')
            os.link(unwfile, linkfile)
            # relative_path = os.path.relpath(unwfile, correct_unw_dir+'/')
            # os.symlink(relative_path, linkfile)
            # shutil.copy(unwfile, correct_unw_dir)

            ## plot_res
            plt.imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
            plt.title(pair + " RMS_res={:.2f}".format(res_rms))
            plt.colorbar()
            png_path = os.path.join(resdir, 'good_ifgs/')
            plt.tight_layout()
            plt.savefig(good_png_dir + '{}.png'.format(pair), dpi=300, bbox_inches='tight')
            plt.close()

            del res_num_2pi, res_mm, res_rad, res_rms

        else:
            print("RMS residual = {:2f}, not good...".format(res_rms))
            res_integer = np.round(res_num_2pi)
            rms_res_integer_corrected = np.sqrt(np.nanmean((res_num_2pi - res_integer) ** 2))
            if rms_res_integer_corrected > thresh:
                bad_ifg_not_corrected.append(pair)
                print("Integer reduces rms residuals to {:.2f}, still above threshold of {:.2f}, discard...".format(
                    rms_res_integer_corrected, thresh))

                ## plot_res
                fig, ax = plt.subplots(1, 2, figsize=(9, 6))
                fig.suptitle(pair)
                for x in ax:
                    x.axes.xaxis.set_ticklabels([])
                    x.axes.yaxis.set_ticklabels([])
                im_res = ax[0].imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
                im_res = ax[1].imshow(res_integer, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
                ax[0].scatter(ref_x, ref_y, c='r', s=10)
                ax[0].set_title("Residual/2pi (RMS={:.2f})".format(res_rms))
                ax[1].set_title("Nearest integer")
                plt.colorbar(im_res, ax=ax, location='right', shrink=0.8)
                plt.savefig(bad_png_dir + '{}.png'.format(pair), dpi=300, bbox_inches='tight')
                plt.close()
                del res_num_2pi, res_mm, res_rad, res_rms, res_integer, rms_res_integer_corrected

            else:
                # read in unwrapped ifg and connected components
                unwfile = os.path.join(unwdir, pair, pair + '.unw')
                con_file = os.path.join(ccdir, pair, pair + '.conncomp')
                unw = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
                con = np.fromfile(con_file, dtype=np.int8).reshape((length, width))

                # calculate component modes
                uniq_components = np.unique(con.flatten())[1:]  # use [1:] to exclude the 0th component
                res_mode = copy.copy(res_integer)
                for j in uniq_components:
                    component_values = res_integer[con == j]
                    int_values = component_values[~np.isnan(component_values)].astype(int)
                    mode = stats.mode(int_values)[0][0]
                    res_mode[con == j] = mode

                # check if component modes does a good job
                rms_res_mode_corrected = np.sqrt(np.nanmean((res_num_2pi - res_mode) ** 2))

                # if component mode is useful
                if rms_res_mode_corrected < thresh:
                    print(
                        "Component modes reduces rms residuals to {:.2f}, below threshold of {:.2f}, correcting by component mode...".format(
                            rms_res_mode_corrected, thresh))
                    unw_corrected = unw - res_mode * 2 * np.pi
                    correction_title = "Mode_corrected"
                    ifg_corrected_by_mode.append(pair)
                    png_path = os.path.join(mode_png_dir, '{}.png'.format(pair))

                else:  # if component mode is not useful
                    print("Component modes reduces rms residuals to {:.2f}, above threshold of {:.2f}...".format(
                        rms_res_mode_corrected, thresh))
                    print("Integer reduces rms residuals to {:.2f}, correcting by nearest integer...".format(
                        rms_res_integer_corrected))
                    unw_corrected = unw - res_integer * 2 * np.pi
                    correction_title = "Integer_corrected"
                    ifg_corrected_by_integer.append(pair)
                    png_path = os.path.join(integer_png_dir, '{}.png'.format(pair))

                plot_correction(pair, unw, con, unw_corrected, res_num_2pi, res_integer, res_mode, correction_title, res_rms, png_path)

                # define output dir
                correct_pair_dir = os.path.join(correct_dir, pair)
                Path(correct_pair_dir).mkdir(parents=True, exist_ok=True)

                # save the corrected unw
                unw_corrected.flatten().tofile(os.path.join(correct_pair_dir, pair + '.unw'))
                del con, unw, unw_corrected, res_num_2pi, res_integer, res_mm, res_rad, res_rms, correction_title



def plot_correction(pair, unw, con, unw_corrected, res_num_2pi, res_integer, res_mode, correction_title, res_rms, png_path):
    fig, ax = plt.subplots(2, 3, figsize=(9, 5))
    fig.suptitle(pair)
    for x in ax[:, :].flatten():
        x.axes.xaxis.set_ticklabels([])
        x.axes.yaxis.set_ticklabels([])
    unw_vmin = np.nanpercentile(unw, 0.5)
    unw_vmax = np.nanpercentile(unw, 99.5)
    im_con = ax[0, 0].imshow(con, cmap=cm.tab10, interpolation='nearest')
    im_unw = ax[0, 1].imshow(unw, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_unw = ax[0, 2].imshow(unw_corrected, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 0].imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 1].imshow(res_integer, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 2].imshow(res_mode, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    ax[1, 0].scatter(ref_x, ref_y, c='r', s=10)
    ax[0, 0].set_title("Components")
    ax[0, 1].set_title("Unw (rad)")
    ax[0, 2].set_title(correction_title)
    ax[1, 0].set_title("Residual/2pi (RMS={:.2f})".format(res_rms))
    ax[1, 1].set_title("Nearest integer")
    ax[1, 2].set_title("Component mode")
    # fig.colorbar(im_con, ax=ax[0, 0], location='right', shrink=0.8)
    fig.colorbar(im_unw, ax=ax[0, :], location='right', shrink=0.8)
    fig.colorbar(im_res, ax=ax[1, :], location='right', shrink=0.8)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_lists():

    #%% save ifg lists to text files.
    bad_ifg_file = os.path.join(infodir, '132bad_ifg{}.txt'.format(args.suffix))
    if os.path.exists(bad_ifg_file): os.remove(bad_ifg_file)
    with open(bad_ifg_file, 'w') as f:
        for i in bad_ifg_not_corrected:
            print('{}'.format(i), file=f)

    mode_ifg_file = os.path.join(infodir, '132corrected_by_component_mode_ifg{}.txt'.format(args.suffix))
    if os.path.exists(mode_ifg_file): os.remove(mode_ifg_file)
    with open(mode_ifg_file, 'w') as f:
        for i in ifg_corrected_by_mode:
            print('{}'.format(i), file=f)

    nearest_ifg_file = os.path.join(infodir, '132corrected_by_nearest_integer_ifg{}.txt'.format(args.suffix))
    if os.path.exists(nearest_ifg_file): os.remove(nearest_ifg_file)
    with open(nearest_ifg_file, 'w') as f:
        for i in ifg_corrected_by_integer:
            print('{}'.format(i), file=f)

    good_ifg_file = os.path.join(infodir, '132good_ifg_uncorrected{}.txt'.format(args.suffix))
    if os.path.exists(good_ifg_file): os.remove(good_ifg_file)
    with open(good_ifg_file, 'w') as f:
        for i in good_ifg:
            print('{}'.format(i), file=f)


def plot_networks():

    ### Read date, network information and size
    retained_ifgs = good_ifg + ifg_corrected_by_mode + ifg_corrected_by_integer
    corrected_ifgs = ifg_corrected_by_mode + ifg_corrected_by_integer
    retained_ifgs.sort()
    corrected_ifgs.sort()
    imdates = tools_lib.ifgdates2imdates(retained_ifgs)
    n_im = len(imdates)

    ### Plot network
    ## Read bperp data or dummy
    bperp_file = os.path.join(ccdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else: #dummy
        bperp = np.random.random(n_im).tolist()

    pngfile = os.path.join(netdir, 'network132_only_good_without_correction{}.png'.format(args.suffix))
    plot_lib.plot_corrected_network(retained_ifgs, bperp, corrected_ifgs, pngfile, plot_bad=False)

    pngfile = os.path.join(netdir, 'network132_with_corrected{}.png'.format(args.suffix))
    plot_lib.plot_corrected_network(retained_ifgs, bperp, corrected_ifgs, pngfile)

    pngfile = os.path.join(netdir, 'network132_all{}.png'.format(args.suffix))
    plot_lib.plot_corrected_network(retained_ifgs, bperp, [], pngfile)


def main():
    start_time = start()
    init_args()
    set_input_output()
    get_para()
    correction_decision()
    save_lists()
    plot_networks()
    finish(start_time)


if __name__ == "__main__":
    main()