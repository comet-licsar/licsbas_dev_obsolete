#!/usr/bin/env python3
"""
v1.0 202201019 Qi Ou, University of Leeds

========
Overview
========
This script sets up an iterative loop to perform time series inversion and automatic unwrapping correction until time series reaches good quality

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw
   - yyyymmdd_yyyymmdd.cc
   - yyyymmdd_yyyymmdd.conncomp
 - EQA.dem_par
 - slc.mli.par
 - baselines (may be dummy)
[- [ENU].geo]

Inputs in TS_GEOCml*/ :
 - info/
   - 11bad_ifg.txt
   - 12bad_ifg.txt
   - 120ref.txt
[-results/]
[  - coh_avg]
[  - hgt]
[  - n_loop_err]
[  - n_unw]
[  - slc.mli]

Outputs in TS_GEOCml*/ :
 - cum*.h5             : Cumulative displacement (time-seires) in mm
 - results*/
   - vel[.png]        : Velocity in mm/yr (positive means LOS decrease; uplift)
   - vintercept[.png] : Constant part of linear velocity (c for vt+c) in mm
   - resid_rms[.png]  : RMS of residual in mm
   - n_gap[.png]      : Number of gaps in SB network
   - n_ifg_noloop[.png] :  Number of ifgs with no loop
   - maxTlen[.png]    : Max length of continous SB network in year
 - info/
   - 13parameters.txt : List of used parameters
   - 13used_image.txt : List of used images
   - 13resid.txt      : List of RMS of residual for each ifg
   - 13ref.txt[kml]   : Auto-determined stable ref point
   - 13rms_cum_wrt_med[.png] : RMS of cum wrt median used for ref selection
 - 13increment*/yyyymmdd_yyyymmdd.increment.png
     : Comparison between unw and inverted incremental displacement
 - 13resid*/yyyymmdd_yyyymmdd.res.png : Residual for each ifg
 - network/network13*.png : Figures of the network

"""
#%% Import
import os
import time
import shutil
import numpy as np
import h5py as h5
from pathlib import Path
import argparse
import sys
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib


def init_args():
    global args
    parser = argparse.ArgumentParser(description="Detect coregistration error")
    parser.add_argument('-f', "--frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-g', '--unw_dir', default="GEOCml10GACOS", help="folder containing unw input")
    parser.add_argument('-t', '--ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-p', '--percentile', default="80", type=float, help="percentile RMS for thresholding")
    parser.add_argument('--thresh', default="0.2", type=float, help="percentile RMS for thresholding")
    args = parser.parse_args()


def start():
    # intialise and print info on screen
    start = time.time()
    ver="1.0"; date=20221020; author="Qi Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    return start


def finish(start_time):
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print('\n{} {} Successfully finished!!\n'.format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])))
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


def set_input_output():
    global ifgdir, tsadir, infodir, ccdir
    # define input directories
    ccdir = args.unw_dir
    ifgdir = os.path.abspath(os.path.join(args.frame_dir, ccdir))  # to read .unw
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))   # to read 120.ref, to write cum.h5
    infodir = os.path.join(tsadir, 'info')  # to read 11bad_ifg.txt, 12bad_ifg.txt


def get_ifgdates():
    global ifgdates
    bad_ifg11file = os.path.join(infodir, '11bad_ifg.txt')
    bad_ifg12file = os.path.join(infodir, '120bad_ifg.txt')

    #%% Read date and network information
    ### Get all ifgdates in ifgdir
    ifgdates_all = tools_lib.get_ifgdates(ifgdir)

    ### Remove bad_ifg11 and 120bad_ifg
    bad_ifg11 = io_lib.read_ifg_list(bad_ifg11file)
    bad_ifg12 = io_lib.read_ifg_list(bad_ifg12file)
    bad_ifg_all = list(set(bad_ifg11+bad_ifg12))
    bad_ifg_all.sort()

    ifgdates = list(set(ifgdates_all)-set(bad_ifg_all))
    ifgdates.sort()


def iterative_correction():
    # define first iteration output dir
    iter = 1
    iter_unwdir = ccdir+"{}".format(int(iter))
    iter_unw_path = os.path.abspath(os.path.join(args.frame_dir, iter_unwdir))  # to read .unw
    if os.path.exists(iter_unw_path): shutil.rmtree(iter_unw_path)
    Path(iter_unw_path).mkdir(parents=True, exist_ok=True)

    # Link unw
    for pair in ifgdates:
        pair_dir = os.path.join(iter_unw_path, pair)
        Path(pair_dir).mkdir(parents=True, exist_ok=True)
        unwfile = os.path.join(ifgdir, pair, pair + '.unw')
        linkfile = os.path.join(pair_dir, pair + '.unw')
        os.link(unwfile, linkfile)

    # run 0th iteration
    run_130(iter_unwdir, iter)
    run_131(iter)
    resid_threshold_file = os.path.join(infodir, '131resid_2pi{}.txt'.format(int(iter)))
    current_thresh = float(io_lib.get_param_par(resid_threshold_file, 'RMS_thresh'))
    print("current threshold is {}".format(current_thresh))

    # iterative correction
    while current_thresh > args.thresh:
        print("Iteration {}".format(int(iter)))
        print("Correction threshold = {:2f}, above target {:2f}, keep correcting...".format(current_thresh, args.thresh))
        next_iter = iter + 1
        next_iter_unwdir = ccdir + "{}".format(int(next_iter))

        run_132(iter_unwdir, next_iter_unwdir, iter)
        run_130(next_iter_unwdir, next_iter)
        run_131(next_iter)

        resid_threshold_file = os.path.join(infodir, '131resid_2pi{}.txt'.format(int(next_iter)))
        current_thresh = float(io_lib.get_param_par(resid_threshold_file, 'RMS_thresh'))

        iter = next_iter

    # compile all results into cum.h5
    run_133(iter)


def run_130(d_dir, iter):
    os.system('LiCSBAS130_sb_inv.py -c {} -d {} -t {} --suffix {} --inv_alg WLS --n_para 6 --keep_incfile --nopngs'.format(
        ccdir, d_dir, args.ts_dir, int(iter)))


def run_131(iter):
    os.system('LiCSBAS131_residual_threshold.py -g {} -t {} -p {} --suffix {} '.format(
        ccdir, args.ts_dir, args.percentile, int(iter)))


def run_132(before_dir, after_dir, iter):
    os.system('LiCSBAS132_3D_correction.py -c {} -g {} -r {} -t {} --suffix {} '.format(
        ccdir, before_dir, after_dir, args.ts_dir, int(iter)))


def run_133(iter):
    os.system('LiCSBAS133_write_h5.py -c {} -t {} --suffix {} '.format(
        ccdir, args.ts_dir, int(iter)))


def main():
    start_time = start()
    init_args()
    set_input_output()
    get_ifgdates()
    iterative_correction()
    finish(start_time)


if __name__ == "__main__":
    main()