#!/usr/bin/env python3
"""
v1.0 20220928 Qi Ou, Leeds Uni
Assemble all results into cum.h5
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
import re
import xarray as xr
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib


def init_args():
    parser = argparse.ArgumentParser(description="Detect coregistration error")
    parser.add_argument('-f', "--frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-c', '--comp_cc_dir', default="GEOCml10GACOS", help="folder containing connected components and cc files")
    parser.add_argument('-g', '--geoc_dir', default="GEOC", help="folder containing geo.E/N/U.tif")
    parser.add_argument('-t', '--ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('--suffix', default="", type=str, help="suffix of the last iteration")
    args = parser.parse_args()
    return args


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
    print('\nLiCSBAS13_iterative_ts_inversion.py Successfully finished!!\n')
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


def set_input_output(args):
    # define input directories
    geoc_dir = os.path.abspath(os.path.join(args.frame_dir, args.geoc_dir))
    ccdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir))
    ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir+args.suffix))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    infodir = os.path.join(tsadir, 'info')
    last_result_dir = os.path.join(tsadir, '130results{}'.format(iter))
    resultsdir = os.path.join(tsadir, 'results')
    last_cumh5file = os.path.join(tsadir, '130cum{}.h5'.format(args.suffix))
    cumh5file = os.path.join(tsadir, 'cum.h5')
    global geoc_dir, ccdir, ifgdir, tsadir, infodir, last_result_dir, resultsdir, last_cumh5file, cumh5file


def read_length_width():
    # read ifg size
    mlipar = os.path.join(ccdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    return length, width


def calc_n_unw():
    n_unw = np.zeros((length, width), dtype=np.int16)
    for ifgd in ifgdates:
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)
        unw[unw == 0] = np.nan # Fill 0 with nan
        n_unw += ~np.isnan(unw) # Summing number of unnan unw
    n_unwfile = os.path.join(resultsdir, 'n_unw')
    np.float32(n_unw).tofile(n_unwfile)


def calc_coh_avg():
    # calc n_unw and avg_coh of final data set
    coh_avg = np.zeros((length, width), dtype=np.float32)
    n_coh = np.zeros((length, width), dtype=np.int16)
    for ifgd in ifgdates:
        ccfile = os.path.join(ccdir, ifgd, ifgd+'.cc')
        if os.path.getsize(ccfile) == length*width:
            coh = io_lib.read_img(ccfile, length, width, np.uint8)
            coh = coh.astype(np.float32)/255
        else:
            coh = io_lib.read_img(ccfile, length, width)
            coh[np.isnan(coh)] = 0  # Fill nan with 0
        coh_avg += coh
        n_coh += (coh!=0)
    coh_avg[n_coh==0] = np.nan
    n_coh[n_coh==0] = 1 #to avoid zero division
    coh_avg = coh_avg/n_coh
    coh_avg[coh_avg==0] = np.nan
    coh_avgfile = os.path.join(resultsdir, 'coh_avg')
    coh_avg.tofile(coh_avgfile)


def calc_n_loop_error():
    ''' same as loop_closure_4th in LiCSBAS12_loop_closure.py '''
    print('Compute n_loop_error and n_ifg_noloop...', flush=True)

    # read reference
    reffile = os.path.join(infodir, '120ref.txt')
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  # str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

    # create 3D cube - False means presumed error in the loop
    a = np.full((length, width,len(ifgdates)), False)
    da = xr.DataArray(
        data=a,
        dims=[ "y", "x", "ifgd"],
        coords=dict(y=np.arange(length), x=np.arange(width), ifgd=ifgdates))

    ### Get loop matrix
    Aloop = loop_lib.make_loop_matrix(ifgdates)
    n_loop = Aloop.shape[0]

    ### Count loop error by pixel
    n_loop_err = np.zeros((length, width), dtype=np.int16)
    for i in range(0, len(Aloop)):
        if np.mod(i, 100) == 0:
            print("  {0:3}/{1:3}th loop...".format(i, n_loop), flush=True)

        ### Read unw
        unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = loop_lib.read_unw_loop_ph(Aloop[i, :], ifgdates, ifgdir, length, width)

        ## Compute ref
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

        ## Calculate loop phase taking into account ref phase
        loop_ph = unw12 + unw23 - unw13 - (ref_unw12 + ref_unw23 - ref_unw13)

        ## Count number of loops with suspected unwrap error (>pi)
        loop_ph[np.isnan(loop_ph)] = 0  # to avoid warning
        is_ok = np.abs(loop_ph) < np.pi
        da.loc[:, :, ifgd12] = np.logical_or(da.loc[:, :, ifgd12], is_ok)
        da.loc[:, :, ifgd23] = np.logical_or(da.loc[:, :, ifgd23], is_ok)
        da.loc[:, :, ifgd13] = np.logical_or(da.loc[:, :, ifgd13], is_ok)
        n_loop_err = n_loop_err + ~is_ok  # suspected unw error

    n_loop_err = np.array(n_loop_err, dtype=np.int16)
    n_loop_err_file = os.path.join(resultsdir, 'n_loop_err')
    n_loop_err.tofile(n_loop_err_file)


def write_h5(cumh5file):
    # Write additional results to h5
    print('\nWriting to HDF5 file...')
    cumh5 = h5.File(cumh5file, 'w')
    compress = 'gzip'
    indices = ['coh_avg', 'hgt', 'n_loop_err', 'n_unw', 'slc.mli']

    for index in indices:
        file = os.path.join(resultsdir, index)
        if os.path.exists(file):
            data = io_lib.read_img(file, length, width)
            cumh5.create_dataset(index, data=data, compression=compress)
        else:
            print('  {} not exist in results dir. Skip'.format(index))

    LOSvecs = ['E.geo', 'N.geo', 'U.geo']
    for LOSvec in LOSvecs:
        file = os.path.join(geoc_dir, LOSvec)
        if os.path.exists(file):
            data = io_lib.read_img(file, length, width)
            cumh5.create_dataset(LOSvec, data=data, compression=compress)
        else:
            print('  {} not exist in GEOCml dir. Skip'.format(LOSvec))

    cumh5.close()


def main():
    # intialise
    start_time = start()
    args = init_args()

    # directory settings
    set_input_output(args)
    length, width = read_length_width(ccdir)
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    global length, width, ifgdates

    # copy everything from last iter to final
    shutil.copyfile(last_cumh5file, cumh5file)
    shutil.copytree(last_result_dir, resultsdir, dirs_exist_ok=True)

    # calc quality stats based on the final corrected unw
    calc_n_unw()
    calc_coh_avg()
    calc_n_loop_error()

    # compile all results to h5
    write_h5(cumh5file)

    # report finish
    finish(start_time)


if __name__ == '__main__':
    main()





