#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib


def block_sum(array, k):
    result = np.add.reduceat(np.add.reduceat(array, np.arange(0, array.shape[0], k), axis=0),
                             np.arange(0, array.shape[1], k), axis=1)
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect coregistration error")
    parser.add_argument('-f', "--frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-g', '--GEOCml_dir', dest="unw_dir", default="GEOCml10GACOS", help="folder containing unw input")
    parser.add_argument('-t', '--ts_dir', dest="ts_dir", default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-w', '--window_size', dest="win", default="5", type=float, help="Window size in km")
    args = parser.parse_args()

    ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    resultsdir = os.path.join(tsadir, 'results')
    infodir = os.path.join(tsadir, 'info')

    ### Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    print("\nSize         : {} x {}".format(width, length), flush=True)

    ### Get resolution
    dempar = os.path.join(ifgdir, 'EQA.dem_par')
    lattitude_resolution = float(io_lib.get_param_par(dempar, 'post_lat'))
    window_size = int(abs(args.win / 110 / lattitude_resolution) + 0.5)   # 110 km per degree latitude
    print("\nWindow size : ", window_size)

    ifgdates = tools_lib.get_ifgdates(ifgdir)

    n_unw = np.zeros((length, width), dtype=np.float32)
    n_coh = np.zeros((length, width), dtype=np.float32)
    n_con = np.zeros((length, width), dtype=np.float32)

    for ifgd in ifgdates:
        # turn ifg into ones and zeros for non-nan and nan values
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)
        unw[unw == 0] = np.nan # Fill 0 with nan
        n_unw += ~np.isnan(unw) # Summing number of unnan unw

        # coherence values from 0 to 1
        ccfile = os.path.join(ifgdir, ifgd, ifgd + '.cc')
        coh = io_lib.read_img(ccfile, length, width, np.uint8)
        coh = coh.astype(np.float32) / 255  # keep 0 as 0 which represent nan values
        n_coh += coh

        # connected components in terms of component area (pixel count)
        confile = os.path.join(ifgdir, ifgd, ifgd+'.conncomp')
        con = io_lib.read_img(confile, length, width, np.uint8)
        # replace component index by component size. the first component is the 0th component, which should be of size 0
        uniq_components, pixel_counts = np.unique(con.flatten(), return_counts=True)
        for i, component in enumerate(uniq_components[1:]):
            con[con==component] = pixel_counts[i+1]
        n_con += con

        del unw, coh, con

    block_unw = block_sum(n_unw, window_size)
    block_coh = block_sum(n_coh, window_size)
    block_con = block_sum(n_con, window_size)
    block_unw = block_unw / np.max(block_unw)
    block_coh = block_coh / np.max(block_coh)
    block_con = block_con / np.max(block_con)

    hgtfile = os.path.join(resultsdir, 'hgt')
    hgt = io_lib.read_img(hgtfile, length, width)
    block_mean_hgt = block_sum(hgt, window_size)/(window_size**2)
    repeat_block_mean_hgt = np.repeat(block_mean_hgt, window_size, axis=1)
    broadcast_mean_hgt = np.repeat(repeat_block_mean_hgt, window_size, axis=0)
    hgt_demean = hgt - broadcast_mean_hgt[:hgt.shape[0], :hgt.shape[1]]
    hgt_demean_square = hgt_demean ** 2
    block_rms_hgt = np.sqrt( block_sum(hgt_demean_square, window_size) / (window_size ** 2) )
    block_rms_hgt = block_rms_hgt / np.max(block_rms_hgt)

    fig, ax = plt.subplots(2, 2, sharey='all', sharex='all')
    im_unw = ax[0,0].imshow(block_unw)
    im_coh = ax[0,1].imshow(block_coh)
    im_con = ax[1,0].imshow(block_con)
    im_hgt = ax[1,1].imshow(block_rms_hgt)
    plt.colorbar(im_unw, ax=ax[0,0])
    plt.colorbar(im_coh, ax=ax[0,1])
    plt.colorbar(im_con, ax=ax[1,0])
    plt.colorbar(im_hgt, ax=ax[1,1])
    fig.savefig("reference.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(block_rms_hgt)
    ax[1].imshow(hgt)
    fig.savefig("height.png", dpi=300, bbox_inches='tight')

