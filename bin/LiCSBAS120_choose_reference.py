#!/usr/bin/env python3
"""
========
Overview
========
This script:
 - calculates the block sum of unw pixels
 - calculates the block sum of coherence
 - calculates the block sum of connected component size
 - calculates the block std of height
 - combine and normalise a proxy [0-1] of suitability of reference window
 - choose amongst the selected windows (above threshold) the nearest to desired reference location
 - discard ifgs with all nan values in the chosen reference window

===============
Input & output files
===============

Inputs in GEOCml*/:
 - slc.mli.par
 - hgt
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.cc
   - yyyymmdd_yyyymmdd.conncomp
   - yyyymmdd_yyyymmdd.unw

Outputs in TS_GEOCml*/ :
 - info/
  - 120ref.txt        : refx1:refx2/refy1:refy2
  - 120bad_ifg.txt    : list of ifgs with all nan values within the chosen reference window
  - 120_reference.png : proxy plots

 - networks/
  - network120*png

=====
Usage
=====
LiCSBAS120_choose_reference.py [-h] [-f FRAME_DIR] [-g UNW_DIR] [-t TS_DIR] [-w WIN] [-r [0-1]] [--w_unw [0-1]] [--w_coh [0-1]] [--w_con [0-1]] [--w_hgt [0-1]] [--refx [0-1]] [--refy [0-1]]
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import sys
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from matplotlib import cm


def block_sum(array, k):
    result = np.add.reduceat(np.add.reduceat(array, np.arange(0, array.shape[0], k), axis=0),
                             np.arange(0, array.shape[1], k), axis=1)
    return result


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass


def init_args():
    global args
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest="frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-g', dest='unw_dir', default="GEOCml10GACOS", help="folder containing unw input")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-w', dest='win', default="5", type=float, help="Window size in km")
    parser.add_argument('-r', dest='thresh', default=0.9, choices=range(0, 1), metavar="[0-1]", type=float, help="proxy threshold above which the window nearest to desired center will be chosen as the reference window")
    parser.add_argument("--w_unw", default=1, choices=range(0, 1), metavar="[0-1]", type=float, help="weight for block_sum_unw_pixel")
    parser.add_argument('--w_coh', default=1, choices=range(0, 1), metavar="[0-1]", type=float, help="weight for block_sum_coherence")
    parser.add_argument('--w_con', default=1, choices=range(0, 1), metavar="[0-1]", type=float, help="weight for block_sum_component_size")
    parser.add_argument('--w_hgt', default=1, choices=range(0, 1), metavar="[0-1]", type=float, help="weight for block_std_hgt")
    parser.add_argument('--refx', default=0.5, choices=range(0, 1), metavar="[0-1]", type=float, help="x axis fraction of desired ref center from left (default 0.5)")
    parser.add_argument('--refy', default=0.5, choices=range(0, 1), metavar="[0-1]", type=float, help="y axis fraction of desired ref center from top (default 0.5)")
    args = parser.parse_args()


def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20221020; author="Qi Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish():
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


def set_input_output():
    global ifgdir, tsadir, infodir, resultsdir, netdir, noref_ifgfile, no_ref_dir, reference_png

    ### Define input directories
    ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    resultsdir = os.path.join(tsadir, 'results')
    infodir = os.path.join(tsadir, 'info')

    ### Define output dir
    no_ref_dir = os.path.join(tsadir, '120no_ref')
    if not os.path.exists(no_ref_dir): os.mkdir(no_ref_dir)
    netdir = os.path.join(tsadir, 'network')
    noref_ifgfile = os.path.join(infodir, '120bad_ifg.txt')
    reference_png = os.path.join(infodir, "120_reference.png")


def read_length_width():
    global length, width

    ### Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    print("\nSize         : {} x {}".format(width, length), flush=True)


def decide_reference_window_size():
    global window_size
    ### Get resolution
    dempar = os.path.join(ifgdir, 'EQA.dem_par')
    lattitude_resolution = float(io_lib.get_param_par(dempar, 'post_lat'))
    window_size = int(abs(args.win / 110 / lattitude_resolution) + 0.5)   # 110 km per degree latitude
    print("\nWindow size : ", window_size)


def get_ifgdates():
    global ifgdates

    ### Get dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)

    ### Read bad_ifg11 and rm_ifg
    bad_ifg11file = os.path.join(infodir, '11bad_ifg.txt')
    bad_ifg11 = io_lib.read_ifg_list(bad_ifg11file)

    ### Remove bad ifgs and images from list
    ifgdates = list(set(ifgdates)-set(bad_ifg11))
    ifgdates.sort()


def calc_block_sum_of_unw_coh_component_size():
    global block_unw, block_coh, block_con, ifgd
    ### Start counting indices for choosing the reference
    n_unw = np.zeros((length, width), dtype=np.float32)
    n_coh = np.zeros((length, width), dtype=np.float32)
    n_con = np.zeros((length, width), dtype=np.float32)

    ### Accumulate through network (1)unw pixel counts, (2) coherence and (3) size of connected components
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

    ### calculate block sum
    block_unw = block_sum(n_unw, window_size)
    block_coh = block_sum(n_coh, window_size)
    block_con = block_sum(n_con, window_size)


def calc_height_std():
    global block_rms_hgt
    ### calculate block standard deviation of height
    hgtfile = os.path.join(resultsdir, 'hgt')
    hgt = io_lib.read_img(hgtfile, length, width)
    block_mean_hgt = block_sum(hgt, window_size)/(window_size**2)
    repeat_block_mean_hgt = np.repeat(block_mean_hgt, window_size, axis=1)
    broadcast_mean_hgt = np.repeat(repeat_block_mean_hgt, window_size, axis=0)
    hgt_demean = hgt - broadcast_mean_hgt[:hgt.shape[0], :hgt.shape[1]]
    hgt_demean_square = hgt_demean ** 2
    block_rms_hgt = np.sqrt( block_sum(hgt_demean_square, window_size) / (window_size ** 2) )


def clip_normalise_combine_indices():
    global block_proxy, block_unw, block_coh, block_con, block_rms_hgt
    ### turn 0 to nan
    block_unw[block_unw == 0] = np.nan
    block_coh[block_coh == 0] = np.nan
    block_con[block_con == 0] = np.nan
    block_rms_hgt[block_rms_hgt == 0] = np.nan

    ### clipping values at zigzagy edges which are the lowest for block sums and highest for std
    block_unw[block_unw < np.nanpercentile(block_unw, 10)] = np.nanpercentile(block_unw, 10)
    block_coh[block_coh < np.nanpercentile(block_coh, 10)] = np.nanpercentile(block_coh, 10)
    block_con[block_con < np.nanpercentile(block_con, 10)] = np.nanpercentile(block_con, 10)
    block_rms_hgt[block_rms_hgt > np.nanpercentile(block_rms_hgt, 90)] = np.nanpercentile(block_rms_hgt, 90)

    ### normalise with nan minmax
    block_unw = (block_unw - np.nanmin(block_unw)) / (np.nanmax(block_unw) - np.nanmin(block_unw))
    block_coh = (block_coh - np.nanmin(block_coh)) / (np.nanmax(block_coh) - np.nanmin(block_coh))
    block_con = (block_con - np.nanmin(block_con)) / (np.nanmax(block_con) - np.nanmin(block_con))
    block_rms_hgt = (block_rms_hgt - np.nanmin(block_rms_hgt)) / (np.nanmax(block_rms_hgt) - np.nanmin(block_rms_hgt))

    ### calculate proxy from 4 indices and normalise
    block_proxy = args.w_unw * block_unw + args.w_coh * block_coh + args.w_con * block_con - args.w_hgt * block_rms_hgt
    block_proxy = (block_proxy - np.nanmin(block_proxy)) / (np.nanmax(block_proxy) - np.nanmin(block_proxy))


def closest_to_ref_center():
    ''' Find the point with large enough (>args.thresh) block_proxy that is the closest to
    the desired ref center defined by args.refx and args.refy'''
    global desired_ref_center_x, desired_ref_center_y, refx, refy
    ## choose distance closer to center
    desired_ref_center_y = int(block_proxy.shape[0] * args.refy)
    desired_ref_center_x = int(block_proxy.shape[1] * args.refx)
    refys, refxs = np.where(block_proxy > args.thresh)
    distance_to_center = np.sqrt((refys - desired_ref_center_y) ** 2 + (refxs - desired_ref_center_x) ** 2)
    nearest_to_center = np.min(distance_to_center)
    index_nearest_to_center = np.where(distance_to_center == nearest_to_center)
    refy = refys[index_nearest_to_center][0]
    refx = refxs[index_nearest_to_center][0]
    print("Reference nearest to center: refy={}, refx={}".format(refy, refx))


def plot_ref_proxies():
    ### load example unw for plotting in block resolution
    unwfile = os.path.join(ifgdir, ifgd, ifgd + '.unw')
    unw = io_lib.read_img(unwfile, length, width)
    unw_example = block_sum(unw, window_size)
    unw_example[unw_example == 0] = np.nan

    # plot figure
    fig, ax = plt.subplots(2, 3, sharey='all', sharex='all')
    im_unw = ax[0, 0].imshow(block_unw, vmin=0, vmax=1)
    im_coh = ax[0, 1].imshow(block_coh, vmin=0, vmax=1)
    im_con = ax[1, 0].imshow(block_con, vmin=0, vmax=1)
    im_hgt = ax[1, 1].imshow(block_rms_hgt, vmin=0, vmax=1)
    im_proxy = ax[0, 2].imshow(block_proxy)
    im_example = ax[1, 2].imshow(unw_example, cmap=cm.RdBu)
    plt.colorbar(im_unw, ax=ax, orientation='horizontal')

    ax[0, 0].set_title("block_sum_unw")
    ax[0, 1].set_title("block_sum_coh")
    ax[1, 0].set_title("block_sum_comp_size")
    ax[1, 1].set_title("block_std_hgt")
    ax[0, 2].set_title("proxy")
    ax[1, 2].set_title("unw example")

    ax[0, 0].scatter(refx, refy, s=3, c='red')
    ax[0, 1].scatter(refx, refy, s=3, c='red')
    ax[0, 2].scatter(refx, refy, s=3, c='red')
    ax[0, 2].scatter(desired_ref_center_x, desired_ref_center_y, s=3, c='black')
    ax[1, 0].scatter(refx, refy, s=3, c='red')
    ax[1, 1].scatter(refx, refy, s=3, c='red')
    ax[1, 2].scatter(refx, refy, s=3, c='red')

    fig.savefig(reference_png, dpi=300, bbox_inches='tight')


def save_reference_to_file():
    global refx1, refx2, refy1, refy2

    # calc reference window in full resolution ifg
    refx1, refx2, refy1, refy2 = refx*window_size, (refx+1)*window_size, refy*window_size, (refy+1)*window_size
    print('Selected ref: {}:{}/{}:{}'.format(refx1, refx2, refy1, refy2), flush=True)

    ### Save ref
    refsfile = os.path.join(infodir, '120ref.txt')
    with open(refsfile, 'w') as f:
        print('{}:{}/{}:{}'.format(refx1, refx2, refy1, refy2), file=f)


def discard_ifg_with_all_nans_at_ref():
    global noref_ifg
    ### identify IFGs with all nan in the reference window
    ### Check ref exist in unw. If not, list as noref_ifg
    noref_ifg = []
    for ifgd in ifgdates:

        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw_data = io_lib.read_img(unwfile, length, width)
        unw_ref = unw_data[refy1:refy2, refx1:refx2]

        unw_ref[unw_ref == 0] = np.nan # Fill 0 with nan
        if np.all(np.isnan(unw_ref)):
            noref_ifg.append(ifgd)

            # plot no_ref_ifg with reference window to no_ref folder
            pngfile = os.path.join(no_ref_dir, ifgd + '.png')
            plt.imshow(unw_data, vmin=np.nanpercentile(unw_data, 1), vmax=np.nanpercentile(unw_data, 99), cmap=cm.RdBu)
            plt.plot([refx1, refx2, refx2, refx1, refx1], [refy1, refy1, refy2, refy2, refy1], 'r')
            plt.title(ifgd)
            plt.savefig(pngfile, dpi=300, bbox_inches='tight')

    # save list of no_ref_ifg to a text file in info directory
    with open(noref_ifgfile, 'w') as f:
        for i in noref_ifg:
            print('{}'.format(i), file=f)


def plot_networks():
    #%% Plot network
    ## Read bperp data or dummy
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_im = len(imdates)
    bperp_file = os.path.join(ifgdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else: #dummy
        bperp = np.random.random(n_im).tolist()

    pngfile = os.path.join(netdir, 'network120_all.png')
    plot_lib.plot_network(ifgdates, bperp, [], pngfile)

    pngfile = os.path.join(netdir, 'network120.png')
    plot_lib.plot_network(ifgdates, bperp, noref_ifg, pngfile)

    pngfile = os.path.join(netdir, 'network120_remain.png')
    plot_lib.plot_network(ifgdates, bperp, noref_ifg, pngfile, plot_bad=False)


def main():
    start()
    init_args()
    set_input_output()
    read_length_width()
    decide_reference_window_size()
    get_ifgdates()

    calc_block_sum_of_unw_coh_component_size()
    calc_height_std()
    clip_normalise_combine_indices()

    closest_to_ref_center()
    plot_ref_proxies()
    save_reference_to_file()
    discard_ifg_with_all_nans_at_ref()
    plot_networks()
    finish()


if __name__ == "__main__":
    main()




