#!/usr/bin/env python3

# Doing selfcal on a quocka field
import configparser
import glob
import logging
import os
import shutil
import sys
import time
from http import client
from subprocess import call

import numpy as np
from astropy.io import fits
from dask import compute, delayed
from dask.distributed import Client, LocalCluster

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logger = logging.getLogger(__name__)
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger.setLevel(logging.INFO)


def get_noise(img_name):
    # Get the rms and peak flux of an image. RMS is estimated using a clipped version of the image data.
    hdu = fits.open(img_name)
    data = hdu[0].data[0, 0]
    rms_initial = np.std(data)
    rms = np.std(
        data[np.logical_and(data > -2.5 * rms_initial, data < 2.5 * rms_initial)]
    )
    peak_max = np.amax(data)
    peak_min = np.amin(data)
    hdu.close()
    return rms, peak_max, peak_min


@delayed
def selfcal(vis):
    sourcename = os.path.basename(vis).replace(".2100.pscal", "")
    logger.info(
        "***** Processing %s *****" % sourcename,
    )
    t = f"{sourcename}.2100"
    try:
        t_pscal = t + ".pscal"
        t_map = t + ".map"
        t_beam = t + ".beam"
        t_model = t + ".model"
        t_restor = t + ".restor"
        t_p0 = t + ".p0.fits"
        t_dirty = t + ".dirty.fits"
        t_mask = t + ".mask"

        logger.info(
            "***** Start selfcal: %s *****" % t,
        )
        logger.info(
            "Generate the dirty image:",
        )
        # Generate a MFS image without selfcal.
        call(
            [
                "invert",
                "vis=%s" % t_pscal,
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "robust=0.5",
                "stokes=i",
                "options=mfs,double,sdb",
                "imsize=2,2,beam",
                "cell=5,5,res",
            ],
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_map, "out=%s" % t_dirty],
        )

        try:
            sigma, peak_max, peak_min = get_noise(t_dirty)
        except:
            time.sleep(5)
            sigma, peak_max, peak_min = get_noise(t_dirty)

        clean_level = 5.0 * sigma

        logger.info(
            "RMS of dirty image: %s" % sigma,
        )
        # logger.info("Peak flux density of dirty image: %s"%peak_max, )
        logger.info(
            "Generate a cleaned image:",
        )
        call(
            [
                "mfclean",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "out=%s" % t_model,
                "niters=10000",
                "cutoff=%s,%s" % (clean_level, 2 * sigma),
                "region='perc(90)'",
            ],
        )
        call(
            [
                "restor",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "model=%s" % t_model,
                "out=%s" % t_restor,
            ],
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_restor, "out=%s" % t_p0],
        )
        try:
            sigma, peak_max, peak_min = get_noise(t_p0)
        except:
            time.sleep(5)
            sigma, peak_max, peak_min = get_noise(t_p0)

        # First round of phase selfcal.
        # Generate a mask
        logger.info(
            "***** First round of phase selfcal *****",
        )
        mask_level = np.amax([10 * sigma, -peak_min * 1.5])
        clean_level = 5.0 * sigma
        call(
            [
                "maths",
                "exp=<%s>" % t_restor,
                "mask=<%s>.gt.%s" % (t_restor, mask_level),
                "out=%s" % t_mask,
            ],
        )
        shutil.rmtree(t_restor)
        shutil.rmtree(t_model)
        call(
            [
                "mfclean",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "out=%s" % t_model,
                "niters=1500",
                "cutoff=%s,%s" % (clean_level, 2 * sigma),
                "region=mask(%s)" % t_mask,
            ],
        )
        call(
            [
                "selfcal",
                "vis=%s" % t_pscal,
                "model=%s" % t_model,
                "interval=5",
                "nfbin=1",
                "options=phase,mfs",
            ],
        )
        shutil.rmtree(t_map)
        shutil.rmtree(t_beam)
        shutil.rmtree(t_mask)
        shutil.rmtree(t_model)

        t_p1 = t + ".p1.fits"
        call(
            [
                "invert",
                "vis=%s" % t_pscal,
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "robust=0.5",
                "stokes=i",
                "options=mfs,double,sdb",
                "imsize=2,2,beam",
                "cell=5,5,res",
            ],
        )
        call(
            [
                "mfclean",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "out=%s" % t_model,
                "niters=10000",
                "cutoff=%s,%s" % (clean_level, 2 * sigma),
                "region='perc(90)'",
            ],
        )
        call(
            [
                "restor",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "model=%s" % t_model,
                "out=%s" % t_restor,
            ],
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_restor, "out=%s" % t_p1],
        )
        try:
            sigma, peak_max, peak_min = get_noise(t_p1)
        except:
            time.sleep(5)
            sigma, peak_max, peak_min = get_noise(t_p1)

        # Second round.
        logger.info(
            "***** Second round of phase selfcal *****",
        )
        mask_level = np.amax([10 * sigma, -peak_min * 1.5])
        clean_level = 5.0 * sigma

        call(
            [
                "maths",
                "exp=<%s>" % t_restor,
                "mask=<%s>.gt.%s" % (t_restor, mask_level),
                "out=%s" % t_mask,
            ],
        )
        shutil.rmtree(t_restor)
        shutil.rmtree(t_model)
        call(
            [
                "mfclean",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "out=%s" % t_model,
                "niters=1500",
                "cutoff=%s,%s" % (clean_level, 2 * sigma),
                "region=mask(%s)" % t_mask,
            ],
        )

        call(
            [
                "selfcal",
                "vis=%s" % t_pscal,
                "model=%s" % t_model,
                "interval=0.5",
                "nfbin=1",
                "options=phase,mfs",
            ],
        )
        shutil.rmtree(t_map)
        shutil.rmtree(t_beam)
        shutil.rmtree(t_mask)
        shutil.rmtree(t_model)

        t_p2 = t + ".p2.fits"
        call(
            [
                "invert",
                "vis=%s" % t_pscal,
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "robust=0.5",
                "stokes=i",
                "options=mfs,double,sdb",
                "imsize=2,2,beam",
                "cell=5,5,res",
            ],
        )
        call(
            [
                "mfclean",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "out=%s" % t_model,
                "niters=10000",
                "cutoff=%s,%s" % (clean_level, 2 * sigma),
                "region='perc(90)'",
            ],
        )
        call(
            [
                "restor",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "model=%s" % t_model,
                "out=%s" % t_restor,
            ],
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_restor, "out=%s" % t_p2],
        )
        try:
            sigma, peak_max, peak_min = get_noise(t_p2)
        except:
            time.sleep(5)
            sigma, peak_max, peak_min = get_noise(t_p2)

        # move on to amp selfcal.
        logger.info(
            "***** One round of amp+phase selfcal *****",
        )

        mask_level = np.amax([10 * sigma, -peak_min * 1.5])
        clean_level = 5.0 * sigma

        call(
            [
                "maths",
                "exp=<%s>" % t_restor,
                "mask=<%s>.gt.%s" % (t_restor, mask_level),
                "out=%s" % t_mask,
            ],
        )
        shutil.rmtree(t_restor)
        shutil.rmtree(t_model)
        call(
            [
                "mfclean",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "out=%s" % t_model,
                "niters=1500",
                "cutoff=%s,%s" % (clean_level, 2 * sigma),
                "region=mask(%s)" % t_mask,
            ],
        )
        t_ascal = t + ".ascal"
        call(
            ["uvaver", "vis=%s" % t_pscal, "out=%s" % t_ascal],
        )

        # do the first round of amp selfcal with model generated using phase selfcal.
        call(
            [
                "selfcal",
                "vis=%s" % t_ascal,
                "model=%s" % t_model,
                "interval=5",
                "nfbin=1",
                "options=amp,mfs",
            ],
        )
        shutil.rmtree(t_map)
        shutil.rmtree(t_beam)
        shutil.rmtree(t_mask)
        shutil.rmtree(t_model)

        t_p2a1 = t + ".p2a1.fits"
        call(
            [
                "invert",
                "vis=%s" % t_ascal,
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "robust=0.5",
                "stokes=i",
                "options=mfs,double,sdb",
                "imsize=2,2,beam",
                "cell=5,5,res",
            ],
        )
        call(
            [
                "mfclean",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "out=%s" % t_model,
                "niters=10000",
                "cutoff=%s,%s" % (clean_level, 2 * sigma),
                "region='perc(90)'",
            ],
        )
        call(
            [
                "restor",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "model=%s" % t_model,
                "out=%s" % t_restor,
            ],
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_restor, "out=%s" % t_p2a1],
        )

        shutil.rmtree(t_map)
        shutil.rmtree(t_beam)
        shutil.rmtree(t_restor)
        shutil.rmtree(t_model)

    except Exception as e:
        logger.info(
            "Failed to run selfcal: %s" % e,
        )


def main(cfg, vislist):
    # Set globals
    global NFBIN
    NFBIN = cfg.getint("output", "nfbin")

    slist = []
    for vis in vislist:
        slist.append(selfcal(vis))

    _ = compute(*slist)


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Input configuration file")
    parser.add_argument("vislist", nargs="+")
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument(
        "-l",
        "--log_file",
        help="Name of output log file [default log.txt]",
        default="log.txt",
    )
    args = parser.parse_args()

    cfg = configparser.RawConfigParser()
    cfg.read(args.config_file)

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    with Client(n_workers=args.ncores, threads_per_worker=1) as client:
        main(cfg, args.vislist)


if __name__ == "__main__":
    cli()
