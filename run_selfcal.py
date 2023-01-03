#!/usr/bin/env python

# Doing selfcal on a quocka field
import glob
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

# change nfbin to 2
NFBIN = 2


def logprint(s2p, lf):
    print(s2p, file=lf)
    print(s2p)


# Get the rms and peak flux of an image. RMS is estimated using a clipped version of the image data.


def get_noise(img_name):
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
    logf = open(sourcename + ".scal.log", "w", 1)
    logprint("***** Processing %s *****" % sourcename, logf)
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

        logprint("***** Start selfcal: %s *****" % t, logf)
        logprint("Generate the dirty image:", logf)
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
            stdout=logf,
            stderr=logf,
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_map, "out=%s" % t_dirty],
            stdout=logf,
            stderr=logf,
        )

        try:
            sigma, peak_max, peak_min = get_noise(t_dirty)
        except:
            time.sleep(5)
            sigma, peak_max, peak_min = get_noise(t_dirty)

        clean_level = 5.0 * sigma

        logprint("RMS of dirty image: %s" % sigma, logf)
        # logprint("Peak flux density of dirty image: %s"%peak_max, logf)
        logprint("Generate a cleaned image:", logf)
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
            stdout=logf,
            stderr=logf,
        )
        call(
            [
                "restor",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "model=%s" % t_model,
                "out=%s" % t_restor,
            ],
            stdout=logf,
            stderr=logf,
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_restor, "out=%s" % t_p0],
            stdout=logf,
            stderr=logf,
        )
        try:
            sigma, peak_max, peak_min = get_noise(t_p0)
        except:
            time.sleep(5)
            sigma, peak_max, peak_min = get_noise(t_p0)

        # First round of phase selfcal.
        # Generate a mask
        logprint("***** First round of phase selfcal *****", logf)
        mask_level = np.amax([10 * sigma, -peak_min * 1.5])
        clean_level = 5.0 * sigma
        call(
            [
                "maths",
                "exp=<%s>" % t_restor,
                "mask=<%s>.gt.%s" % (t_restor, mask_level),
                "out=%s" % t_mask,
            ],
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
        )
        call(
            [
                "restor",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "model=%s" % t_model,
                "out=%s" % t_restor,
            ],
            stdout=logf,
            stderr=logf,
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_restor, "out=%s" % t_p1],
            stdout=logf,
            stderr=logf,
        )
        try:
            sigma, peak_max, peak_min = get_noise(t_p1)
        except:
            time.sleep(5)
            sigma, peak_max, peak_min = get_noise(t_p1)

        # Second round.
        logprint("***** Second round of phase selfcal *****", logf)
        mask_level = np.amax([10 * sigma, -peak_min * 1.5])
        clean_level = 5.0 * sigma

        call(
            [
                "maths",
                "exp=<%s>" % t_restor,
                "mask=<%s>.gt.%s" % (t_restor, mask_level),
                "out=%s" % t_mask,
            ],
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
        )
        call(
            [
                "restor",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "model=%s" % t_model,
                "out=%s" % t_restor,
            ],
            stdout=logf,
            stderr=logf,
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_restor, "out=%s" % t_p2],
            stdout=logf,
            stderr=logf,
        )
        try:
            sigma, peak_max, peak_min = get_noise(t_p2)
        except:
            time.sleep(5)
            sigma, peak_max, peak_min = get_noise(t_p2)

        # move on to amp selfcal.
        logprint("***** One round of amp+phase selfcal *****", logf)

        mask_level = np.amax([10 * sigma, -peak_min * 1.5])
        clean_level = 5.0 * sigma

        call(
            [
                "maths",
                "exp=<%s>" % t_restor,
                "mask=<%s>.gt.%s" % (t_restor, mask_level),
                "out=%s" % t_mask,
            ],
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
        )
        t_ascal = t + ".ascal"
        call(
            ["uvaver", "vis=%s" % t_pscal, "out=%s" % t_ascal], stdout=logf, stderr=logf
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
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
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
            stdout=logf,
            stderr=logf,
        )
        call(
            [
                "restor",
                "map=%s" % t_map,
                "beam=%s" % t_beam,
                "model=%s" % t_model,
                "out=%s" % t_restor,
            ],
            stdout=logf,
            stderr=logf,
        )
        call(
            ["fits", "op=xyout", "in=%s" % t_restor, "out=%s" % t_p2a1],
            stdout=logf,
            stderr=logf,
        )

        shutil.rmtree(t_map)
        shutil.rmtree(t_beam)
        shutil.rmtree(t_restor)
        shutil.rmtree(t_model)

    except Exception as e:
        logprint("Failed to run selfcal: %s" % e, logf)

    logf.close()


def main(vislist):
    slist = []
    for vis in vislist:
        slist.append(selfcal(vis))

    _ = compute(*slist)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("vislist", nargs="+")
    parser.add_argument("--ncores", type=int, default=1)
    args = parser.parse_args()
    with Client(n_workers=args.ncores, threads_per_worker=1) as client:
        main(args.vislist)
