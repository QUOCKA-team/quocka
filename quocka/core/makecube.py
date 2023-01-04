#!/usr/bin/env python3
"""Make QUOCKA cubes"""

import logging
import sys
from functools import partial
from glob import glob

from quocka.aux import au2
import matplotlib.pyplot as plt
import numpy as np
import schwimmbad
import scipy.signal
from astropy import units as u
from astropy.io import fits
from IPython import embed
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from tqdm import tqdm

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
DATE_FORMAT="%Y-%m-%d %H:%M:%S"
logger = logging.getLogger(__name__)
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger.setLevel(logging.INFO)


def round_up(n, decimals=0):
    multiplier = 10**decimals
    return np.ceil(n * multiplier) / multiplier


def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def getmaxbeam(
    data_dict,
    band,
    cutoff=15 * u.arcsec,
    tolerance=0.0001,
    nsamps=200,
    epsilon=0.0005,
    verbose=False,
    debug=False,
):
    """Find common beam.

    Arguments:
        data_dict {dict} -- Dict containing fits files.
        band {int} -- ATCA band name.

    Keyword Arguments:
        tolerance {float} -- See common_beam (default: {0.0001})
        nsamps {int} -- See common_beam (default: {200})
        epsilon {float} -- See common_beam (default: {0.0005})
        verbose {bool} -- Verbose output (default: {False})
        debug {bool} -- Show dubugging plots (default: {False})

    Returns:
        beam_dict {dict} -- Beam and frequency data.
    """
    files = data_dict[band]
    stokes = ["i", "q", "u", "v"]
    beam_dict = {}
    for stoke in stokes:
        beams = []
        freqs = []
        for file in files[stoke]:
            header = fits.getheader(file, memmap=True)
            freqs.append(header["CRVAL3"])
            beam = Beam.from_fits_header(header)
            beams.append(beam)
        beams = Beams(
            [beam.major.value for beam in beams] * u.deg,
            [beam.minor.value for beam in beams] * u.deg,
            [beam.pa.value for beam in beams] * u.deg,
        )
        flags = beams.major > cutoff
        beam_dict.update(
            {
                stoke + "_beams": beams,
                stoke + "_freqs": np.array(freqs) * u.Hz,
                stoke + "_flags": flags,
            }
        )
    if debug:
        plt.figure()
        plt.title(band)
        for stoke in stokes:
            idx = [not flag for flag in beam_dict[stoke + "_flags"]]
            plt.plot(
                beam_dict[stoke + "_freqs"][idx],
                beam_dict[stoke + "_beams"].major.to(u.arcsec)[idx],
                ".",
                alpha=0.5,
                label=stoke + "--BMAJ",
            )

        plt.plot(
            beam_dict[stoke + "_freqs"][idx],
            beam_dict[stoke + "_beams"].minor.to(u.arcsec)[idx],
            ".",
            alpha=0.5,
            label=stoke + "--BMIN",
        )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Beam size [arcsec]")
        plt.legend()
        plt.show()

    bmaj = []
    bmin = []
    bpa = []
    for stoke in stokes:
        bmaj += list(
            beam_dict[f"{stoke}_beams"].major.value[~beam_dict[f"{stoke}_flags"]]
        )
        bmin += list(
            beam_dict[f"{stoke}_beams"].minor.value[~beam_dict[f"{stoke}_flags"]]
        )
        bpa += list(beam_dict[f"{stoke}_beams"].pa.value[~beam_dict[f"{stoke}_flags"]])

    big_beams = Beams(bmaj * u.deg, bmin * u.deg, bpa * u.deg)

    try:
        cmn_beam = big_beams.common_beam(
            tolerance=tolerance, epsilon=epsilon, nsamps=nsamps
        )
    except BeamError:
        if verbose:
            logger.info("Couldn't find common beam with defaults")
            logger.info("Trying again with smaller tolerance")
        cmn_beam = big_beams.common_beam(
            tolerance=tolerance * 0.1, epsilon=epsilon, nsamps=nsamps
        )

    cmn_beam = Beam(
        major=my_ceil(cmn_beam.major.to(u.arcsec).value, precision=1) * u.arcsec,
        minor=my_ceil(cmn_beam.minor.to(u.arcsec).value, precision=1) * u.arcsec,
        pa=round_up(cmn_beam.pa.to(u.deg), decimals=2),
    )

    target_header = fits.getheader(data_dict[band]["i"][0], memmap=True)
    dx = target_header["CDELT1"] * -1 * u.deg
    dy = target_header["CDELT2"] * u.deg
    grid = dy
    conbeams = [cmn_beam.deconvolve(beam) for beam in big_beams]

    # Check that convolving beam will be nyquist sampled
    min_samps = []
    for b_idx, conbeam in enumerate(conbeams):
        # Get maj, min, pa
        samp = conbeam.minor / grid.to(u.arcsec)
        if samp < 2:
            min_samps.append([samp, b_idx])

    if len(min_samps) > 0:
        logger.info("Adjusting common beam to be sampled by grid!")
        worst_idx = np.argmin([samp[0] for samp in min_samps], axis=0)
        samp_cor_fac, idx = 2 / min_samps[worst_idx][0], int(min_samps[worst_idx][1])
        conbeam = conbeams[idx]
        major = conbeam.major
        minor = conbeam.minor * samp_cor_fac
        pa = conbeam.pa
        # Check for small major!
        if major < minor:
            major = minor
            pa = 0 * u.deg

        cor_beam = Beam(major, minor, pa)
        if verbose:
            logger.info(f"Smallest common beam is: {cmn_beam}")
        cmn_beam = big_beams[idx].convolve(cor_beam)
        cmn_beam = Beam(
            major=my_ceil(cmn_beam.major.to(u.arcsec).value, precision=1) * u.arcsec,
            minor=my_ceil(cmn_beam.minor.to(u.arcsec).value, precision=1) * u.arcsec,
            pa=round_up(cmn_beam.pa.to(u.deg), decimals=2),
        )
        if verbose:
            logger.info(f"Smallest common Nyquist sampled beam is: {cmn_beam}")
    if debug:
        from matplotlib.patches import Ellipse

        pixscale = 1 * u.arcsec
        fig = plt.figure()
        ax = plt.gca()
        for beam in big_beams:
            ellipse = Ellipse(
                (0, 0),
                width=(beam.major.to(u.deg) / pixscale)
                .to(u.dimensionless_unscaled)
                .value,
                height=(beam.minor.to(u.deg) / pixscale)
                .to(u.dimensionless_unscaled)
                .value,
                # PA is 90 deg offset from x-y axes by convention
                # (it is angle from NCP)
                angle=(beam.pa + 90 * u.deg).to(u.deg).value,
                edgecolor="k",
                fc="None",
                lw=1,
                alpha=0.1,
            )
            ax.add_artist(ellipse)
        ellipse = Ellipse(
            (0, 0),
            width=(cmn_beam.major.to(u.deg) / pixscale)
            .to(u.dimensionless_unscaled)
            .value,
            height=(cmn_beam.minor.to(u.deg) / pixscale)
            .to(u.dimensionless_unscaled)
            .value,
            # PA is 90 deg offset from x-y axes by convention
            # (it is angle from NCP)
            angle=(cmn_beam.pa + 90 * u.deg).to(u.deg).value,
            edgecolor="r",
            fc="None",
            lw=2,
            alpha=1,
        )
        ax.add_artist(ellipse)
        label = f"BMAJ={cmn_beam.major.to(u.arcsec).round()}, BMIN={cmn_beam.minor.to(u.arcsec).round()}, BPA={cmn_beam.pa.to(u.deg).round()}"
        plt.plot([np.nan], [np.nan], "r", label=label)
        plt.xlim(-0.2 * 60, 0.2 * 60)
        plt.ylim(-0.2 * 60, 0.2 * 60)
        plt.xlabel("$\Delta$ RA [arcsec]")
        plt.ylabel("$\Delta$ DEC [arcsec]")
        plt.legend()
        plt.show()

    beam_dict.update({"common_beam": cmn_beam})
    return beam_dict


def smooth(inps, new_beam, verbose=False):
    """Smooth an image to a new resolution.

    Arguments:
        inps {tuple} -- (filename, old_beam, flag)
            filename {str} -- name of fits file
            old_beam {Beam} -- beam of image
            flag {bool} -- flag this channel
        new_beam {Beam} -- Target resolution

    Keyword Arguments:
        verbose {bool} -- Verbose output (default: {False})

    Returns:
        newim {array} -- Image smoothed to new resolution.
    """
    filename, old_beam, flag = inps
    if verbose:
        logger.info(f"Getting image data from {filename}")
    with fits.open(filename, memmap=True, mode="denywrite") as hdu:
        dx = hdu[0].header["CDELT1"] * -1 * u.deg
        dy = hdu[0].header["CDELT2"] * u.deg
        nx, ny = hdu[0].data[0, 0, :, :].shape[0], hdu[0].data[0, 0, :, :].shape[1]
        image = np.squeeze(hdu[0].data).astype("float64")

    if flag:
        newim = np.ones_like(image) * np.nan

    else:
        con_beam = new_beam.deconvolve(old_beam)
        fac, amp, outbmaj, outbmin, outbpa = au2.gauss_factor(
            [
                con_beam.major.to(u.arcsec).value,
                con_beam.minor.to(u.arcsec).value,
                con_beam.pa.to(u.deg).value,
            ],
            beamOrig=[
                old_beam.major.to(u.arcsec).value,
                old_beam.minor.to(u.arcsec).value,
                old_beam.pa.to(u.deg).value,
            ],
            dx1=dx.to(u.arcsec).value,
            dy1=dy.to(u.arcsec).value,
        )

        if verbose:
            logger.info(f"Smoothing so beam is {new_beam}")
            logger.info(f"Using convolving beam {con_beam}")

        pix_scale = dy

        gauss_kern = con_beam.as_kernel(pix_scale)

        conbm1 = gauss_kern.array / gauss_kern.array.max()

        newim = scipy.signal.convolve(image, conbm1, mode="same")

        newim *= fac
        if verbose:
            sys.stdout.flush()
    return newim


def writecube(data, freqs, header, beam, band, stoke, field, outdir, verbose=True):
    """Write cube to disk.

    Arguments:
        data {array} -- Datacube to save.
        freqs {array} -- Frequency list correspondng to cube.
        header {header} -- Header for image.
        beam {Beam} -- New common resolution beam.
        band {int} -- ATCA band name.
        stoke {str} -- Stokes parameter.
        field {str} -- QUOCKA field name.
        outdir {str} -- Directory to save output.

    Keyword Arguments:
        verbose {bool} -- Verbose output (default: {True})
    """
    # Make filename
    outfile = f"{field}.{band}.{stoke}.cutout.bandcube.fits"

    # Sort data
    sort_idx = freqs.argsort()
    freqs_sorted = freqs[sort_idx]
    data_sorted = data[sort_idx, :, :]

    # Make header
    d_freq = np.nanmedian(np.diff(freqs_sorted))
    del header["HISTORY"]
    header = beam.attach_to_header(header)
    header["CRVAL3"] = freqs_sorted[0].to_value()
    header["CDELT3"] = d_freq.to_value()
    header["COMMENT"] = "DO NOT rely on this header for correct frequency data!"
    header["COMMENT"] = "Use accompanying frequency text file."

    # Save the data
    fits.writeto(f"{outdir}/{outfile}", data_sorted, header=header, overwrite=True)
    if verbose:
        logger.info(f"Saved cube to {outdir}/{outfile}")

    if stoke == "i":
        freqfile = f"{field}.{band}.bandcube.frequencies.txt"
        np.savetxt(f"{outdir}/{freqfile}", freqs_sorted.to_value())
        if verbose:
            logger.info(f"Saved frequencies to {outdir}/{freqfile}")


def main(pool, args, verbose=False):
    """Main script"""
    bands = [2100]
    stokes = ["i", "q", "u", "v"]
    datadir = args.datadir
    if datadir is not None:
        if datadir[-1] == "/":
            datadir = datadir[:-1]

    outdir = args.outdir
    if outdir is not None:
        if outdir[-1] == "/":
            outdir = outdir[:-1]
    elif outdir is None:
        outdir = datadir

    # Glob out files
    data_dict = {}
    for band in bands:
        data_dict.update({band: {}})
        for stoke in stokes:
            data_dict[band].update(
                {stoke: sorted(glob(f"{datadir}/{args.field}.{band}.*.{stoke}.fits"))}
            )

    # Check files were found
    for band in bands:
        for stoke in stokes:
            if len(data_dict[band][stoke]) == 0:
                raise Exception(f"No Band {band} Stokes {stoke} files found!")
    # Get common beams
    for band in tqdm(
        bands, desc="Finding commmon beam per band", disable=(not verbose)
    ):
        beam_dict = getmaxbeam(
            data_dict,
            band,
            cutoff=args.cutoff * u.arcsec,
            tolerance=args.tolerance,
            nsamps=args.nsamps,
            epsilon=args.epsilon,
            verbose=verbose,
            debug=args.debug,
        )
        if verbose:
            logger.info(f"Common beam for band {band} is {beam_dict['common_beam']}")
        data_dict[band].update(beam_dict)

    # Do the convolution
    if verbose:
        logger.info("Making cube per band")
    for band in bands:
        if verbose:
            logger.info(f"Band: {band}")
        smooth_partial = partial(
            smooth, new_beam=data_dict[band]["common_beam"], verbose=False
        )
        for stoke in stokes:
            if verbose:
                logger.info(f"Stokes: {stoke}")
            data = list(
                tqdm(
                    pool.imap(
                        smooth_partial,
                        zip(
                            data_dict[band][stoke],
                            data_dict[band][stoke + "_beams"],
                            data_dict[band][stoke + "_flags"],
                        ),
                    ),
                    total=len(data_dict[band][stoke]),
                    disable=(not verbose),
                    desc="Smoothing channels",
                )
            )
            data = np.array(data)
            freqs = data_dict[band][stoke + "_freqs"]
            head_temp = fits.getheader(data_dict[band][stoke][0])
            beam = data_dict[band]["common_beam"]
            if not args.dryrun:
                # Save the cubes
                writecube(
                    data,
                    freqs,
                    head_temp,
                    beam,
                    band,
                    stoke,
                    args.field,
                    outdir,
                    verbose=verbose,
                )

    if verbose:
        logger.info("Done!")


def cli():
    """Command-line interface"""
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Produce common resolution cubes for QUOCKA data.

    A seperate cube per band will be produced, along with frequency text files.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "datadir",
        metavar="datadir",
        type=str,
        help="Directory containing a single QUOCKA field images.",
    )

    parser.add_argument("field", metavar="field", type=str, help="QUOCKA field name.")

    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default=None,
        help="(Optional) Save cubes to different directory [datadir].",
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        dest="cutoff",
        type=float,
        default=15,
        help="Flags channels with BMAJ > cutoff in arcsec [15]",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="verbose output [False].",
    )

    parser.add_argument(
        "-d",
        "--dryrun",
        dest="dryrun",
        action="store_true",
        help="Compute common beam and stop [False].",
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Show debugging plots [False].",
    )

    parser.add_argument(
        "-t",
        "--tolerance",
        dest="tolerance",
        type=float,
        default=0.0001,
        help="tolerance for radio_beam.commonbeam.",
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        dest="epsilon",
        type=float,
        default=0.0005,
        help="epsilon for radio_beam.commonbeam.",
    )

    parser.add_argument(
        "-n",
        "--nsamps",
        dest="nsamps",
        type=int,
        default=200,
        help="nsamps for radio_beam.commonbeam.",
    )
    parser.add_argument(
        "-l",
        "--log_file",
        help="Name of output log file [default log.txt]",
        default="log.txt",
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi", dest="mpi", default=False, action="store_true", help="Run with MPI."
    )

    args = parser.parse_args()

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    with schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores) as pool:
        if args.mpi:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        # make it so we can use imap in serial and mpi mode
        if not isinstance(pool, schwimmbad.MultiPool):
            pool.imap = pool.map

        verbose = args.verbose

        main(pool, args, verbose=verbose)


if __name__ == "__main__":
    cli()
