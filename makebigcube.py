#!/usr/bin/env python
"""Make big QUOCKA cubes"""

import schwimmbad
import sys
from glob import glob
from tqdm import tqdm
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
import au2
import scipy.signal
import numpy as np
from functools import partial
import reproject as rpj
from IPython import embed


def getmaxbeam(file_dict, tolerance=0.0001, nsamps=200, epsilon=0.0005, verbose=False):
    """Find common beam

    Arguments:
        file_dict {dict} -- Filenames for each bandcube.

    Keyword Arguments:
        tolerance {float} -- See common_beam (default: {0.0001})
        nsamps {int} -- See common_beam (default: {200})
        epsilon {float} -- See common_beam (default: {0.0005})
        verbose {bool} -- Verbose output (default: {False})

    Returns:
        cmn_beam {Beam} -- Common beam
    """
    if verbose:
        print('Finding common beam...')
    stokes = ['i', 'q', 'u', 'v']
    beam_dict = {}
    beams = []
    for stoke in stokes:
        for file in file_dict[stoke]:
            header = fits.getheader(file, memmap=True)
            beam = Beam.from_fits_header(header)
            beams.append(beam)
    beams = Beams(
        [beam.major.value for beam in beams]*u.deg,
        [beam.minor.value for beam in beams]*u.deg,
        [beam.pa.value for beam in beams]*u.deg
    )

    try:
        cmn_beam = beams.common_beam(
            tolerance=tolerance, epsilon=epsilon, nsamps=nsamps)
    except BeamError:
        if verbose:
            print("Couldn't find common beam with defaults")
            print("Trying again with smaller tolerance")
        cmn_beam = beams.common_beam(
            tolerance=tolerance*0.1, epsilon=epsilon, nsamps=nsamps)
    return cmn_beam


def getdata(file_dict, new_beam, stoke, verbose=False):
    """Get data from band cubes

    Arguments:
        file_dict {dict} -- Filenames for each bandcube
        new_beam {Beam} -- Target common resolution
        stoke {str} -- Stokes parameter

    Keyword Arguments:
        verbose {bool} -- Verbose output (default: {False})

    Returns:
        data {dict} -- Smoothed data and metadata
    """
    freqs = []
    for freqfile in file_dict['freqs']:
        freqs.append(np.loadtxt(freqfile))
    freqs = np.hstack(freqs)*u.Hz
    sort_idx = freqs.argsort()
    freqs_sorted = freqs[sort_idx]

    datacube = []
    beams = []
    headers = []

    for file in file_dict[stoke]:
        data = fits.getdata(file)
        datacube.append(data)
        header = fits.getheader(file, memmap=True)
        beam = Beam.from_fits_header(header)
        for i in range(data.shape[0]):
            beams.append(beam)
            headers.append(header)

    beams_sorted = [beams[idx] for idx in sort_idx]
    beams_sorted = Beams(
        [beam.major.value for beam in beams_sorted]*u.deg,
        [beam.minor.value for beam in beams_sorted]*u.deg,
        [beam.pa.value for beam in beams_sorted]*u.deg
    )

    datacube = np.vstack(datacube)

    datacube_sorted = datacube[sort_idx]

    with fits.open(file_dict[stoke][0], memmap=True, mode='denywrite') as hdu:
        targ_header = hdu[0].header
        dx = targ_header['CDELT1']*-1*u.deg
        dy = targ_header['CDELT2']*u.deg

    dx_lst = [dx for i in range(len(freqs_sorted))]
    dy_lst = [dy for i in range(len(freqs_sorted))]
    data = {
        "cube": datacube_sorted,
        "freqs": freqs_sorted,
        "beams": beams_sorted,
        "dx": dx_lst,
        "dy": dy_lst,
        "target header": targ_header,
        "headers": headers

    }
    return data


def smooth(inps, new_beam, verbose=False):
    """Smooth cubes to common beam

    Arguments:
        inps {tuple} -- Inputs
            image {array} -- image plane from cube
            dx {Quantity} -- Pixel scale in x-direction
            dy {Quantity} -- Pixel scale in y-direction
            old_beam {Beam} -- Current resolution of image
        new_beam {Beam} -- Target common resolution

    Keyword Arguments:
        verbose {bool} -- Verbose output (default: {False})

    Returns:
        newim {array} -- Smoothed image plane
    """
    image, dx, dy, old_beam = inps
    if new_beam == old_beam:
        return image
    else:
        con_beam = new_beam.deconvolve(old_beam)
        fac, amp, outbmaj, outbmin, outbpa = au2.gauss_factor(
            [
                con_beam.major.to(u.arcsec).value,
                con_beam.minor.to(u.arcsec).value,
                con_beam.pa.to(u.deg).value
            ],
            beamOrig=[
                old_beam.major.to(u.arcsec).value,
                old_beam.minor.to(u.arcsec).value,
                old_beam.pa.to(u.deg).value
            ],
            dx1=dx.to(u.arcsec).value,
            dy1=dy.to(u.arcsec).value
        )

        if verbose:
            print(f'Smoothing so beam is', new_beam)
            print(f'Using convolving beam', con_beam)

        pix_scale = dy

        gauss_kern = con_beam.as_kernel(pix_scale)

        conbm1 = gauss_kern.array/gauss_kern.array.max()

        newim = scipy.signal.convolve(image, conbm1, mode='same')

        newim *= fac
        if verbose:
            sys.stdout.flush()
        return newim


def writecube(data, beam, stoke, field, outdir, verbose=False):
    """Write cubes to disk

    Arguments:
        data {dict} -- Image and frequency data and metadata
        beam {Beam} -- New common resolution
        stoke {str} -- Stokes parameter
        field {str} -- Field name
        outdir {str} -- Output directory

    Keyword Arguments:
        verbose {bool} -- Verbose output (default: {False})
    """
    # Make filename
    outfile = f"{field}.{stoke}.cutout.bigcube.fits"

    # Make header
    d_freq = np.nanmedian(np.diff(data['freqs']))
    header = data['target header']
    header = beam.attach_to_header(header)
    header['CRVAL3'] = data['freqs'][0].to_value()
    header['CDELT3'] = d_freq.to_value()

    # Save the data
    fits.writeto(f'{outdir}/{outfile}', data['regrid cube'],
                 header=header, overwrite=True)
    if verbose:
        print("Saved cube to", f'{outdir}/{outfile}')

    if stoke == 'i':
        freqfile = f"{field}.bigcube.frequencies.txt"
        np.savetxt(f"{outdir}/{freqfile}", data['freqs'].to_value())
        if verbose:
            print("Saved frequencies to", f"{outdir}/{freqfile}")


def regrid_worker(inps, target_wcs):

    image, imag_wcs = inps
    if np.isnan(image).all():
        newim = image
    else:
        newim = rpj.reproject_exact(
            (image, imag_wcs),
            target_wcs,
            return_footprint=False,
            shape_out=image.shape,
            parallel=False
        )
    return newim


def regrid(data, pool, verbose=False):

    imag_wcss = [WCS(header).celestial for header in data['headers']]

    targ_wcs = WCS(data['target header']).celestial

    worker_partial = partial(regrid_worker, target_wcs=targ_wcs)

    im_list = list(
        tqdm(
            pool.imap(
                worker_partial,
                zip(data['smooth cube'], imag_wcss)
            ),
            total=len(imag_wcss),
            desc='Regridding channels',
            disable=(not verbose)
        )
    )

    cube = np.array(im_list)
    return cube


def main(pool, args, verbose=False):
    """Main script
    """
    bands = [2100, 5500, 7500]
    stokes = ['i', 'q', 'u', 'v']
    datadir = args.datadir
    if datadir is not None:
        if datadir[-1] == '/':
            datadir = datadir[:-1]

    outdir = args.outdir
    if outdir is not None:
        if outdir[-1] == '/':
            outdir = outdir[:-1]
    elif outdir is None:
        outdir = datadir

    # Glob out files
    file_dict = {}
    for stoke in stokes:
        file_dict.update(
            {
                stoke: sorted(
                    glob(f'{datadir}/{args.field}.*.{stoke}.cutout.bandcube.fits')
                )
            }
        )
    file_dict.update(
        {
            'freqs': sorted(
                glob(f'{datadir}/{args.field}.*.bandcube.frequencies.txt')
            )
        }
    )
    if args.target is None:
        new_beam = getmaxbeam(file_dict,
                              tolerance=args.tolerance,
                              nsamps=args.nsamps,
                              epsilon=args.epsilon,
                              verbose=verbose)
    elif args.target is not None:
        new_beam = Beam(major=args.target*u.arcsec,
                        minor=args.target*u.arcsec,
                        pa=0*u.deg)
    if verbose:
        print('Common beam is', new_beam)
    # Get data from files
    data_dict = {stoke: {} for stoke in stokes}
    for stoke in tqdm(stokes, desc='Getting data', disable=(not verbose)):
        data = getdata(file_dict,
                       new_beam,
                       stoke,
                       verbose=verbose)
        data_dict[stoke].update(data)

    # Smooth to common beam
    for stoke in tqdm(stokes, desc='Smoothing data', disable=(not verbose)):
        smooth_partial = partial(smooth,
                                 new_beam=new_beam,
                                 verbose=False
                                 )
        data = list(
            tqdm(
                pool.imap(smooth_partial,
                          zip(data_dict[stoke]['cube'],
                              data_dict[stoke]['dx'],
                              data_dict[stoke]['dy'],
                              data_dict[stoke]['beams'])
                          ),
                total=len(data_dict[stoke]['freqs']),
                disable=(not verbose),
                desc='Smoothing channels'
            )
        )
        data = np.array(data)
        data_dict[stoke].update(
            {
                "smooth cube": data
            }
        )
    
    # Regrid to match common beam header
    for stoke in tqdm(stokes, desc='Regridding data', disable=(not verbose)):
        rgrd_cube = regrid(data_dict[stoke], pool, verbose=verbose)
        data_dict[stoke].update(
            {
                "regrid cube": rgrd_cube
            }
        )
    if not args.dryrun:
        # Save the cubes
        for stoke in tqdm(stokes, desc='Writing cubes', disable=(not verbose)):
            writecube(data_dict[stoke],
                      new_beam,
                      stoke,
                      args.field,
                      outdir,
                      verbose=verbose)

    if verbose:
        print('Done!')


def cli():
    """Command-line interface
    """
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Produce common resolution cubes for QUOCKA data.

    Combines seperate cubes per band into single cube.
    Make sure to run makecube.py first!

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        'datadir',
        metavar='datadir',
        type=str,
        help='Directory containing a single QUOCKA field images.')

    parser.add_argument(
        'field',
        metavar='field',
        type=str,
        help='QUOCKA field name.')

    parser.add_argument(
        '-o',
        '--outdir',
        dest='outdir',
        type=str,
        default=None,
        help='(Optional) Save cubes to different directory [datadir].')

    parser.add_argument(
        '--target',
        dest='target',
        type=float,
        default=None,
        help='Target resoltion (circular beam, BMAJ) in arcmin [None].')

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="verbose output [False].")

    parser.add_argument(
        "-d",
        "--dryrun",
        dest="dryrun",
        action="store_true",
        help="Compute common beam and stop [False].")

    parser.add_argument(
        "-t",
        "--tolerance",
        dest="tolerance",
        type=float,
        default=0.0001,
        help="tolerance for radio_beam.commonbeam.")

    parser.add_argument(
        "-e",
        "--epsilon",
        dest="epsilon",
        type=float,
        default=0.0005,
        help="epsilon for radio_beam.commonbeam.")

    parser.add_argument(
        "-n",
        "--nsamps",
        dest="nsamps",
        type=int,
        default=200,
        help="nsamps for radio_beam.commonbeam.")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    if args.mpi:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    # make it so we can use imap in serial and mpi mode
    if not isinstance(pool, schwimmbad.MultiPool):
        pool.imap = pool.map

    verbose = args.verbose

    main(pool, args, verbose=verbose)
    pool.close()


if __name__ == "__main__":
    cli()
