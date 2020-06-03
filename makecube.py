#!/usr/bin/env python
"""Make QUOCKA cubes"""

import schwimmbad
import sys
from glob import glob
from tqdm import tqdm
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from astropy import units as u
from astropy.io import fits
import matplotlib.pyplot as plt
import au2
import scipy.signal
import numpy as np
from functools import partial
from IPython import embed


def getmaxbeam(data_dict, band, tolerance=0.0001, nsamps=200, epsilon=0.0005, verbose=False, debug=False):
    """Get largest beam
    """
    files = data_dict[band]
    stokes = ['i', 'q', 'u', 'v']
    beam_dict = {}
    for stoke in stokes:
        beams = []
        freqs = []
        for file in files[stoke]:
            header = fits.getheader(file, memmap=True)
            freqs.append(header['CRVAL3'])
            beam = Beam.from_fits_header(header)
            beams.append(beam)
        beams = Beams(
            [beam.major.value for beam in beams]*u.deg,
            [beam.minor.value for beam in beams]*u.deg,
            [beam.pa.value for beam in beams]*u.deg
        )
        beam_dict.update(
            {
                stoke+'_beams': beams,
                stoke+'_freqs': np.array(freqs)*u.Hz
            }
        )
    if debug:
        plt.figure()
        plt.title(band)
        for stoke in stokes:
            plt.plot(beam_dict[stoke+'_freqs'], beam_dict[stoke +
                                                          '_beams'].major.to(u.arcsec), label=stoke)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('BMAJ [arcsec]')
        plt.legend()
        plt.show()

        plt.figure()
        plt.title(band)
        for stoke in stokes:
            plt.plot(beam_dict[stoke+'_freqs'], beam_dict[stoke +
                                                          '_beams'].minor.to(u.arcsec), label=stoke)
        plt.xlabel('Frequency [Hz')
        plt.ylabel('BMIN [arcsec]')
        plt.legend()
        plt.show()

    bmaj = list(beam_dict['i_beams'].major.value) + \
        list(list(beam_dict['q_beams'].major.value)) + \
        list(beam_dict['u_beams'].major.value) + \
        list(beam_dict['v_beams'].major.value)
    bmin = list(beam_dict['i_beams'].minor.value) + \
        list(list(beam_dict['q_beams'].minor.value)) + \
        list(beam_dict['u_beams'].minor.value) + \
        list(beam_dict['v_beams'].minor.value)
    bpa = list(beam_dict['i_beams'].pa.value) + \
        list(list(beam_dict['q_beams'].pa.value)) + \
        list(beam_dict['u_beams'].pa.value) + \
        list(beam_dict['v_beams'].pa.value)

    big_beams = Beams(bmaj*u.deg, bmin*u.deg, bpa*u.deg)

    try:
        cmn_beam = big_beams.common_beam(
            tolerance=tolerance, epsilon=epsilon, nsamps=nsamps)
    except BeamError:
        if verbose:
            print("Couldn't find common beam with defaults")
            print("Trying again with smaller tolerance")
        cmn_beam = big_beams.common_beam(
            tolerance=tolerance*0.1, epsilon=epsilon, nsamps=nsamps)
    beam_dict.update(
        {
            'common_beam': cmn_beam
        }
    )
    return beam_dict


def smooth(inps, new_beam, verbose=False):
    filename, old_beam = inps
    if verbose:
        print(f'Getting image data from {filename}')
    with fits.open(filename, memmap=True, mode='denywrite') as hdu:
        dx = hdu[0].header['CDELT1']*-1*u.deg
        dy = hdu[0].header['CDELT2']*u.deg
        nx, ny = hdu[0].data[0, 0, :,
                             :].shape[0], hdu[0].data[0, 0, :, :].shape[1]
        image = np.squeeze(hdu[0].data).astype('float64')

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


def writecube(data, freqs, header, beam, band, stoke, field, outdir, verbose=True):
    # Make filename
    outfile = f"{field}.{band}.{stoke}.cutout.contcube.fits"

    # Sort data
    sort_idx = freqs.argsort()
    freqs_sorted = freqs[sort_idx]
    data_sorted = data[sort_idx,:,:]
    
    # Make header
    d_freq = np.nanmedian(np.diff(freqs_sorted))
    del header['HISTORY']
    header = beam.attach_to_header(header)
    header['CRVAL3'] = freqs_sorted[0].to_value()
    header['CDELT3'] = d_freq.to_value()
    header['COMMENT'] = 'DO NOT rely on this header for correct frequency data!'
    header['COMMENT'] = 'Use accompanying frequency text file.'
    
    # Save the data
    fits.writeto(f'{outdir}/{outfile}', data_sorted, header=header, overwrite=True)
    if verbose:
        print("Saved cube to", f'{outdir}/{outfile}')

    if stoke == 'i':
        freqfile = f"{field}.{band}.contcube.frequencies.txt"
        np.savetxt(f"{outdir}/{freqfile}", freqs_sorted.to_value())
        if verbose:
            print("Saved frequencies to", f"{outdir}/{freqfile}")


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
    data_dict = {}
    for band in bands:
        data_dict.update(
            {
                band: {}
            }
        )
        for stoke in stokes:
            data_dict[band].update(
                {
                    stoke: sorted(
                        glob(f'{datadir}/{args.field}.{band}.*.{stoke}.cutout.fits'))
                }
            )
    # Get common beams
    for band in tqdm(bands,
                     desc='Finding commmon beam per band',
                     disable=(not verbose)):
        beam_dict = getmaxbeam(data_dict,
                               band,
                               tolerance=args.tolerance,
                               nsamps=args.nsamps,
                               epsilon=args.epsilon,
                               verbose=verbose,
                               debug=args.debug)
        if verbose:
            print(f'Common beam for band {band} is', beam_dict['common_beam'])
        data_dict[band].update(
            beam_dict
        )

    # Do the convolution
    if verbose:
        print('Making cube per band')
    for band in bands:
        if verbose:
            print(f'Band: {band}')
        smooth_partial = partial(smooth,
                                 new_beam=data_dict[band]['common_beam'],
                                 verbose=False
                                 )
        for stoke in stokes:
            if verbose:
                print(f'Stokes: {stoke}')
            data = list(
                tqdm(
                    pool.imap(smooth_partial,
                              zip(data_dict[band][stoke],
                                  data_dict[band][stoke+'_beams'])
                              ),
                    total=len(data_dict[band][stoke]),
                    disable=(not verbose),
                    desc='Smoothing channels'
                )
            )
            data = np.array(data)
            freqs = data_dict[band][stoke+'_freqs']
            head_temp = fits.getheader(data_dict[band][stoke][0])
            beam = data_dict[band]['common_beam']
            if not args.dryrun:
                # Save the cubes
                writecube(data,
                          freqs,
                          head_temp,
                          beam,
                          band,
                          stoke,
                          args.field,
                          outdir,
                          verbose=verbose)


def cli():
    """Command-line interface
    """
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Smooth a field of 2D images to a common resolution.

    Names of output files are 'infile'.sm.fits

    NOTE: Glob is used to parse wildcards. So if you want to run on 
        *.fits, use: python beamcon_2D.py '*.fits'
        i.e. parse the wildcard as a string.

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
        '-p',
        '--prefix',
        dest='prefix',
        type=str,
        default=None,
        help='Add prefix to output filenames.')

    parser.add_argument(
        '-o',
        '--outdir',
        dest='outdir',
        type=str,
        default=None,
        help='(Optional) Save cubes to different directory.')

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
        "--debug",
        dest="debug",
        action="store_true",
        help="Show debugging plots [False].")

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
