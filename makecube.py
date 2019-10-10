#!/usr/bin/env python

from __future__ import division, print_function
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import os
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import timeit
import pdb
from glob import glob
import warnings


def readfiles(datadir, verbose=False):
    """Reads list of FITS files from data-directory.

    Finds and returns the list of FITS files in a directory, and splits them
    by Stokes. Filenames must comply with the Quocka filename convention.

    Args:
        datadir: Sting of path to directory containing data in FITS file
        format.
        No trailing '/'!

    Returns:
        source: Name of source (str).
        ilist: List of Stokes I images.
        qlist: List of Stokes Q images.
        ulist: List of Stokes U images.
    """

    if verbose:
        print('Getting data -- Splitting by Stokes...')
    files = glob(datadir+'/'+'*.fits')
    filelist = []
    chanlist = []
    freqlist = []
    stoklist = []
    for f in files:
        source = f[len(datadir)+1+0:9]
        filelist.append(f)
        freq = f[len(datadir)+1+10:14]
        freqlist.append(int(freq))
        chan = f[len(datadir)+1+15:19]
        chanlist.append(int(chan))
        stok = f[len(datadir)+1+20]
        stoklist.append(stok)

    stoklist = np.array(stoklist)
    filelist = np.array(filelist)
    freqlist = np.array(freqlist)
    chanlist = np.array(chanlist)

    icond = stoklist == 'i'
    qcond = stoklist == 'q'
    ucond = stoklist == 'u'

    lowcond = freqlist == 2100
    midcond = freqlist == 5500
    hihcond = freqlist == 7500

    ilist = filelist[icond]
    qlist = filelist[qcond]
    ulist = filelist[ucond]
    return source, ilist, qlist, ulist

def getfreq(datalist, verbose=False):
    """Parse the frequency information from FITS images.

    Loops over files in data directory, pulls out the frequency information,
    and sorts file list by frequency.

    Args:
        datalist: List containing [ilist, qlist, ulist], as produced by
        readfiles

    Returns:
        freqi: List of frequencies as given by Stokes I images
        sortlist: List of [sortlisti, sortlistq, sortlistu]. Each of which
        contain the Stokes I, Q, U filelist, respectively, sorted by
        frequency.

    """

    if verbose:
        print('Getting frequencies from Stokes I headers...')
    ilist, qlist, ulist = datalist
    freqi = []
    for f in ilist:
        #pdb.set_trace()
        with fits.open(f, mode='denywrite') as hdulist:
            hdu = hdulist[0]
            freqi.append(hdu.header['CRVAL3'])
    freqq = []
    for f in qlist:
        with fits.open(f, mode='denywrite') as hdulist:
            hdu = hdulist[0]
            freqq.append(hdu.header['CRVAL3'])

    frequ = []
    for f in ulist:
        with fits.open(f, mode='denywrite') as hdulist:
            hdu = hdulist[0]
            frequ.append(hdu.header['CRVAL3'])

    # Sort by frequency
    freqi = np.array(freqi)
    freqi, sortlisti = zip(*sorted(zip(freqi, ilist)))

    freqq = np.array(freqq)
    freqq, sortlistq = zip(*sorted(zip(freqq, qlist)))

    frequ = np.array(frequ)
    frequ, sortlistu = zip(*sorted(zip(frequ, ulist)))
    sortlist = [sortlisti, sortlistq, sortlistu]
    return freqi, sortlist

def getbigframe(sortlist, verbose=False):
    """Find the biggest frequency/beam

    Find biggest beamsize using the lowest frequency.
    Set that to be the common smoothing FWHM.

    Args:
        sortlist: List of [sortlisti, sortlistq, sortlistu]. Each of which
        contain the Stokes I, Q, U filelist, respectively, sorted by
        frequency.

    Returns:
        refs: List of [freq_r, bmaj_n, bmin_n]
            freq_r: Reference frequency in Hz.
            bmaj_n: New common resolution major axis.
            bmin_n: New common resolution minor axis.

    """

    freqlist = []
    for f in sortlist:
        with fits.open(f, mode='denywrite') as hdulist:
            hdu = hdulist[0]
            freqlist.append(hdu.header['CRVAL3'])

    freqlist = np.array(freqlist)

    # Look for lowest frequency
    loc = np.argmin(freqlist)
    bigfile = sortlist[loc]

    # Get reference values
    with fits.open(bigfile, mode='denywrite') as bighdulist:
        bighdu = bighdulist[0]
        # Reference frequency
        freq_r = bighdu.header['CRVAL3']
        # Reference resolution
        bmaj_r = bighdu.header['BMAJ']
        bmin_r = bighdu.header['BMIN']
        bpa_r = bighdu.header['BPA']

    # New common resolution
    bmaj_n = np.round(bmaj_r*60, decimals=3)/60.
    bmin_n = np.round(bmin_r*60, decimals=3)/60.

    refs = [freq_r, bmaj_n, bmin_n, bpa_r]
    return refs

def smoothloop(args):
    """Main loop to smooth image.

    Opens a FITS file and gets its frequency and beam info. Does the same for
    the reference file. Smooths the FITS file to a common resolution with the
    reference file.

    Args:
        args: List of [refs, f]
            refs: List of [freq_r, bmaj_n, bmin_n]
                freq_r: Reference frequency in Hz.
                bmaj_n: New common resolution major axis.
                bmin_n: New common resolution minor axis.
            f: File to open and smooth


    Returns: (in a list)
        data: Smoothed image.
        freq: Frequency of smoothed image.

    """

    refs, f = args

    # Get new/reference values
    freq_r, bmaj_n, bmin_n, bpa_r = refs

    with fits.open(f) as hdulist:
        hdu = hdulist[0]
        head = hdu.header
        data = hdu.data[0,0,:,:]

    grid = abs(head['CDELT1'])
    freq = head['CRVAL3']


    # Old resolution
    bmaj_o = head['BMAJ']
    bmin_o = head['BMIN']

    # Sanity check -- New resolution should be greater than old
    if bmaj_n <= bmaj_o:
        pass

    else:
        # Get kernel to convolve with
        conv_width_maj = np.sqrt(bmaj_n ** 2 - bmaj_o ** 2)
        conv_width_min = np.sqrt(bmin_n ** 2 - bmin_o** 2)

        # Convert to sigma in pixels
        sig_min = conv_width_min / (2 * np.sqrt(2 * np.log(2))) / grid
        sig_maj = conv_width_maj / (2 * np.sqrt(2 * np.log(2))) / grid


        # Python 2: CAN ONLY DO CIRCULAR
        g = Gaussian2DKernel(sig_maj)

        # PYTHON3 VERSION:
        #g = Gaussian2DKernel(
        #    sig_min,
        #    sig_maj,
        #    theta=bpa_r)

        warnings.warn('Can only convolve with circular beam in Py2.7')
        # Do the convolution with correct weighting
        data = convolve(data,
                        g,
                        boundary='extend') * (2 * np.pi * sig_maj * sig_maj)
        return [data, freq]

def smcube(pool, refs, sortlist, verbose=False):
    """Smooth data to common spatial resolution.

    Loop over files to smooth to a common resolution. Use pool.map syntax for
    parallelisation.

    Args:
        pool: Which pool to use for map.
        refs: List of [freq_r, bmaj_n, bmin_n]
            freq_r: Reference frequency in Hz.
            bmaj_n: New common resolution major axis.
            bmin_n: New common resolution minor axis.
        sortlist: List of [sortlisti, sortlistq, sortlistu]. Each of which
        contain the Stokes I, Q, U filelist, respectively, sorted by
        frequency.

    Returns:
        datacube: Cube of smoothed images.
        freqs: List containing frequency of each image in Hz.

    """

    if verbose:
        print('Smoothing data to HPBW of %f' % refs[1] +
            'arcmin by %f' % refs[1]+ 'arcmin')
        print('Entering loop')
        tic = timeit.default_timer()
    output = pool.map(smoothloop,
        ([refs, f] for f in sortlist))

    if verbose:
        print('Loop done')
        toc = timeit.default_timer()
        print('Time taken = %f' % (toc - tic))

    output = [x for x in output if x is not None]

    datacube = []
    freqs = []
    for i in range(len(output)):
        datacube.append(output[i][0])
        freqs.append(output[i][1])

    freqs = np.array(freqs)
    datacube = np.array(datacube)
    return datacube, freqs

def writetodisk(datadir, smoothcube, source, stoke, sortlist, refs, freqs, verbose=False):
    """Write data to disk.

    Writes smoothed cubes to a FITS file per Stokes, and a list of frequencies
    to a text file.

    Args:
        datadir: Sting of path to directory containing data in FITS file
        format. No trailing '/'!
        smoothcube: Cube of smoothed images.
        source: Name of source (str).
        stoke: String of 'i', 'q' or 'u' corresponding to Stokes parameter.
        sortlist: List of [sortlisti, sortlistq, sortlistu]. Each of which
        contain the Stokes I, Q, U filelist, respectively, sorted by
        frequency.
        refs: List of [freq_r, bmaj_n, bmin_n]
            freq_r: Reference frequency in Hz.
            bmaj_n: New common resolution major axis.
            bmin_n: New common resolution minor axis.
        freqs: List containing frequency of each image in Hz.


    TO-DO: Proper headers, proper filenames

    """
    headfile = sortlist[0][0]
    with fits.open(headfile, mode='denywrite') as hdulist:
        hdu = hdulist[0]
        head = hdu.header
    targ_head = head.copy()
    del targ_head[0:8]
    bad_cards = ['CRPIX4', 'CDELT4', 'CRVAL4', 'CTYPE4', 'RMS','CRPIX3',
                'CDELT3', 'CRVAL3']
    for card in bad_cards:
        del targ_head[card]
    new_cards = ['BMAJ', 'BMIN', 'BPA']

    freq_r, bmaj_n, bmin_n, bpa_r = refs
    new_vals = [bmaj_n, bmaj_n, 0] # FOR CIRCULAR BEAM
    for i in range(len(new_cards)):
        targ_head[new_cards[i]] = new_vals[i]

    outfile = datadir+'/'+source+'.'+stoke+'.frequencies.txt'
    if verbose:
        print('Written frequencies to ' + outfile)
    np.savetxt(outfile, freqs, fmt='%f')

    outfile = datadir+'/'+source+'.'+stoke+'.smooth.fits'
    if verbose:
        print('Written FITS to ' + outfile)
    fits.writeto(outfile, smoothcube, targ_head)

def main(pool, datadir, verbose=False):

    if verbose:
        print('Combining data in ' + datadir)

    source, ilist, qlist, ulist = readfiles(datadir)

    datalist = [ilist, qlist, ulist]
    freqlist, sortlist = getfreq(datalist,
                                verbose=verbose)

    refs = getbigframe(sortlist[0],
                        verbose=verbose)

    stokes = ['i', 'q', 'u']
    for i,stoke in enumerate(stokes):
        if verbose:
            print('Smoothing...')
        if verbose:
            print('Stokes ' + stoke)
        smoothcube, freqs = smcube(pool,
                                    refs,
                                    sortlist[i],
                                    verbose=verbose)

        if verbose:
            print('Writing to disk...')

        writetodisk(datadir,
                    smoothcube,
                    source,
                    stoke,
                    sortlist,
                    refs,
                    freqs,
                    verbose=verbose)
    pool.close()

    if verbose:
        print('Done!')


if __name__ == "__main__":
    import argparse
    import schwimmbad

    # Help string to be shown using the -h option
    descStr = """
    Makes a cube from individual channel maps of Quocka observations. Saves
    data to a FITS file per Stokes along with a list of frequencies
    corresponding to each channel in a text file.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("datadir", metavar="datadir", nargs=1, default='.',
                        type=str, help="Directory containing data.  No trailing '/'!")

    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="verbose output [False].")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")


    args = parser.parse_args()
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    verbose=args.verbose

    if args.mpi:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    datadir = args.datadir[0]

    main(
        pool,
        datadir,
        verbose=verbose
        )









