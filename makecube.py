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


def readfiles(datadir):
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
    print 'Getting data -- Splitting by Stokes...'
    files = glob(datadir+'/'+'*.fits')
    filelist = []
    chanlist = []
    freqlist = []
    stoklist = []
    for f in files:
        source = f[0:9]
        filelist.append(f)
        freq = f[10:14]
        freqlist.append(int(freq))
        chan = f[15:19]
        chanlist.append(int(chan))
        stok = f[20]
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

def getfreq(datadir, datalist):
    '''Parse the frequency information from FITS images.

    Loops over files in data directory, pulls out the frequency information,
    and sorts file list by frequency.

    Args:
        datadir: Sting of path to directory containing data in FITS file
        format. No trailing '/'!
        datalist: List containing [ilist, qlist, ulist], as produced by
        readfiles

    Returns:
        freqi: List of frequencies as given by Stokes I images
        sortlist: List of [sortlisti, sortlistq, sortlistu]. Each of which
        contain the Stokes I, Q, U filelist, respectively, sorted by
        frequency.

    '''
    print 'Getting frequencies from Stokes I headers...'
    ilist, qlist, ulist = datalist
    freqi = []
    for f in ilist:
        with fits.open(datadir + f, mode='denywrite')[0] as hdu:
            freqi.append(hdu.header['CRVAL3'])
    freqq = []
    for f in qlist:
        with fits.open(datadir + f, mode='denywrite')[0] as hdu:
            freqq.append(hdu.header['CRVAL3'])

    frequ = []
    for f in ulist:
        with fits.open(datadir + f, mode='denywrite')[0] as hdu:
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

def getbigframe(datadir, sortlist):
    '''
    Get all BMAJ from files.
    Find biggest one using the lowest frequency.
    Set that to be the common smoothing FWHM.
    '''

    freqlist = []
    for f in sortlist:
        with fits.open(datadir + f, mode='denywrite')[0] as hdu:
            freqlist.append(hdu.header['CRVAL3'])

    freqlist = np.array(freqlist)

    # Look for lowest frequency
    loc = np.argmin(freqlist)
    bigfile = datadir + sortlist[loc]
    bighdu =  fits.open(bigfile, mode='denywrite')[0]
    return bighdu

def smoothloop(args):
    bighdu, datadir, f = args

    with fits.open(datadir + f)[0] as hdu:
        hdu = hdulist[0]
        head = hdu.header
        data = hdu.data[0,0,:,:]

    grid = abs(head['CDELT1'])
    freq = head['CRVAL3']

    freq_r = bighdu.header['CRVAL3']

    # Reference resolution
    bmaj_r = bighdu.header['BMAJ']
    bmin_r = bighdu.header['BMIN']

    # Old resolution
    bmaj_o = head['BMAJ']
    bmin_o = head['BMIN']

    # New common resolution
    bmaj_n = np.round(bmaj_r*60, decimals=3)/60.
    bmin_n = np.round(bmin_r*60, decimals=3)/60.

    # Sanity check -- New resolution should be greater than old
    if bmaj_n <= bmaj_o:
        print 'continue'
        pass

    else:
        pa = np.deg2rad(head['BPA'])

        conv_width_maj = np.sqrt(bmaj_n ** 2 - bmaj_o ** 2)
        conv_width_min = np.sqrt(bmin_n ** 2 - bmin_o** 2)

        sig_min = conv_width_min / (2 * np.sqrt(2 * np.log(2))) / grid
        sig_maj = conv_width_maj / (2 * np.sqrt(2 * np.log(2))) / grid

        g = Gaussian2DKernel(
            sig_min,
            sig_maj,
            theta=pa)
        data = convolve(data, g, boundary='extend') * (2 * np.pi * sig_min * sig_maj)
        return [data, freq]

def smcube(pool, freq_r, datadir, sortlist):
    '''
    Smooth data to common spatial resolution.
    hpbw_r -- reference FWHM (arcmin)?
    freq_r -- reference frequency
    hpbw_n -- new common FWHM (arcmin)?
    TO-DO: Write freq file
    '''
    print 'Smoothing data to HPBW of %f' % hpbw_n + 'arcmin'
    print 'Entering loop'
    tic = timeit.default_timer()
    output = pool.map(smoothloop,
        ([freq_r, datadir, f] for f in sortlist))
    print 'Loop done'
    toc = timeit.default_timer()
    print 'Time taken = %f' % (toc - tic)

    output = [x for x in output if x is not None]

    datacube = []
    freqs = []
    for i in range(len(output)):
        datacube.append(output[i][0])
        freqs.append(output[i][1])

    freqs = np.array(freqs)
    datacube = np.array(datacube)
    return datacube, freqs

def writetodisk(datadir, smoothcube, source, stoke, sortlist, hpbw_n, freqs):
    '''
    Write data to FITS file.
    TO-DO: Proper headers, proper filenames
    '''
    headfile = sortlist[0][0]
    hdulist = fits.open(datadir+headfile)
    hdu = hdulist[0]
    head = hdu.header
    hdulist.close()
    targ_head = head.copy()
    del targ_head[0:8]
    bad_cards = ['CRPIX4', 'CDELT4', 'CRVAL4', 'CTYPE4', 'RMS',\
         'CRPIX3', 'CDELT3', 'CRVAL3']
    for card in bad_cards:
        del targ_head[card]
    new_cards = ['BMAJ', 'BMIN', 'BPA']
    new_vals = [hpbw_n, hpbw_n, 0.]
    for i in range(len(new_cards)):
        targ_head[new_cards[i]] = new_vals[i]
    print 'Written frequencies to ' + datadir+source+'.'+stoke+'.frequencies.txt'
    np.savetxt(datadir+source+'.'+stoke+'.frequencies.txt', freqs, fmt='%f')
    print 'Written FITS to ' + datadir+source+'.'+stoke+'.smooth.fits'
    fits.writeto(datadir+source+'.'+stoke+'.smooth.fits', smoothcube, targ_head)

def main(pool, datadir):
    print 'Combining data in ' + datadir
    source, ilist, qlist, ulist = readfiles(datadir)

    freqlist, sortlist = getfreq(datadir, [ilist, qlist, ulist])
    bighdu = getbigframe(datadir, sortlist[0])

    for i in range(len(sortlist)):
        print 'Smoothing...'
        if i==0:
            stoke = 'i'
        if i==1:
            stoke = 'q'
        if i==2:
            stoke = 'u'
        print 'Stokes ' + stoke
        smoothcube, freqs = smcube(pool, bighdu, datadir, sortlist[i])
        print 'Writing to disk...'
        writetodisk(datadir, smoothcube, source, stoke, sortlist, hpbw_n, freqs)
    pool.close()
    print 'Done!'


if __name__ == "__main__":
    import argparse
    import schwimmbad

    # Help string to be shown using the -h option
    descStr = """
    Makes a cube from individual channel maps of Quocka observations.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("datadir", metavar="datadir", nargs=1, default='.',
                        type=str, help="Directory containing data.")
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

    datadir = args.datadir[0]

    main(pool, datadir)









