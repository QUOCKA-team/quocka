#!/usr/bin/env python
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import os
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import timeit

def list_files(directory, extension):
    '''
    Find all files with given extension.
    '''
    for (dirpath, dirnames, filenames) in os.walk(directory):
        return (f for f in filenames if f.endswith('.' + extension))


def readfiles(datadir):
    '''
    Read list of fits files from data-directory.
    Split by Stokes.
    '''
    print 'Getting data -- Splitting by Stokes...'
    files = list_files(datadir,'fits')
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
        #print str(f)
        #print freq
        #print chan
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
    '''
    Get frequencies from headers.
    Sort file list by frequency.
    '''
    print 'Getting frequencies from Stokes I headers...'
    ilist, qlist, ulist = datalist
    freqlist = []
    for f in ilist:
        #print datadir + f
        hdulist = fits.open(datadir + f)
        hdu = hdulist[0]
        data = hdu.data
        #print data.shape
        head = hdu.header
        hdulist.close()
        freq = head['CRVAL3']
        freqlist.append(freq)
    freqlist = np.array(freqlist)
    sortlisti = np.array([temp for _,temp in sorted(zip(freqlist, ilist))])
    sortlistq = np.array([temp for _,temp in sorted(zip(freqlist, qlist))])
    sortlistu = np.array([temp for _,temp in sorted(zip(freqlist, ulist))])
    freqlist = np.array(sorted(freqlist))
    return freqlist, [sortlisti, sortlistq, sortlistu]

def getbigframe(datadir, sortlist):
    '''
    Get all BMAJ from files.
    Find biggest one.
    Set that to be the common smoothing FWHM.
    '''
    FWHM_list = []
    for f in sortlist:
        #print datadir + f
        hdulist = fits.open(datadir + f)
        hdu = hdulist[0]
        head = hdu.header
        FWHM = head['BMAJ']
        FWHM_list.append(FWHM)
        hdulist.close()
    FWHM_list = np.array(FWHM_list)
    hpbw_r = np.max(FWHM_list)
    hpbw_n = np.round(hpbw_r, decimals=3)
    loc = np.argmax(FWHM_list)
    bigfile = datadir + sortlist[loc]
    bighdu = fits.open(bigfile)[0]
    bighead = bighdu.header
    freq_r = bighead['CRVAL3']
    return hpbw_r, freq_r, hpbw_n

def smoothloop(args):
    hpbw_r, freq_r, hpbw_n, datadir, sortlist, i = \
        args[0], args[1], args[2], args[3], args[4], args[5]
    f = sortlist[i]
    #print f
    #print 'BEEP BOOP'
    #print datadir
    hdulist = fits.open(datadir + f)
    hdu = hdulist[0]
    head = hdu.header
    data = hdu.data
    data = data[0,0,:,:]
    #print data.shape
    hdulist.close()
    grid = abs(head['CDELT1'])
    freq = head['CRVAL3'] + (i + 1 - head['CRPIX3']) * head['CDELT3']
    hpbw_o = hpbw_r * (freq_r) / freq
    if hpbw_n <= hpbw_o:
        print 'continue'
    else:
        hpbw = np.sqrt(hpbw_n**2 - hpbw_o**2) #/ 60.
        g = Gaussian2DKernel(hpbw / (2. * np.sqrt(2. * np.log(2.))) / grid)
        data = convolve(data, g, boundary='extend')
    return data

def smcube(pool, hpbw_r, freq_r, hpbw_n, datadir, sortlist):
    '''
    Smooth data to common spatial resolution.
    hpbw_r -- reference FWHM (arcmin)?
    freq_r -- reference frequency
    hpbw_n -- new common FWHM (arcmin)?
    TO-DO: What are units of BMAJ?
    '''
    print 'Smoothing data to HPBW of %f' % hpbw_n
    print 'Entering loop'
    tic = timeit.default_timer()
    datacube = pool.map(smoothloop, \
        ([hpbw_r, freq_r, hpbw_n, datadir, sortlist, i] for i in range(len(sortlist))))
    print 'Loop done'
    toc = timeit.default_timer()
    print 'Time taken = %f' % (toc - tic)
    #print len(datacube)
    datacube = np.array(datacube)
    return datacube

def writetofits(datadir, smoothcube, source, stoke):
    '''
    Write data to FITS file.
    TO-DO: Proper headers, proper filenames
    '''
    print smoothcube.shape
    print 'Written to ' + datadir+source+'.'+stoke+'.smooth.fits'
    fits.writeto(datadir+source+'.'+stoke+'.smooth.fits', smoothcube)



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
    #parser.add_argument("--d", dest="datadir", default='.', nargs=1,
    #                   type=str, help="Directory containing data.")
    parser.add_argument("datadir", metavar="datadir", nargs=1, default='.',
                        type=str, help="Directory containing data.")
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")


    args = parser.parse_args()
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    datadir = args.datadir[0]
    print 'Combining data in ' + datadir
    source, ilist, qlist, ulist = readfiles(datadir)

    freqlist, sortlist = getfreq(datadir, [ilist, qlist, ulist])
    #print freqlist/1e6
    #print sortlist

    hpbw_r, freq_r, hpbw_n = getbigframe(datadir, sortlist[0])
    #print hpbw_r, freq_r, hpbw_n


    for i in range(len(sortlist)):
        'Smoothing...'
        if i==0:
            stoke = 'i'
        if i==1:
            stoke = 'q'
        if i==2:
            stoke = 'u'
        print 'Stokes ' + stoke
        smoothcube = smcube(pool, hpbw_r, freq_r, hpbw_n, datadir, sortlist[i])
        'Writing to disk...'
        writetofits(datadir, smoothcube, source, stoke)
    pool.close()
    print 'Done!'








