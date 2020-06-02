#!/usr/bin/env python
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import os
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import timeit
import pdb

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
    print('Getting data -- Splitting by Stokes...')
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
    print('Getting frequencies from Stokes I headers...')
    ilist, qlist, ulist = datalist
    freqi = []
    for f in ilist:
        #print datadir + f
        hdulist = fits.open(datadir + f)
        hdu = hdulist[0]
        data = hdu.data
        #print data.shape
        head = hdu.header
        hdulist.close()
        freq = head['CRVAL3']
        freqi.append(freq)
    freqq = []
    for f in qlist:
        #print datadir + f
        hdulist = fits.open(datadir + f)
        hdu = hdulist[0]
        data = hdu.data
        #print data.shape
        head = hdu.header
        hdulist.close()
        freq = head['CRVAL3']
        freqq.append(freq)

    frequ = []
    for f in ulist:
        #print datadir + f
        hdulist = fits.open(datadir + f)
        hdu = hdulist[0]
        data = hdu.data
        #print data.shape
        head = hdu.header
        hdulist.close()
        freq = head['CRVAL3']
        frequ.append(freq)

    freqi = np.array(freqi)
    freqi, sortlisti = list(zip(*sorted(zip(freqi, ilist))))

    freqq = np.array(freqq)
    freqq, sortlistq = list(zip(*sorted(zip(freqq, qlist))))

    frequ = np.array(frequ)
    frequ, sortlistu = list(zip(*sorted(zip(frequ, ulist))))
    #print freqlist
    return freqi, [sortlisti, sortlistq, sortlistu]

def getbigframe(datadir, sortlist):
    '''
    Get all BMAJ from files.
    Find biggest one.
    Set that to be the common smoothing FWHM.
    '''
    FWHM_list = []
    freqlist = []
    for f in sortlist:
        #print datadir + f
        hdulist = fits.open(datadir + f)
        hdu = hdulist[0]
        head = hdu.header
        FWHM = head['BMAJ']
        FWHM_list.append(FWHM)
        freq = head['CRVAL3']
        freqlist.append(freq)
        hdulist.close()
    freqlist = np.array(freqlist)
    FWHM_list = np.array(FWHM_list)
    '''
    Deprecated -- looks for biggest BMAJ
    hpbw_r = np.max(FWHM_list)*60 # In arcmin now
    hpbw_n = np.round(hpbw_r, decimals=3)
    loc = np.argmax(FWHM_list)
    bigfile = datadir + sortlist[loc]
    bighdu = fits.open(bigfile)[0]
    bighead = bighdu.header
    freq_r = bighead['CRVAL3']
    '''
    # Look for lowest frequency
    loc = np.argmin(freqlist)
    bigfile = datadir + sortlist[loc]
    bighdu = fits.open(bigfile)[0]
    bighead = bighdu.header
    freq_r = bighead['CRVAL3']
    hpbw_r = bighead['BMAJ']*60 # In arcmin now
    hpbw_n = np.round(hpbw_r, decimals=3)
    return hpbw_r, freq_r, hpbw_n

def smoothloop(args):
    #pdb.set_trace()
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
    freq = head['CRVAL3'] #+ (i + 1 - head['CRPIX3']) * head['CDELT3']
    #print freq
    hpbw_o = hpbw_r * (freq_r) / freq
    if hpbw_n <= hpbw_o:
        print('continue')
        pass
    else:
        hpbw = np.sqrt(hpbw_n**2 - hpbw_o**2) / 60.
        g = Gaussian2DKernel(hpbw / (2. * np.sqrt(2. * np.log(2.))) / grid)
        data = convolve(data, g, boundary='extend')
        return [data, freq]

def smcube(pool, hpbw_r, freq_r, hpbw_n, datadir, sortlist):
    '''
    Smooth data to common spatial resolution.
    hpbw_r -- reference FWHM (arcmin)?
    freq_r -- reference frequency
    hpbw_n -- new common FWHM (arcmin)?
    TO-DO: Write freq file
    '''
    print('Smoothing data to HPBW of %f' % hpbw_n + 'arcmin')
    print('Entering loop')
    tic = timeit.default_timer()
    output = pool.map(smoothloop, \
        ([hpbw_r, freq_r, hpbw_n, datadir, sortlist, i] for i in range(len(sortlist))))
    print('Loop done')
    toc = timeit.default_timer()
    print('Time taken = %f' % (toc - tic))
    #print len(datacube)
    output = [x for x in output if x is not None]
    #print 'BEEP BOOP'
    datacube = []
    freqs = []
    for i in range(len(output)):
        datacube.append(output[i][0])
        freqs.append(output[i][1])
    #datacube = [x for x in datacube if x is not None]
    #freqs = [x for x in freqs if x is not None]
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
    print('Written frequencies to ' + datadir+source+'.'+stoke+'.frequencies.txt')
    np.savetxt(datadir+source+'.'+stoke+'.frequencies.txt', freqs, fmt='%f')
    print('Written FITS to ' + datadir+source+'.'+stoke+'.smooth.fits')
    fits.writeto(datadir+source+'.'+stoke+'.smooth.fits', smoothcube, targ_head)



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
    #pdb.set_trace()
    datadir = args.datadir[0]
    print('Combining data in ' + datadir)
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
        print('Stokes ' + stoke)
        smoothcube, freqs = smcube(pool, hpbw_r, freq_r, hpbw_n, datadir, sortlist[i])
        'Writing to disk...'
        writetodisk(datadir, smoothcube, source, stoke, sortlist, hpbw_n, freqs)
    pool.close()
    print('Done!')








