#!/usr/bin/env python
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import os
from joblib import dump, load
import errno
import timeit

def list_files(directory, extension):
    for (dirpath, dirnames, filenames) in os.walk(directory):
        return (f for f in filenames if f.endswith('.' + extension))



def readfiles(datadir):
    '''
    Read list of fits files from data-directory.
    Sort by channel number
    '''
    files = list_files(datadir,'fits')
    filelist = []
    chanlist = []
    freqlist = []
    stoklist = []
    for f in files:
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


    #print filelist[icond & lowcond]

    sortlow_i = [temp for _,temp in sorted(zip(chanlist[lowcond & icond], filelist[lowcond & icond]))]
    sortmid_i = [temp for _,temp in sorted(zip(chanlist[midcond & icond], filelist[midcond & icond]))]
    sorthih_i = [temp for _,temp in sorted(zip(chanlist[hihcond & icond], filelist[hihcond & icond]))]

    sortlow_q = [temp for _,temp in sorted(zip(chanlist[lowcond & qcond], filelist[lowcond & qcond]), reverse=True)]
    sortmid_q = [temp for _,temp in sorted(zip(chanlist[midcond & qcond], filelist[midcond & qcond]), reverse=True)]
    sorthih_q = [temp for _,temp in sorted(zip(chanlist[hihcond & qcond], filelist[hihcond & qcond]), reverse=True)]

    sortlow_u = [temp for _,temp in sorted(zip(chanlist[lowcond & ucond], filelist[lowcond & ucond]), reverse=True)]
    sortmid_u = [temp for _,temp in sorted(zip(chanlist[midcond & ucond], filelist[midcond & ucond]), reverse=True)]
    sorthih_u = [temp for _,temp in sorted(zip(chanlist[hihcond & ucond], filelist[hihcond & ucond]), reverse=True)]

    return sortlow_q, sortmid_q, sorthih_q, sortlow_u, sortmid_u, sorthih_u

def getdata(datadir, sortlist):
    datacube = []
    FWHM_list = []
    for f in sortlist:
        print datadir + f
        hdulist = fits.open(datadir + f)
        hdu = hdulist[0]
        data = hdu.data
        head = hdu.header
        FWHM = head['BMAJ']
        FWHM_list.append(FWHM)
        hdulist.close()
        datacube.append(data)
    datacube = np.array(datacube)
    FWHM_list = np.array(FWHM_list)
    hpbw_r = np.max(FWHM_list)
    loc = np.argmax(FWHM_list)
    bigfile = datadir + sortlist[loc]
    bighdu = fits.open(bigfile)[0]
    bighead = bighdu.header
    grid = abs(bighead['CDELT1'])
    M = bighead['NAXIS3']
    freq_r = bighead['CRVAL3']
    return datacube, grid, M, hpbw_r, freq_r

def smcube(data, grid, M, hpbw_r, freq_r, hpbw_n):
    '''
    Smooth data to common spatial resolution.
    data -- datacube
    grid = abs(hdr['CDELT1'])
    M = hdr['NAXIS3']
    hpbw_r -- reference FWHM (arcmin)
    freq_r -- reference frequency (arcmin)
    hpbw_n -- new common FWHM (arcmin)
    '''
    print 'Smoothing data to HPBW of %f' % hpbw_n
    for i in tqdm(range(M)):
        freq = hdr['CRVAL3'] + (i + 1 - hdr['CRPIX3']) * hdr['CDELT3']
        hpbw_o = hpbw_r * (freq_r * 1.e6) / freq
        if hpbw_n <= hpbw_o:
            continue
        hpbw = np.sqrt(hpbw_n * hpbw_n - hpbw_o * hpbw_o) / 60.
        g = Gaussian2DKernel(hpbw / (2. * np.sqrt(2. * np.log(2.))) / grid)
        data[i] = convolve(data[i], g, boundary='extend')
    return data




if __name__ == "__main__":
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Makes a cube from individual channel maps of Quocka observations.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--d", dest="datadir", default='.',     nargs=1,
                       type=str, help="Directory containing data.")
    #group = parser.add_mutually_exclusive_group()

    #group.add_argument("--d", dest="datadir", default='.',
    #                   type=str, help="Directory containing data.")
    args = parser.parse_args()

    datadir = args.datadir[0]
    print datadir
    sortlow_q, sortmid_q, sorthih_q, sortlow_u, sortmid_u, sorthih_u = readfiles(datadir)

    datacube, grid, M, hpbw_r, freq_r = getdata(datadir, sortlow_q)

    smoothcube = smcube(datacube, grid, M, hpbw_r, freq_r, np.round(hpbw_r, decimals=3))









