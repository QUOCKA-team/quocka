#!/usr/bin/env python

'''
Script to generate channel images from calibrated miriad data
First version by Philippa Patterson, 10 December 2018
Updated by GH and PP 10 December 2018
'''

import sys
import argparse
import configparser
import glob
import os
from subprocess import call
import numpy as np
from astropy.io import fits

sourcename = sys.argv[1]
mfsdir = '../../scal_makeup/'
vislist = sorted(glob.glob(sourcename+'.????'))
print(vislist)


def getnoise(img_name):
    hdu = fits.open(img_name)
    data = hdu[0].data[0, 0]
    rms_initial = np.std(data)
    rms = np.std(
        data[np.logical_and(data > -2.5*rms_initial, data < 2.5*rms_initial)])
    peak_min = np.amin(data)
    hdu.close()
    return rms, peak_min


for vis in vislist:
    freqband = vis.split('.')[-1]
    if freqband == '2100':
        selstring = ''
        imsize = 4096
        cellsize = 1
# 	imsize = 500
    elif freqband == '5500':
        selstring = 'select=-ant(6)'
        imsize = 4096
        cellsize = 0.7
# 	imsize = 1200
    elif freqband == '7500':
        selstring = 'select=-ant(6)'
        imsize = 4096
        cellsize = 0.5
# 	imsize = 1600
    else:
        print('Which frequency is this?')
        exit(1)

    call(['invert', 'vis=%s.%s' % (sourcename, freqband),
          'map=%s.d.%s.mfs.i' % (sourcename, freqband),
          'beam=%s.beam.%s.mfs' % (sourcename, freqband),
          'imsize=%s' % (imsize), 'cell=%s' % (
              cellsize), 'robust=0.5', 'stokes=i', selstring,
          'options=mfs,double,sdb'],
         stdin=None, stdout=None, stderr=None, shell=False)

    call(['fits', 'in=%s%s.%s.p2.fits' % (mfsdir, sourcename, freqband), 'op=xyin',
          'out=%s.%s.p2.fits.map' % (sourcename, freqband)],
         stdin=None, stdout=None, stderr=None, shell=False)
    # call(['fits','op=xyout','in=%s.d.%s.mfs.i'%(sourcename,freqband),'out=%s.d.%s.mfs.i.fits'%(sourcename,freqband)],
    # stdin=None, stdout=None, stderr=None, shell=False)
    # imnoise = getnoise('%s.d.%s.mfs.i.fits'%(sourcename,freqband))
    imnoise, peak_min = getnoise(mfsdir+sourcename+'.'+freqband+'.p2.fits')
    mask_level = np.amax([10*imnoise, -peak_min*1.5])
    maskname = '%s.%s.mask' % (sourcename, freqband)
    regridname = '%s.%s.regrid' % (sourcename, freqband)

    call(['regrid', 'in=%s.%s.p2.fits.map' % (sourcename, freqband), 'out=%s' % (regridname), 'tin=%s.d.%s.mfs.i' % (sourcename, freqband)],
         stdin=None, stdout=None, stderr=None, shell=False)
    call(['maths', 'exp=<%s>' % (regridname), 'mask=<%s>.gt.%f' % (regridname, mask_level),
          'out=%s' % (maskname)], stdin=None, stdout=None, stderr=None, shell=False)

    call(['fits', 'in=%s' % (regridname), 'op=xyout', 'out=%s.fits' %
          (regridname)], stdin=None, stdout=None, stderr=None, shell=False)
    call(['fits', 'in=%s' % (maskname), 'op=xyout', 'out=%s.fits' %
          (maskname)], stdin=None, stdout=None, stderr=None, shell=False)
    call(['rm', '-rf', '%s.d.%s.mfs.i' % (sourcename, freqband)])
    call(['rm', '-rf', '%s.beam.%s.mfs' % (sourcename, freqband)])
    call(['rm', '-rf', '%s.%s.p2.fits.map' % (sourcename, freqband)])
    call(['rm', '-rf', '%s' % (regridname)])
# 	# XZ: print the image noise, and try changing the cutoff level
# #     print 'Image noise is:', imnoise
# 	call(['mfclean','map=%s.d.%s.mfs.i'%(sourcename,freqband),
# 		'beam=%s.beam.%s.mfs'%(sourcename,freqband),
# 		'out=%s.model.%s.mfs'%(sourcename,freqband),
# 		'cutoff=%f'%(5.*imnoise),'niters=10000', "region='perc(90)'"],
# 		stdin=None, stdout=None, stderr=None, shell=False)
# 	call(['restor','map=%s.d.%s.mfs.i'%(sourcename,freqband),
# 		'beam=%s.beam.%s.mfs'%(sourcename,freqband),
# 		'out=%s.restor.%s.mfs'%(sourcename,freqband),
# 		'model=%s.model.%s.mfs'%(sourcename,freqband)],
# 		stdin=None, stdout=None, stderr=None, shell=False)
# 	call(['fits','op=xyout','in=%s.restor.%s.mfs'%(sourcename,freqband),'out=%s.restor.%s.mfs.fits'%(sourcename,freqband)],
# 		stdin=None, stdout=None, stderr=None, shell=False)
# 	imnoise = getnoise('%s.restor.%s.mfs.fits'%(sourcename,freqband))
# 	maskname = '%s.mask.%s.mfs'%(sourcename,freqband)
# 	call(['maths','exp=<%s.restor.%s.mfs>'%(sourcename,freqband),'mask=<%s.restor.%s.mfs>.gt.%f'%(sourcename,freqband,10.*imnoise),
# 		'out=%s'%(maskname)],stdin=None, stdout=None, stderr=None, shell=False)
# 	call(['rm','-rf','%s.d.%s.mfs.i'%(sourcename,freqband)])
# 	call(['rm','-rf','%s.d.%s.mfs.i.fits'%(sourcename,freqband)])
# 	call(['rm','-rf','%s.beam.%s.mfs'%(sourcename,freqband)])
# 	call(['rm','-rf','%s.model.%s.mfs'%(sourcename,freqband)])
# 	call(['rm','-rf','%s.restor.%s.mfs'%(sourcename,freqband)])
# #     call(['rm','-rf','%s.restor.%s.mfs.fits'%(sourcename,freqband)])

    for i in range(1, 2049, 10):
        call(['invert', 'vis=%s.%s' % (sourcename, freqband),
              'map=%s.d.%s.%04d.i' % (sourcename, freqband, i)+',%s.d.%s.%04d.q' % (sourcename, freqband, i) +
              ',%s.d.%s.%04d.u' % (sourcename, freqband, i) +
              ',%s.d.%s.%04d.v' % (sourcename, freqband, i),
              'beam=%s.beam.%s.%04d' % (sourcename, freqband, i),
              'imsize=%s' % (imsize), 'cell=%s' % (
            cellsize), 'robust=0.5', 'stokes=i,q,u,v', selstring,
            'options=mfs,double', 'line=chan,10,'+str(i)],
            stdin=None, stdout=None, stderr=None, shell=False)

        for stokes in ['i', 'q', 'u', 'v']:
            if not os.path.exists('%s.d.%s.%04d.%s' % (sourcename, freqband, i, stokes)):
                continue
            else:
                call(['clean', 'map=%s.d.%s.%04d.%s' % (sourcename, freqband, i, stokes),
                      'beam=%s.beam.%s.%04d' % (sourcename, freqband, i),
                      'out=%s.model.%s.%04d.%s' % (
                          sourcename, freqband, i, stokes),
                      'cutoff=%f' % (5.*imnoise), 'niters=1500', 'region=mask(%s)' % (maskname)],
                     stdin=None, stdout=None, stderr=None, shell=False)
                call(['restor', 'map=%s.d.%s.%04d.%s' % (sourcename, freqband, i, stokes),
                      'beam=%s.beam.%s.%04d' % (sourcename, freqband, i),
                      'model=%s.model.%s.%04d.%s' % (
                          sourcename, freqband, i, stokes),
                      'out=%s.%s.%04d.%s' % (sourcename, freqband, i, stokes)],
                     stdin=None, stdout=None, stderr=None, shell=False)
                # call(['rm','-rf','%s.%s.%04d.%s.fits'%(sourcename,freqband,i,stokes)])
                call(['fits', 'in=%s.%s.%04d.%s' % (sourcename, freqband, i, stokes),
                      'out=%s.%s.%04d.%s.fits' % (sourcename, freqband, i, stokes), 'op=xyout'],
                     stdin=None, stdout=None, stderr=None, shell=False)
                call(['rm', '-rf', '%s.d.%s.%04d.%s' %
                      (sourcename, freqband, i, stokes)])
                call(['rm', '-rf', '%s.d.%s.%04d.%s.fits' %
                      (sourcename, freqband, i, stokes)])
                call(['rm', '-rf', '%s.model.%s.%04d.%s' %
                      (sourcename, freqband, i, stokes)])
                call(['rm', '-rf', '%s.%s.%04d.%s' %
                      (sourcename, freqband, i, stokes)])

        call(['rm', '-rf', '%s.beam.%s.%04d' % (sourcename, freqband, i)])
    call(['rm', '-rf', '%s' % (maskname)])

