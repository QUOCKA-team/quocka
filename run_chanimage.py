#!/usr/bin/env python

# Script to generate channel images from calibrated miriad data
# First version by Philippa Patterson, 10 December 2018
# Updated by GH and PP 10 December 2018

import sys, pyfits 
import argparse, ConfigParser
import glob, os
from subprocess import call
from numpy import unique,std

sourcename = sys.argv[1]
vislist = sorted(glob.glob(sourcename+'.????'))
print vislist

def getnoise(fitsname):
	h = pyfits.open(fitsname)
	data = h[0].data
	return std(data)

for vis in vislist:
    freqband = vis.split('.')[-1]
    if freqband == '2100':
	selstring = ''
    else:
	selstring = 'select=-ant(6)'

    call(['invert','vis=%s.%s'%(sourcename,freqband),
                'map=%s.d.%s.mfs.i'%(sourcename,freqband),
                'beam=%s.beam.%s.mfs'%(sourcename,freqband),
                'imsize=1024','cell=1','robust=0.5','stokes=i',selstring,
                'options=mfs,double'],
                stdin=None, stdout=None, stderr=None, shell=False)
    call(['fits','op=xyout','in=%s.d.%s.mfs.i'%(sourcename,freqband),'out=%s.d.%s.mfs.i.fits'%(sourcename,freqband)],
		stdin=None, stdout=None, stderr=None, shell=False)
    imnoise = getnoise('%s.d.%s.mfs.i.fits'%(sourcename,freqband))
    call(['clean','map=%s.d.%s.mfs.i'%(sourcename,freqband),
                                'beam=%s.beam.%s.mfs'%(sourcename,freqband),
                                'out=%s.model.%s.mfs'%(sourcename,freqband),
                                'cutoff=%f'%(2.*imnoise),'niters=100000'],
                                stdin=None, stdout=None, stderr=None, shell=False)
    call(['restor','map=%s.d.%s.mfs.i'%(sourcename,freqband),
                                'beam=%s.beam.%s.mfs'%(sourcename,freqband),
                                'out=%s.restor.%s.mfs'%(sourcename,freqband),
                                'model=%s.model.%s.mfs'%(sourcename,freqband)],
				stdin=None, stdout=None, stderr=None, shell=False)
    call(['fits','op=xyout','in=%s.restor.%s.mfs'%(sourcename,freqband),'out=%s.restor.%s.mfs.fits'%(sourcename,freqband)],
                stdin=None, stdout=None, stderr=None, shell=False)
    imnoise = getnoise('%s.restor.%s.mfs.fits'%(sourcename,freqband))
    maskname = '%s.mask.%s.mfs'%(sourcename,freqband)
    call(['maths','exp=<%s.restor.%s.mfs>'%(sourcename,freqband),'mask=<%s.restor.%s.mfs>.gt.%f'%(sourcename,freqband,10.*imnoise),
		'out=%s'%(maskname)],stdin=None, stdout=None, stderr=None, shell=False)
    call(['rm','-rf','%s.d.%s.mfs.i'%(sourcename,freqband)])
    call(['rm','-rf','%s.d.%s.mfs.i.fits'%(sourcename,freqband)])
    call(['rm','-rf','%s.beam.%s.mfs'%(sourcename,freqband)])
    call(['rm','-rf','%s.model.%s.mfs'%(sourcename,freqband)])
    call(['rm','-rf','%s.restor.%s.mfs'%(sourcename,freqband)])
    call(['rm','-rf','%s.restor.%s.mfs.fits'%(sourcename,freqband)])

    #for i in range(1,2049,10):
    for i in range(1,128,10):
	call(['invert','vis=%s.%s'%(sourcename,freqband),
		'map=%s.d.%s.%04d.i'%(sourcename,freqband,i)+',%s.d.%s.%04d.q'%(sourcename,freqband,i)+ ',%s.d.%s.%04d.u'%(sourcename,freqband,i),
		'beam=%s.beam.%s.%04d'%(sourcename,freqband,i),
		'imsize=1024','cell=1','robust=0.5','stokes=i,q,u',selstring,
		'options=mfs,double','line=chan,10,'+str(i)],
		stdin=None, stdout=None, stderr=None, shell=False)

	for stokes in ['i','q','u']:
		 if not os.path.exists('%s.d.%s.%04d.%s'%(sourcename,freqband,i,stokes)):
                	    continue
            	 else:
			call(['fits','op=xyout','in=%s.d.%s.%04d.%s'%(sourcename,freqband,i,stokes),
				'out=%s.d.%s.%04d.%s.fits'%(sourcename,freqband,i,stokes)],
                		stdin=None, stdout=None, stderr=None, shell=False)
    			imnoise = getnoise('%s.d.%s.%04d.%s.fits'%(sourcename,freqband,i,stokes))
                 	call(['clean','map=%s.d.%s.%04d.%s'%(sourcename,freqband,i,stokes),
				'beam=%s.beam.%s.%04d'%(sourcename,freqband,i),
				'out=%s.model.%s.%04d.%s'%(sourcename,freqband,i,stokes),
				'cutoff=%f'%imnoise,'niters=1000','region=mask(%s)'%(maskname)],
				stdin=None, stdout=None, stderr=None, shell=False)
		   	call(['restor','map=%s.d.%s.%04d.%s'%(sourcename,freqband,i,stokes),
				'beam=%s.beam.%s.%04d'%(sourcename,freqband,i),
				'model=%s.model.%s.%04d.%s'%(sourcename,freqband,i,stokes),
				'out=%s.%s.%04d.%s'%(sourcename,freqband,i,stokes)],
				stdin=None, stdout=None, stderr=None, shell=False)
			call(['rm','-rf','%s.%s.%04d.%s.fits'%(sourcename,freqband,i,stokes)])
			call(['fits','in=%s.%s.%04d.%s'%(sourcename,freqband,i,stokes),
				'out=%s.%s.%04d.%s.fits'%(sourcename,freqband,i,stokes),
				'op=xyout'],
				stdin=None, stdout=None, stderr=None, shell=False)
			call(['rm','-rf','%s.d.%s.%04d.%s'%(sourcename,freqband,i,stokes)])
			call(['rm','-rf','%s.d.%s.%04d.%s.fits'%(sourcename,freqband,i,stokes)])
			call(['rm','-rf','%s.model.%s.%04d.%s'%(sourcename,freqband,i,stokes)])
			call(['rm','-rf','%s.%s.%04d.%s'%(sourcename,freqband,i,stokes)])
	call(['rm','-rf','%s.beam.%s.%04d'%(sourcename,freqband,i)])
    call(['rm','-rf',maskname])


