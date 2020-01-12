#!/usr/bin/env python

# Doing selfcal on a quocka field

import glob, os, sys
from subprocess import call
import numpy as np
import shutil
from astropy.io import fits

# change nfbin to 2
NFBIN = 2

# Print a log file
def logprint(s2p,lf):
	print >>lf, s2p
	print s2p

# Get the noise and peak flux of an image
def get_noise(img_name):
	hdu = fits.open(img_name)
	data = hdu[0].data[0,0]
	dimen = data.shape
	mask = np.ones(dimen)
	mask[int(dimen[0]/2)-200:int(dimen[0]/2)+200, int(dimen[1]/2)-200:int(dimen[1]/2)+200] = 0
	mask = mask.astype(bool)
	rms = np.std(data[mask])
	peak = np.amax(data)
	hdu.close()
	return rms,peak

sourcename = sys.argv[1]
vislist = [sourcename+'.2100', sourcename+'.5500', sourcename+'.7500']
logf = open(sourcename+'.scal.log','w')

for t in vislist:
	t_pscal = t + '.pscal'
	t_map = t + '.map'
	t_beam = t + '.beam'
	t_model = t + '.model'
	t_restor = t + '.restor'
	t_p0 = t + '.p0.fits'
	t_dirty = t + '.dirty.fits'
	t_mask = t + '.mask'
	
	logprint("***** Start selfcal: %s *****"%t, logf)
	logprint("Generate the dirty image:", logf)
	# Generate a MFS image without selfcal.
	call(['invert', 'vis=%s'%t_pscal, 'map=%s'%t_map, 'beam=%s'%t_beam, 'robust=0.5', 'stokes=i', 'options=mfs,double,sdb', 'imsize=3,3,beam', 'cell=5,5,res'],stdout=logf,stderr=logf)
	call(['fits', 'op=xyout', 'in=%s'%t_map, 'out=%s'%t_dirty],stdout=logf,stderr=logf)
	sigma, peak_flux = get_noise(t_dirty)
	sigma10 = 10.0*sigma
	sigma5 = 5.0*sigma
	
	logprint("RMS of dirty image: %s"%sigma, logf)
	logprint("Peak flux density of dirty image: %s"%peak_flux, logf)
	logprint("Generate a shallow cleaned image:", logf)
	call(['mfclean', 'map=%s'%t_map, 'beam=%s'%t_beam, 'out=%s'%t_model, 'niters=10000', 'cutoff=%s,%s'%(sigma10, sigma5), "region='perc(90)'"],stdout=logf,stderr=logf)
	call(['restor', 'map=%s'%t_map, 'beam=%s'%t_beam, 'model=%s'%t_model, 'out=%s'%t_restor],stdout=logf,stderr=logf)
	call(['fits', 'op=xyout', 'in=%s'%t_restor, 'out=%s'%t_p0],stdout=logf,stderr=logf)
	sigma, peak_flux = get_noise(t_p0)
	sigma50 = 50.0*sigma
	sigma20 = 20.0*sigma
	sigma5 = 5.0*sigma
	cutoff_level = np.amax([sigma50, 0.4*peak_flux])
	logprint("RMS of p0 image: %s"%sigma, logf)
	logprint("Peak flux density of p0 image: %s"%peak_flux, logf)
	
	# First round of phase selfcal.
	# Generate a mask
	logprint("***** First round of phase selfcal *****", logf)
	call(['maths', 'exp=<%s>'%t_restor, 'mask=<%s>.gt.%s'%(t_restor,cutoff_level), 'out=%s'%t_mask],stdout=logf,stderr=logf)
	shutil.rmtree(t_restor)
	shutil.rmtree(t_model)
	call(['mfclean', 'map=%s'%t_map, 'beam=%s'%t_beam, 'out=%s'%t_model, 'niters=10000', 'cutoff=%s,%s'%(sigma10, sigma5), 'region=mask(%s)'%t_mask],stdout=logf,stderr=logf)
	call(['selfcal', 'vis=%s'%t_pscal, 'model=%s'%t_model, 'interval=5', 'nfbin=2', 'options=phase,mfs'],stdout=logf,stderr=logf)
	shutil.rmtree(t_map)
	shutil.rmtree(t_beam)
	shutil.rmtree(t_mask)
	shutil.rmtree(t_model)
	# os.remove(t_dirty)

	t_p1 = t + '.p1.fits'
	call(['invert', 'vis=%s'%t_pscal, 'map=%s'%t_map, 'beam=%s'%t_beam, 'robust=0.5', 'stokes=i', 'options=mfs,double,sdb', 'imsize=3,3,beam', 'cell=5,5,res'],stdout=logf,stderr=logf)
	call(['mfclean', 'map=%s'%t_map, 'beam=%s'%t_beam, 'out=%s'%t_model, 'niters=10000', 'cutoff=%s,%s'%(sigma20, sigma5), "region='perc(90)'"],stdout=logf,stderr=logf)
	call(['restor', 'map=%s'%t_map, 'beam=%s'%t_beam, 'model=%s'%t_model, 'out=%s'%t_restor],stdout=logf,stderr=logf)
	call(['fits', 'op=xyout', 'in=%s'%t_restor, 'out=%s'%t_p1],stdout=logf,stderr=logf)
	sigma, peak_flux = get_noise(t_p1)
	sigma5 = 5.0*sigma
	sigma20 = 20.0*sigma
	sigma50 = 50.0*sigma
	logprint("RMS of p1 image: %s"%sigma, logf)
	logprint("Peak flux density of p1 image: %s"%peak_flux, logf)

	# Second round.
	logprint("***** Second round of phase selfcal *****", logf)
	call(['maths', 'exp=<%s>'%t_restor, 'mask=<%s>.gt.%s'%(t_restor,sigma50), 'out=%s'%t_mask],stdout=logf,stderr=logf)
	shutil.rmtree(t_restor)
	shutil.rmtree(t_model)
	call(['mfclean', 'map=%s'%t_map, 'beam=%s'%t_beam, 'out=%s'%t_model, 'niters=10000', 'cutoff=%s,%s'%(sigma20, sigma5), 'region=mask(%s)'%t_mask],stdout=logf,stderr=logf)
	
	call(['selfcal', 'vis=%s'%t_pscal, 'model=%s'%t_model, 'interval=0.5', 'nfbin=2', 'options=phase,mfs'],stdout=logf,stderr=logf)
	shutil.rmtree(t_map)
	shutil.rmtree(t_beam)
	shutil.rmtree(t_mask)
	shutil.rmtree(t_model)

	t_p2 = t + '.p2.fits'	
	call(['invert', 'vis=%s'%t_pscal, 'map=%s'%t_map, 'beam=%s'%t_beam, 'robust=0.5', 'stokes=i', 'options=mfs,double,sdb', 'imsize=3,3,beam', 'cell=5,5,res'],stdout=logf,stderr=logf)
	call(['mfclean', 'map=%s'%t_map, 'beam=%s'%t_beam, 'out=%s'%t_model, 'niters=10000', 'cutoff=%s,%s'%(sigma20, sigma5), "region='perc(90)'"],stdout=logf,stderr=logf)
	call(['restor', 'map=%s'%t_map, 'beam=%s'%t_beam, 'model=%s'%t_model, 'out=%s'%t_restor],stdout=logf,stderr=logf)
	call(['fits', 'op=xyout', 'in=%s'%t_restor, 'out=%s'%t_p2],stdout=logf,stderr=logf)
	sigma, peak_flux = get_noise(t_p2)
	sigma5 = 5.0*sigma
	sigma20 = 20.0*sigma
	sigma50 = 50.0*sigma
	logprint("RMS of p2 image: %s"%sigma, logf)
	logprint("Peak flux density of p2 image: %s"%peak_flux, logf)

	# move on to amp selfcal.
	logprint("***** One round of amp+phase selfcal *****", logf)
	call(['maths', 'exp=<%s>'%t_restor, 'mask=<%s>.gt.%s'%(t_restor,sigma50), 'out=%s'%t_mask],stdout=logf,stderr=logf)
	shutil.rmtree(t_restor)
	shutil.rmtree(t_model)
	call(['mfclean', 'map=%s'%t_map, 'beam=%s'%t_beam, 'out=%s'%t_model, 'niters=10000', 'cutoff=%s,%s'%(sigma20, sigma5), 'region=mask(%s)'%t_mask],stdout=logf,stderr=logf)
	t_ascal = t + '.ascal'
	call(['uvaver', 'vis=%s'%t_pscal, 'out=%s'%t_ascal],stdout=logf,stderr=logf)
	
	# do the first round of amp selfcal with model generated using phase selfcal.
	call(['selfcal', 'vis=%s'%t_ascal, 'model=%s'%t_model, 'interval=5', 'nfbin=2', 'options=amp,mfs'],stdout=logf,stderr=logf)
	shutil.rmtree(t_map)
	shutil.rmtree(t_beam)
	shutil.rmtree(t_mask)
	shutil.rmtree(t_model)

	t_p2a1 = t + '.p2a1.fits'	
	call(['invert', 'vis=%s'%t_ascal, 'map=%s'%t_map, 'beam=%s'%t_beam, 'robust=0.5', 'stokes=i', 'options=mfs,double,sdb', 'imsize=3,3,beam', 'cell=5,5,res'],stdout=logf,stderr=logf)
	call(['mfclean', 'map=%s'%t_map, 'beam=%s'%t_beam, 'out=%s'%t_model, 'niters=10000', 'cutoff=%s,%s'%(sigma20, sigma5), "region='perc(90)'"],stdout=logf,stderr=logf)
	call(['restor', 'map=%s'%t_map, 'beam=%s'%t_beam, 'model=%s'%t_model, 'out=%s'%t_restor],stdout=logf,stderr=logf)
	call(['fits', 'op=xyout', 'in=%s'%t_restor, 'out=%s'%t_p2a1],stdout=logf,stderr=logf)
	sigma, peak_flux = get_noise(t_p2a1)
	logprint("RMS of p2a1 image: %s"%sigma, logf)
	logprint("Peak flux density of p2a1 image: %s"%peak_flux, logf)
	shutil.rmtree(t_map)
	shutil.rmtree(t_beam)
	shutil.rmtree(t_restor)
	shutil.rmtree(t_model)	

logf.close()
