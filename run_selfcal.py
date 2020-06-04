#!/usr/bin/env python

# Doing selfcal on a quocka field

import glob
import os
import sys
from subprocess import call
import numpy as np
import shutil
from astropy.io import fits

# change nfbin to 2
NFBIN = 2

# Print a log file


def logprint(s2p, lf):
    print(s2p, file=lf)
    print(s2p)

# Get the rms and peak flux of an image. RMS is estimated using a clipped version of the image data.


def get_noise(img_name):
    hdu = fits.open(img_name)
    data = hdu[0].data[0, 0]
# 	dimen = data.shape
# 	mask = np.ones(dimen)
# 	mask[int(dimen[0]/2)-200:int(dimen[0]/2)+200, int(dimen[1]/2)-200:int(dimen[1]/2)+200] = 0
# 	mask = mask.astype(bool)
# 	rms = np.std(data[mask])
    rms_initial = np.std(data)
    rms = np.std(
        data[np.logical_and(data > -2.5*rms_initial, data < 2.5*rms_initial)])
    peak_max = np.amax(data)
    peak_min = np.amin(data)
    hdu.close()
    return rms, peak_max, peak_min


sourcename = sys.argv[1]
vislist = [sourcename+'.2100', sourcename+'.5500', sourcename+'.7500']

logf = open(sourcename+'.scal.log', 'w', 1)

for t in vislist:
    t_pscal = t + '.pscal'
    t_map = t + '.map'
    t_beam = t + '.beam'
    t_model = t + '.model'
    t_restor = t + '.restor'
    t_p0 = t + '.p0.fits'
    t_dirty = t + '.dirty.fits'
    t_mask = t + '.mask'

    logprint("***** Start selfcal: %s *****" % t, logf)
    logprint("Generate the dirty image:", logf)
    # Generate a MFS image without selfcal.
    call(['invert', 'vis=%s' % t_pscal, 'map=%s' % t_map, 'beam=%s' % t_beam, 'robust=0.5', 'stokes=i',
          'options=mfs,double,sdb', 'imsize=2,2,beam', 'cell=5,5,res'], stdout=logf, stderr=logf)
    call(['fits', 'op=xyout', 'in=%s' % t_map, 'out=%s' %
          t_dirty], stdout=logf, stderr=logf)
    sigma, peak_max, peak_min = get_noise(t_dirty)
    # sigma5 = 0.0005
    # sigma2 = 0.0002

    clean_level = 5.0*sigma

    logprint("RMS of dirty image: %s" % sigma, logf)
    # logprint("Peak flux density of dirty image: %s"%peak_max, logf)
    logprint("Generate a cleaned image:", logf)
    call(['mfclean', 'map=%s' % t_map, 'beam=%s' % t_beam, 'out=%s' % t_model, 'niters=10000',
          'cutoff=%s,%s' % (clean_level, 2*sigma), "region='perc(90)'"], stdout=logf, stderr=logf)
    call(['restor', 'map=%s' % t_map, 'beam=%s' % t_beam, 'model=%s' %
          t_model, 'out=%s' % t_restor], stdout=logf, stderr=logf)
    call(['fits', 'op=xyout', 'in=%s' % t_restor, 'out=%s' %
          t_p0], stdout=logf, stderr=logf)
    sigma, peak_max, peak_min = get_noise(t_p0)
    # dynamic_range = peak_flux/sigma
    # logprint("RMS of p0 image: %s"%sigma, logf)
    # logprint("Peak flux density of p0 image: %s"%peak_flux, logf)
    # if dynamic_range >= 100:
    # 	logprint("Wow this is a bright source!", logf)
    # 	bool_bright = 1
    # elif 100 > dynamic_range >= 20:
    # 	logprint("This is a moderate source.", logf)
    # 	bool_bright = 0
    # else:
    # 	logprint("This is a faint source. Selfcal might not be a good idea...", logf)
    # 	exit(1)

    # First round of phase selfcal.
    # Generate a mask
    logprint("***** First round of phase selfcal *****", logf)
    # sigma2 = 2.0*sigma
    # if bool_bright == True:
    # 	# mask_level = np.amax([50.0*sigma, 0.4*peak_flux])
    # 	mask_level = 10.0*sigma
    # 	clean_level = 3.0*sigma
    # else:
    # 	# mask_level = np.amax([20.0*sigma, 0.4*peak_flux])
    # 	mask_level = 10.0*sigma
    # 	clean_level = 3.0*sigma
    mask_level = np.amax([10*sigma, -peak_min*1.5])
    clean_level = 5.0*sigma
    call(['maths', 'exp=<%s>' % t_restor, 'mask=<%s>.gt.%s' %
          (t_restor, mask_level), 'out=%s' % t_mask], stdout=logf, stderr=logf)
    shutil.rmtree(t_restor)
    shutil.rmtree(t_model)
    call(['mfclean', 'map=%s' % t_map, 'beam=%s' % t_beam, 'out=%s' % t_model, 'niters=1500',
          'cutoff=%s,%s' % (clean_level, 2*sigma), 'region=mask(%s)' % t_mask], stdout=logf, stderr=logf)
    call(['selfcal', 'vis=%s' % t_pscal, 'model=%s' % t_model, 'interval=5',
          'nfbin=1', 'options=phase,mfs'], stdout=logf, stderr=logf)
    shutil.rmtree(t_map)
    shutil.rmtree(t_beam)
    shutil.rmtree(t_mask)
    shutil.rmtree(t_model)
    # os.remove(t_dirty)

    t_p1 = t + '.p1.fits'
    call(['invert', 'vis=%s' % t_pscal, 'map=%s' % t_map, 'beam=%s' % t_beam, 'robust=0.5', 'stokes=i',
          'options=mfs,double,sdb', 'imsize=2,2,beam', 'cell=5,5,res'], stdout=logf, stderr=logf)
    call(['mfclean', 'map=%s' % t_map, 'beam=%s' % t_beam, 'out=%s' % t_model, 'niters=10000',
          'cutoff=%s,%s' % (clean_level, 2*sigma), "region='perc(90)'"], stdout=logf, stderr=logf)
    call(['restor', 'map=%s' % t_map, 'beam=%s' % t_beam, 'model=%s' %
          t_model, 'out=%s' % t_restor], stdout=logf, stderr=logf)
    call(['fits', 'op=xyout', 'in=%s' % t_restor, 'out=%s' %
          t_p1], stdout=logf, stderr=logf)
    sigma, peak_max, peak_min = get_noise(t_p1)
    # logprint("RMS of p1 image: %s"%sigma, logf)
    # logprint("Peak flux density of p1 image: %s"%peak_flux, logf)

    # Second round.
    logprint("***** Second round of phase selfcal *****", logf)
    mask_level = np.amax([10*sigma, -peak_min*1.5])
    clean_level = 5.0*sigma

    call(['maths', 'exp=<%s>' % t_restor, 'mask=<%s>.gt.%s' %
          (t_restor, mask_level), 'out=%s' % t_mask], stdout=logf, stderr=logf)
    shutil.rmtree(t_restor)
    shutil.rmtree(t_model)
    call(['mfclean', 'map=%s' % t_map, 'beam=%s' % t_beam, 'out=%s' % t_model, 'niters=1500',
          'cutoff=%s,%s' % (clean_level, 2*sigma), 'region=mask(%s)' % t_mask], stdout=logf, stderr=logf)

    call(['selfcal', 'vis=%s' % t_pscal, 'model=%s' % t_model, 'interval=0.5',
          'nfbin=1', 'options=phase,mfs'], stdout=logf, stderr=logf)
    shutil.rmtree(t_map)
    shutil.rmtree(t_beam)
    shutil.rmtree(t_mask)
    shutil.rmtree(t_model)

    t_p2 = t + '.p2.fits'
    call(['invert', 'vis=%s' % t_pscal, 'map=%s' % t_map, 'beam=%s' % t_beam, 'robust=0.5', 'stokes=i',
          'options=mfs,double,sdb', 'imsize=2,2,beam', 'cell=5,5,res'], stdout=logf, stderr=logf)
    call(['mfclean', 'map=%s' % t_map, 'beam=%s' % t_beam, 'out=%s' % t_model, 'niters=10000',
          'cutoff=%s,%s' % (clean_level, 2*sigma), "region='perc(90)'"], stdout=logf, stderr=logf)
    call(['restor', 'map=%s' % t_map, 'beam=%s' % t_beam, 'model=%s' %
          t_model, 'out=%s' % t_restor], stdout=logf, stderr=logf)
    call(['fits', 'op=xyout', 'in=%s' % t_restor, 'out=%s' %
          t_p2], stdout=logf, stderr=logf)
    sigma, peak_max, peak_min = get_noise(t_p2)
    # logprint("RMS of p2 image: %s"%sigma, logf)
    # logprint("Peak flux density of p2 image: %s"%peak_flux, logf)

    # move on to amp selfcal.
    logprint("***** One round of amp+phase selfcal *****", logf)
    # sigma2 = 2.0*sigma
    # if bool_bright == True:
    # 	mask_level = 10.0*sigma
    # 	clean_level = 3.0*sigma
    # else:
    # 	mask_level = 10.0*sigma
    # 	clean_level = 3.0*sigma

    mask_level = np.amax([10*sigma, -peak_min*1.5])
    clean_level = 5.0*sigma

    call(['maths', 'exp=<%s>' % t_restor, 'mask=<%s>.gt.%s' %
          (t_restor, mask_level), 'out=%s' % t_mask], stdout=logf, stderr=logf)
    shutil.rmtree(t_restor)
    shutil.rmtree(t_model)
    call(['mfclean', 'map=%s' % t_map, 'beam=%s' % t_beam, 'out=%s' % t_model, 'niters=1500',
          'cutoff=%s,%s' % (clean_level, 2*sigma), 'region=mask(%s)' % t_mask], stdout=logf, stderr=logf)
    t_ascal = t + '.ascal'
    call(['uvaver', 'vis=%s' % t_pscal, 'out=%s' %
          t_ascal], stdout=logf, stderr=logf)

    # do the first round of amp selfcal with model generated using phase selfcal.
    call(['selfcal', 'vis=%s' % t_ascal, 'model=%s' % t_model,
          'interval=5', 'nfbin=1', 'options=amp,mfs'], stdout=logf, stderr=logf)
    shutil.rmtree(t_map)
    shutil.rmtree(t_beam)
    shutil.rmtree(t_mask)
    shutil.rmtree(t_model)

    t_p2a1 = t + '.p2a1.fits'
    call(['invert', 'vis=%s' % t_ascal, 'map=%s' % t_map, 'beam=%s' % t_beam, 'robust=0.5', 'stokes=i',
          'options=mfs,double,sdb', 'imsize=2,2,beam', 'cell=5,5,res'], stdout=logf, stderr=logf)
    call(['mfclean', 'map=%s' % t_map, 'beam=%s' % t_beam, 'out=%s' % t_model, 'niters=10000',
          'cutoff=%s,%s' % (clean_level, 2*sigma), "region='perc(90)'"], stdout=logf, stderr=logf)
    call(['restor', 'map=%s' % t_map, 'beam=%s' % t_beam, 'model=%s' %
          t_model, 'out=%s' % t_restor], stdout=logf, stderr=logf)
    call(['fits', 'op=xyout', 'in=%s' % t_restor, 'out=%s' %
          t_p2a1], stdout=logf, stderr=logf)
    # sigma, peak_flux = get_noise(t_p2a1)
    # logprint("RMS of p2a1 image: %s"%sigma, logf)
    # logprint("Peak flux density of p2a1 image: %s"%peak_flux, logf)
    shutil.rmtree(t_map)
    shutil.rmtree(t_beam)
    shutil.rmtree(t_restor)
    shutil.rmtree(t_model)

logf.close()
