#!/usr/bin/env python

# MAKECUBE_NEW.PY
# 29 Jan 2019, GHH and PAP
# Smooths IQU frames to common resolution and makes a cube
# Based on MIRIAD CONVOL for the smoothing

from astropy.io import fits
import numpy as np
import glob
from subprocess import call
from os.path import isfile
from tqdm import tqdm

def getinfo(flist):
	beams = []
	freqs = []
	for f in flist:
		h = fits.open(f)
		hdr = h[0].header
		beams.append(hdr['BMAJ'])
		freqs.append(hdr['CRVAL3'])
	return np.array(beams), np.array(freqs)

def do_smooth(flist, blist, new_beam):
	olist = []
	ind = []
	for i,f in enumerate(tqdm(flist)):
		bn = f.split('.fits')[0]
		sm = bn+'.sm'
		jf = open('junk.txt','w')
		call(['fits','op=xyin','in=%s'%f,'out=%s'%bn],stdout=jf,stderr=jf)
		call(['convol','map=%s'%bn,'fwhm=%f'%new_beam,'options=final','out=%s'%sm],stdout=jf,stderr=jf)
		call(['fits','op=xyout','in=%s'%sm,'out=%s.fits'%sm],stdout=jf,stderr=jf)
		jf.close()
		call(['rm','-rf',sm,bn,'junk.txt'])
		if isfile('%s.fits'%sm):
			olist.append('%s.fits'%sm)
			ind.append(i)
	return olist, ind

def makecube(flist,outname):
	h = fits.open(flist[0])
	hdr = h[0].header
	data = np.squeeze(h[0].data)
	outcube = np.zeros((len(flist),hdr['NAXIS2'],hdr['NAXIS1']))
	outcube[0,:,:] = data
	for i in range(1,len(flist)):
		hn = fits.open(flist[i])
		data = np.squeeze(hn[0].data)
		outcube[i,:,:] = data
	h[0].data = outcube
	h.writeto(outname)

def main():
	ilist = sorted(glob.glob('*.i.fits'))
	qlist = sorted(glob.glob('*.q.fits'))
	ulist = sorted(glob.glob('*.u.fits'))

	i_beams, i_freq = getinfo(ilist)
	q_beams, q_freq = getinfo(qlist)
	u_beams, u_freq = getinfo(ulist)

	assert((i_freq == q_freq).all())
	assert((i_freq == u_freq).all())

	new_beam = 11.
	i_smo_list, i_ind_list = do_smooth(ilist, i_beams, new_beam)
	q_smo_list, q_ind_list = do_smooth(qlist, q_beams, new_beam)
	u_smo_list, u_ind_list = do_smooth(ulist, u_beams, new_beam)
	
	outfile = open('freq.txt','w')
	for frq in i_freq[i_ind_list]:
		print >>outfile, frq
	outfile.close()

	makecube(i_smo_list,'icube.fits')
	makecube(q_smo_list,'qcube.fits')
	makecube(u_smo_list,'ucube.fits')
	

if __name__ == '__main__':
	main()

