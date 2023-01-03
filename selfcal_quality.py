#!/usr/bin/env python

import glob
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

mpl.use('Agg')

# Get the noise of an image


def get_noise(img_name):
    hdu = fits.open(img_name)
    data = hdu[0].data[0, 0]
    rms_initial = np.std(data)
    rms = np.std(
        data[np.logical_and(data > -2.5*rms_initial, data < 2.5*rms_initial)])
    peak_max = np.amax(data)
    peak_min = np.amin(data)
    hdu.close()
    return rms, peak_max, peak_min


sourcename = sys.argv[1]
vislist = [sourcename+'.2100', sourcename+'.5500', sourcename+'.7500']

for t in vislist:
    t_p0 = t + '.p0.fits'
    t_p1 = t + '.p1.fits'
    t_p2 = t + '.p2.fits'
    t_p2a1 = t + '.p2a1.fits'
    sigma_p0, peak_max_p0, peak_min_p0 = get_noise(t_p0)
    sigma_p1, peak_max_p1, peak_min_p1 = get_noise(t_p1)
    sigma_p2, peak_max_p2, peak_min_p2 = get_noise(t_p2)
    sigma_p2a1, peak_max_p2a1, peak_min_p2a1 = get_noise(t_p2a1)

    mask_p0 = np.amax([10*sigma_p0, -peak_min_p0*1.5])
    mask_p1 = np.amax([10*sigma_p1, -peak_min_p1*1.5])
    mask_p2 = np.amax([10*sigma_p2, -peak_min_p2*1.5])

    fig = plt.figure(figsize=(8, 6))

    filename = t_p0
    hdu = fits.open(filename)
    data = hdu[0].data[0, 0]
    centre_pix_y = int(data.shape[0]/2)
    centre_pix_x = int(data.shape[1]/2)
    wcs = WCS(hdu[0].header).dropaxis(3).dropaxis(2)
    cutout = Cutout2D(hdu[0].data[0, 0], position=(
        centre_pix_x, centre_pix_y), size=(400, 400), wcs=wcs)

    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(cutout.data, vmin=-10.0*sigma_p2, vmax=30.0 *
               sigma_p2, origin='lower', cmap='cubehelix')
    plt.contour(cutout.data, levels=[mask_p0])
    plt.xticks([])
    plt.yticks([])
    ax.set_title('mask level %s sigma \n rms %s' %
                 (int(mask_p0/sigma_p0), sigma_p0))
    hdu.close()

    filename = t_p1
    hdu = fits.open(filename)
    data = hdu[0].data[0, 0]
    centre_pix_y = int(data.shape[0]/2)
    centre_pix_x = int(data.shape[1]/2)
    wcs = WCS(hdu[0].header).dropaxis(3).dropaxis(2)
    cutout = Cutout2D(hdu[0].data[0, 0], position=(
        centre_pix_x, centre_pix_y), size=(400, 400), wcs=wcs)

    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(cutout.data, vmin=-10.0*sigma_p2, vmax=30.0 *
               sigma_p2, origin='lower', cmap='cubehelix')
    plt.contour(cutout.data, levels=[mask_p1])
    plt.xticks([])
    plt.yticks([])
    ax.set_title('mask level %s sigma \n rms %s' %
                 (int(mask_p1/sigma_p1), sigma_p1))
    hdu.close()

    filename = t_p2
    hdu = fits.open(filename)
    data = hdu[0].data[0, 0]
    centre_pix_y = int(data.shape[0]/2)
    centre_pix_x = int(data.shape[1]/2)
    wcs = WCS(hdu[0].header).dropaxis(3).dropaxis(2)
    cutout = Cutout2D(hdu[0].data[0, 0], position=(
        centre_pix_x, centre_pix_y), size=(400, 400), wcs=wcs)

    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(cutout.data, vmin=-10.0*sigma_p2, vmax=30.0 *
               sigma_p2, origin='lower', cmap='cubehelix')
    plt.contour(cutout.data, levels=[mask_p2])
    plt.xticks([])
    plt.yticks([])
    ax.set_title('mask level %s sigma \n rms %s' %
                 (int(mask_p2/sigma_p2), sigma_p2))
    hdu.close()

    filename = t_p2a1
    hdu = fits.open(filename)
    data = hdu[0].data[0, 0]
    centre_pix_y = int(data.shape[0]/2)
    centre_pix_x = int(data.shape[1]/2)
    wcs = WCS(hdu[0].header).dropaxis(3).dropaxis(2)
    cutout = Cutout2D(hdu[0].data[0, 0], position=(
        centre_pix_x, centre_pix_y), size=(400, 400), wcs=wcs)

    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(cutout.data, vmin=-10.0*sigma_p2, vmax=30.0 *
               sigma_p2, origin='lower', cmap='cubehelix')
    plt.xticks([])
    plt.yticks([])
    ax.set_title('Final rms %s' % sigma_p2a1)
    hdu.close()

    if sigma_p0 > sigma_p1 > sigma_p2:
        plt.suptitle('%s: peak flux %s, min flux %s' % (
            t, peak_max_p0, peak_min_p0), color='tab:green', fontsize=15)
        plt.savefig('%s_0_scal_qua.png' % t, dpi=300, bbox_inches='tight')
    else:
        plt.suptitle('%s: peak flux %s, min flux %s' %
                     (t, peak_max_p0, peak_min_p0), color='tab:red', fontsize=15)
        plt.savefig('%s_1_scal_qua.png' % t, dpi=300, bbox_inches='tight')

    plt.close()
