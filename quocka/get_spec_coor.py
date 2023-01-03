#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:04:50 2019

@author: zha292

Make Q,U spectrum from channel images
"""

import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy import wcs
from astropy.io import fits
from scipy.optimize import curve_fit

mpl.use("Agg")


def getnoise(img_name):
    hdu = fits.open(img_name)
    data = hdu[0].data[0, 0]
    rms_initial = np.std(data)
    rms = np.std(
        data[np.logical_and(data > -2.5 * rms_initial, data < 2.5 * rms_initial)]
    )
    hdu.close()
    return rms


# flist = np.genfromtxt('quocka_select.csv', dtype=str)
filename = sys.argv[1]
sname = filename[0:-5]
coor = np.genfromtxt(filename, dtype=str)
f = coor[0]
peak_coor21 = [float(coor[1]), float(coor[2])]
peak_coor55 = [float(coor[3]), float(coor[4])]
peak_coor75 = [float(coor[5]), float(coor[6])]

c_light = 299792458.0

# get the peak flux pixel from mfs images
mfs21 = f + "/" + f + ".2100.regrid.cutout.fits"
mfs55 = f + "/" + f + ".5500.regrid.cutout.fits"
mfs75 = f + "/" + f + ".7500.regrid.cutout.fits"

mfs21_img = fits.open(mfs21)
mfs21_d = mfs21_img[0].data[0, 0]
wcs_21 = wcs.WCS(mfs21_img[0].header).dropaxis(3).dropaxis(2)
# mfs21_peak = np.where(mfs21_d==np.amax(mfs21_d[924:1124,924:1124]))
# mfs21_peak = np.where(mfs21_d==np.amax(mfs21_d[1948:2148,1948:2148]))
# peak_coor = wcs_21.wcs_pix2world(mfs21_peak[0],mfs21_peak[1],0)
mfs21_peak = wcs_21.wcs_world2pix(peak_coor21[0], peak_coor21[1], 0)
mfs21_img.close()

mfs55_img = fits.open(mfs55)
mfs55_d = mfs55_img[0].data[0, 0]
wcs_55 = wcs.WCS(mfs55_img[0].header).dropaxis(3).dropaxis(2)
# mfs55_peak = np.where(mfs55_d==np.amax(mfs55_d[881:1167,881:1167]))
# mfs55_peak = np.where(mfs55_d==np.amax(mfs55_d[1905:2191,1905:2191]))
mfs55_peak = wcs_55.wcs_world2pix(peak_coor55[0], peak_coor55[1], 0)
mfs55_img.close()

mfs75_img = fits.open(mfs75)
mfs75_d = mfs75_img[0].data[0, 0]
wcs_75 = wcs.WCS(mfs75_img[0].header).dropaxis(3).dropaxis(2)
# mfs75_peak = np.where(mfs75_d==np.amax(mfs75_d[1848:2248,1848:2248]))
mfs75_peak = wcs_75.wcs_world2pix(peak_coor75[0], peak_coor75[1], 0)
mfs75_img.close()

# Now we find the peak flux along channels...
ilist21 = glob.glob(f + ".convol/" + f + ".2100*.i.cutout.fits.con.fits")
ilist21.sort()

ilist55 = glob.glob(f + ".convol/" + f + ".5500*.i.cutout.fits.con.fits")
ilist55.sort()

ilist75 = glob.glob(f + ".convol/" + f + ".7500*.i.cutout.fits.con.fits")
ilist75.sort()

stokes_file = open(sname + ".txt", "w")

for i_name in ilist21:

    if any(freq in i_name for freq in ["0101", "1941"]):
        continue

    peak_x = int(np.round(mfs21_peak[0]))
    peak_y = int(np.round(mfs21_peak[1]))

    i_img = fits.open(i_name)
    chan = i_img[0].header["CRVAL3"] / 1e9  # XZ: frequency in GHz
    data_i = i_img[0].data[0, 0]
    peak_i = data_i[peak_y, peak_x]
    # box_i = data_i[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_i = getnoise(i_name)
    i_img.close()
    # print(str(peak_x[0])+"  "+str(peak_y[0]))

    q_name = i_name.replace(".i.", ".q.")
    q_img = fits.open(q_name)
    data_q = q_img[0].data[0, 0]
    peak_q = data_q[peak_y, peak_x]
    # box_q = data_q[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_q = getnoise(q_name)
    q_img.close()

    u_name = i_name.replace(".i.", ".u.")
    u_img = fits.open(u_name)
    data_u = u_img[0].data[0, 0]
    peak_u = data_u[peak_y, peak_x]
    # box_u = data_u[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_u = getnoise(u_name)
    u_img.close()

    v_name = i_name.replace(".i.", ".v.")
    v_img = fits.open(v_name)
    data_v = v_img[0].data[0, 0]
    peak_v = data_v[peak_y, peak_x]
    # box_v = data_v[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_v = getnoise(v_name)
    v_img.close()

    stokes_file.write(
        str(chan)
        + " "
        + str(peak_i)
        + " "
        + str(noise_i)
        + " "
        + str(peak_q)
        + " "
        + str(noise_q)
        + " "
        + str(peak_u)
        + " "
        + str(noise_u)
        + " "
        + str(peak_v)
        + " "
        + str(noise_v)
        + "\n"
    )


for i_name in ilist55:

    if any(freq in i_name for freq in ["0101", "1941"]):
        continue

    peak_x = int(np.round(mfs55_peak[0]))
    peak_y = int(np.round(mfs55_peak[1]))

    i_img = fits.open(i_name)
    chan = i_img[0].header["CRVAL3"] / 1e9  # XZ: frequency in GHz
    data_i = i_img[0].data[0, 0]
    peak_i = data_i[peak_y, peak_x]
    # box_i = data_i[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_i = getnoise(i_name)
    i_img.close()

    q_name = i_name.replace(".i.", ".q.")
    q_img = fits.open(q_name)
    data_q = q_img[0].data[0, 0]
    peak_q = data_q[peak_y, peak_x]
    # box_q = data_q[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_q = getnoise(q_name)
    q_img.close()

    u_name = i_name.replace(".i.", ".u.")
    u_img = fits.open(u_name)
    data_u = u_img[0].data[0, 0]
    peak_u = data_u[peak_y, peak_x]
    # box_u = data_u[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_u = getnoise(u_name)
    u_img.close()

    v_name = i_name.replace(".i.", ".v.")
    v_img = fits.open(v_name)
    data_v = v_img[0].data[0, 0]
    peak_v = data_v[peak_y, peak_x]
    # box_v = data_v[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_v = getnoise(v_name)
    v_img.close()

    stokes_file.write(
        str(chan)
        + " "
        + str(peak_i)
        + " "
        + str(noise_i)
        + " "
        + str(peak_q)
        + " "
        + str(noise_q)
        + " "
        + str(peak_u)
        + " "
        + str(noise_u)
        + " "
        + str(peak_v)
        + " "
        + str(noise_v)
        + "\n"
    )

for i_name in ilist75:

    if any(freq in i_name for freq in ["0101", "1941"]):
        continue

    peak_x = int(np.round(mfs75_peak[0]))
    peak_y = int(np.round(mfs75_peak[1]))

    i_img = fits.open(i_name)
    chan = i_img[0].header["CRVAL3"] / 1e9  # XZ: frequency in GHz
    data_i = i_img[0].data[0, 0]
    peak_i = data_i[peak_y, peak_x]
    # box_i = data_i[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_i = getnoise(i_name)
    i_img.close()

    q_name = i_name.replace(".i.", ".q.")
    q_img = fits.open(q_name)
    data_q = q_img[0].data[0, 0]
    peak_q = data_q[peak_y, peak_x]
    # box_q = data_q[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_q = getnoise(q_name)
    q_img.close()

    u_name = i_name.replace(".i.", ".u.")
    u_img = fits.open(u_name)
    data_u = u_img[0].data[0, 0]
    peak_u = data_u[peak_y, peak_x]
    # box_u = data_u[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_u = getnoise(u_name)
    u_img.close()

    v_name = i_name.replace(".i.", ".v.")
    v_img = fits.open(v_name)
    data_v = v_img[0].data[0, 0]
    peak_v = data_v[peak_y, peak_x]
    # box_v = data_v[img_size-box_r:img_size-box_l, img_size-box_t:img_size-box_b]
    noise_v = getnoise(v_name)
    v_img.close()

    stokes_file.write(
        str(chan)
        + " "
        + str(peak_i)
        + " "
        + str(noise_i)
        + " "
        + str(peak_q)
        + " "
        + str(noise_q)
        + " "
        + str(peak_u)
        + " "
        + str(noise_u)
        + " "
        + str(peak_v)
        + " "
        + str(noise_v)
        + "\n"
    )

stokes_file.close()


# Let's make some images!

spec = np.genfromtxt(sname + ".txt", delimiter=" ", dtype=float)
spec_qua = spec[:, 1] > 0.004
spec = spec[spec_qua]
spec = spec[spec[:, 0].argsort()]
spec = spec[1:, :]
I = spec[:, 1]
Q_err = spec[:, 4]
U_err = spec[:, 6]
P_err = np.sqrt(Q_err**2 + U_err**2)
spec_qua = P_err / I / np.std(P_err / I) < 5
# spec = spec[spec_qua]

freq = spec[:, 0]
I = spec[:, 1] * 1000
I_err = spec[:, 2] * 1000
Q = spec[:, 3] * 1000
Q_err = spec[:, 4] * 1000
U = spec[:, 5] * 1000
U_err = spec[:, 6] * 1000
V = spec[:, 7] * 1000
V_err = spec[:, 8] * 1000

c_light = 299792458.0
lambda2 = (c_light / freq / 1e9) ** 2.0

hdu1 = fits.open(f + "/" + f + ".2100.regrid.cutout.fits")
img1 = hdu1[0].data[0, 0]
wcs_21 = wcs.WCS(hdu1[0].header).dropaxis(3).dropaxis(2)
# img1_peak = np.where(img1==np.amax(img1[1948:2148,1948:2148]))
# peak_coor = wcs_21.wcs_pix2world(img1_peak[0],img1_peak[1],0)
# img1 = img1[924:1124,924:1124]*1000
# img1 = img1[1948:2148,1948:2148]*1000
# img1_peak = np.where(img1==np.amax(img1))
img1_peak = wcs_21.wcs_world2pix(peak_coor21[0], peak_coor21[1], 0)
# print(img1_peak)
img1_peak = np.array(img1_peak)
img1_peak = img1_peak.round().astype(int)
hdu1.close()
hdu2 = fits.open(f + "/" + f + ".5500.regrid.cutout.fits")
img2 = hdu2[0].data[0, 0]
wcs_55 = wcs.WCS(hdu2[0].header).dropaxis(3).dropaxis(2)
img2_peak = wcs_55.wcs_world2pix(peak_coor55[0], peak_coor55[1], 0)
# print(img2_peak)
img2_peak = np.array(img2_peak)
img2_peak = img2_peak.round().astype(int)
# img2 = img2[881:1167,881:1167]*1000
# img2 = img2[1905:2191,1905:2191]*1000
# img2_peak = np.where(img2==np.amax(img2))
hdu2.close()
hdu3 = fits.open(f + "/" + f + ".7500.regrid.cutout.fits")
img3 = hdu3[0].data[0, 0]
wcs_75 = wcs.WCS(hdu3[0].header).dropaxis(3).dropaxis(2)
img3_peak = wcs_75.wcs_world2pix(peak_coor75[0], peak_coor75[1], 0)
# print(img3_peak)
img3_peak = np.array(img3_peak)
img3_peak = img3_peak.round().astype(int)
# img3 = img3[824:1224,824:1224]*1000
# img3 = img3[1848:2248,1848:2248]*1000
# img3_peak = np.where(img3==np.amax(img3))
hdu3.close()

# print(img1_peak, img2_peak, img3_peak)

# fit the stokes I spec


def func1(x, a, alpha):
    return a * np.power(x, alpha)


def func2(x, a, alpha, q):
    return a * np.power(x, alpha) * np.exp(q * np.log(x) ** 2)


popt, pcov = curve_fit(func2, freq, I)
fit_order = 2

if pcov[0, 0] > 10:
    popt, pcov = curve_fit(func1, freq, I)
    fit_order = 1

perr = np.sqrt(np.diag(pcov))

np.savetxt(sname + ".popt.txt", popt)
np.savetxt(sname + ".perr.txt", perr)

fig = plt.figure(figsize=[12, 12])
gs = fig.add_gridspec(3, 3)
ax2100 = fig.add_subplot(gs[0, 0])
img2100 = ax2100.imshow(img1, cmap="cubehelix", origin="lower")
plt.scatter(
    img1_peak[0], img1_peak[1], marker="o", facecolors="none", s=135, edgecolor="y"
)
# plt.scatter(100,100,marker='o',facecolors='none',s=240,edgecolor='y',linestyle='--')
fig.colorbar(img2100, ax=ax2100, fraction=0.046, pad=0.04)
ax2100.set_xticks([])
ax2100.set_yticks([])
plt.title("2100")
ax5500 = fig.add_subplot(gs[0, 1])
img5500 = ax5500.imshow(img2, cmap="cubehelix", origin="lower")
plt.scatter(
    img2_peak[0], img2_peak[1], marker="o", facecolors="none", s=135, edgecolor="y"
)
# plt.scatter(143,143,marker='o',facecolors='none',s=240,edgecolor='y',linestyle='--')
fig.colorbar(img5500, ax=ax5500, fraction=0.046, pad=0.04)
ax5500.set_xticks([])
ax5500.set_yticks([])
plt.title("5500")
ax7500 = fig.add_subplot(gs[0, 2])
img7500 = ax7500.imshow(img3, cmap="cubehelix", origin="lower")
plt.scatter(
    img3_peak[0], img3_peak[1], marker="o", facecolors="none", s=135, edgecolor="y"
)
# plt.scatter(200,200,marker='o',facecolors='none',s=240,edgecolor='y',linestyle='--')
fig.colorbar(img7500, ax=ax7500, fraction=0.046, pad=0.04)
ax7500.set_xticks([])
ax7500.set_yticks([])
plt.title("7500")
axstokesI = fig.add_subplot(gs[1, :])
axstokesI.errorbar(freq, I, yerr=I_err, fmt=".", label="Stokes I")
if fit_order == 2:
    axstokesI.plot(freq, func2(freq, *popt), "-", label="Stokes I model")
else:
    axstokesI.plot(freq, func1(freq, *popt), "-", label="Stokes I model")

# axstokesI.title("$I_model$ = %.2f$\nu^%.2f$")
plt.legend()
plt.xlim([1, 10])
# plt.ylim([1,1000])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Freq (GHz)")
plt.ylabel("Stokes I (mJy)")
axstokesQU = fig.add_subplot(gs[2, :])
axstokesQU.errorbar(
    lambda2, Q / I, yerr=Q_err / I, fmt=".", label="Stokes Q/Stokes I", c="tab:orange"
)
axstokesQU.errorbar(
    lambda2, U / I, yerr=U_err / I, fmt=".", label="Stokes U/Stokes I", c="tab:green"
)
plt.axhline(0, c="grey", linestyle="--")
plt.legend()
# plt.xlim([0.0009,0.09])
# plt.ylim([-0.7,0.7])
# plt.xscale('log')
plt.xlabel("$\lambda^2 (m^2)$")
plt.ylabel("Fractional polarisation")
plt.suptitle(sname, y=0.92, fontsize=15)
plt.savefig(sname + ".png", dpi=300, bbox_inches="tight")
plt.close()
