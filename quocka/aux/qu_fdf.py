#!/usr/bin/env python3

import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

rootSname = sys.argv[1]

# Get QU fit components from Craig's MCMC output files.

bfModTypeDict = pickle.load(open(rootSname + "/chains/" + "bfModTypeDict.p", "rb"))

archnewDesigStr = "arch"

BFmodType = bfModTypeDict["./" + rootSname][archnewDesigStr]["BFmodType"].replace(
    "sha", "Sha"
)

para_file = np.genfromtxt(
    rootSname + "/chains/" + rootSname + "_arch_" + BFmodType + "__BFparams.txt",
    dtype=str,
    skip_header=3,
)

fit_comps = np.zeros([3, 6])

mod_comps = BFmodType.split("S")[1:]

num_comps = len(mod_comps)

line = 0
for i in range(0, num_comps):
    if mod_comps[i] == "ha":
        fit_comps[i][0] = 1
        fit_comps[i][1] = float(para_file[line, 1])
        fit_comps[i][2] = float(para_file[line + 1, 1])
        fit_comps[i][3] = float(para_file[line + 2, 1])
        line = line + 3
    elif mod_comps[i] == "han":
        fit_comps[i][0] = 2
        fit_comps[i][1] = float(para_file[line, 1])
        fit_comps[i][2] = float(para_file[line + 1, 1])
        fit_comps[i][3] = float(para_file[line + 2, 1])
        fit_comps[i][4] = float(para_file[line + 3, 1])
        line = line + 4
    elif mod_comps[i] == "hae":
        fit_comps[i][0] = 3
        fit_comps[i][1] = float(para_file[line, 1])
        fit_comps[i][2] = float(para_file[line + 1, 1])
        fit_comps[i][3] = float(para_file[line + 2, 1])
        fit_comps[i][5] = float(para_file[line + 3, 1])
        line = line + 4
    elif mod_comps[i] == "hane":
        fit_comps[i][0] = 4
        fit_comps[i][1] = float(para_file[line, 1])
        fit_comps[i][2] = float(para_file[line + 1, 1])
        fit_comps[i][3] = float(para_file[line + 2, 1])
        fit_comps[i][4] = float(para_file[line + 3, 1])
        fit_comps[i][5] = float(para_file[line + 4, 1])
        line = line + 5
    else:
        print("What model is this?\n")

np.savetxt(rootSname + "_qufitcomps.csv", fit_comps, fmt="%f", delimiter=",")

# Get the Stokes I value at weighted mean lambda squared, so we can convert fractional FDF to absolute values.

a = np.genfromtxt(rootSname + "/specPolData/" + rootSname + ".txt")
c = 299792458.0

goodIIndsFreq = np.where(a[:, 0] < 123456.7)[0]
goodIIndsStokesI = np.where(a[:, 1] > 0)[0]
goodIInds = np.array(np.intersect1d(goodIIndsFreq, goodIIndsStokesI))
goodQUinds = np.where(
    (np.abs(a[goodIInds, 4]) > 0.3 * np.nanmedian(np.abs(a[goodIInds, 4])))
    & (np.abs(a[goodIInds, 6]) > 0.3 * np.nanmedian(np.abs(a[goodIInds, 6])))
)[0]
goodInds = np.array(np.intersect1d(goodIInds, goodQUinds))

freqarch = a[goodInds, 0]
Iarch = a[goodInds, 1]
Ierrarch = a[goodInds, 2]
lSQarch = (c / (freqarch * 1e9)) ** 2
lSQdensarch = np.linspace(0, 1.5 * np.max(lSQarch), 10000)

# Logify I & freq data
logfarch = np.log10(freqarch)
logIarch = np.log10(Iarch)
logIerrsarch = (Ierrarch / Iarch) * (1 / np.log(10))

# Fit
stokesIFitDeg = 9  # float(asd[sname]['stIFitDeg'])
Imodelarch = np.polyfit(logfarch, logIarch, stokesIFitDeg)
logfdensvecarch = np.linspace(np.min(logfarch), np.max(logfarch), 1000)
logImoddensarch = np.polyval(Imodelarch, logfdensvecarch)
Imodarch = 10 ** np.polyval(Imodelarch, logfarch)

# Fit to get band-averaged alpha
stokesIFitDeg = 1
ImodelAlphaarch = np.polyfit(logfarch, logIarch, stokesIFitDeg)

lSQ0 = np.sum(lSQarch**2) / np.sum(lSQarch)
freq0 = c / np.sqrt(lSQ0)

ISQ0 = 10 ** np.polyval(Imodelarch, np.log10(freq0 / 1e9))

# The QU fit compnent models.


def Rotation_FDF(p, psi, RM, sigRM, delRM):
    phiVec = np.linspace(-2000, 2000, 4001)  # Define phi
    polInit = np.zeros_like(phiVec)  # create a vector of zeros
    polInit[np.where(phiVec == round(RM))] = p
    compPolVec = polInit
    return compPolVec


def External_FDF(p, psi, RM, sigRM, delRM):
    phiVec = np.linspace(-2000, 2000, 4001)  # Define phi
    polInit = np.zeros_like(phiVec)  # create a vector of zeros
    # Where the F-thin comp is, make it equal to the fractional polarisation
    polInit[np.where(phiVec == round(RM))] = p
    # Kernel which smoothes the polarised emission due to foreground (Burn) depol
    gaussianKernel = (sigRM * np.sqrt(2 * np.pi)) ** -1 * np.exp(
        -0.5 * (phiVec / sigRM) ** 2
    )

    # Distribute the polarised emission over Faraday depth due to external foreground depol
    firstConv = np.convolve(polInit, gaussianKernel, mode="same")
    #     secondConvolve = np.convolve(firstConv,topHatKernel,mode='same') #distribute the polarised emission due to slab depol
    compPolVec = firstConv
    return compPolVec


def Internal_FDF(p, psi, RM, sigRM, delRM):
    phiVec = np.linspace(-2000, 2000, 4001)  # Define phi
    polInit = np.zeros_like(phiVec)  # create a vector of zeros
    # Where the F-thin comp is, make it equal to the fractional polarisation
    polInit[np.where(phiVec == round(RM))] = p
    #     gaussianKernel = (sigRM*np.sqrt(2*np.pi))**-1 * np.exp(-0.5*(phiVec/sigRM)**2) #Kernel which smoothes the polarised emission due to foreground (Burn) depol
    topInit = np.zeros_like(phiVec)  # create a vector of zeros
    # Kernel which smoothes the pol due to slab / gradient depol
    topInit[np.where(np.abs(phiVec) < delRM / 2)] = 1.0 / delRM
    topHatKernel = topInit

    #     firstConv = np.convolve(polInit,gaussianKernel,mode='same') #Distribute the polarised emission over Faraday depth due to external foreground depol
    # distribute the polarised emission due to slab depol
    secondConvolve = np.convolve(polInit, topHatKernel, mode="same")
    compPolVec = secondConvolve
    return compPolVec


def Mixed_FDF(p, psi, RM, sigRM, delRM):
    phiVec = np.linspace(-2000, 2000, 4001)  # Define phi
    polInit = np.zeros_like(phiVec)  # create a vector of zeros
    # Where the F-thin comp is, make it equal to the fractional polarisation
    polInit[np.where(phiVec == round(RM))] = p
    # Kernel which smoothes the polarised emission due to foreground (Burn) depol
    gaussianKernel = (sigRM * np.sqrt(2 * np.pi)) ** -1 * np.exp(
        -0.5 * (phiVec / sigRM) ** 2
    )
    topInit = np.zeros_like(phiVec)  # create a vector of zeros
    # Kernel which smoothes the pol due to slab / gradient depol
    topInit[np.where(np.abs(phiVec) < delRM / 2)] = 1.0 / delRM
    topHatKernel = topInit

    # Distribute the polarised emission over Faraday depth due to external foreground depol
    firstConv = np.convolve(polInit, gaussianKernel, mode="same")
    # distribute the polarised emission due to slab depol
    secondConvolve = np.convolve(firstConv, topHatKernel, mode="same")
    compPolVec = secondConvolve
    return compPolVec


plt.figure(figsize=(8, 4))
# Plot the FDF and clean components from RM synthesis.
rmsyn_fdfclean = np.genfromtxt("./rmsyn/" + rootSname + ".reformat_FDFclean.dat")
plt.plot(
    rmsyn_fdfclean[:, 0],
    np.sqrt(rmsyn_fdfclean[:, 1] ** 2 + rmsyn_fdfclean[:, 2] ** 2),
    label="RMclean FDF",
    color="tab:gray",
    linestyle="dotted",
    alpha=0.7,
)
rmsyn_comps = np.genfromtxt("./rmsyn/" + rootSname + ".reformat_FDFmodel.dat")
plt.plot(
    rmsyn_comps[:, 0],
    np.sqrt(rmsyn_comps[:, 1] ** 2 + rmsyn_comps[:, 2] ** 2),
    label="RMclean comps",
    linestyle="--",
    alpha=0.7,
)

# Put QUfit comps in the title!
title_str = rootSname + "\n"

# Plot the QU fit components.
phiVec = np.linspace(-2000, 2000, 4001)
for i in range(0, num_comps):
    if fit_comps[i][0] == 1:
        comp_vec = Rotation_FDF(
            fit_comps[i][1],
            fit_comps[i][2],
            fit_comps[i][3],
            fit_comps[i][4],
            fit_comps[i][5],
        )
        comp_style = "Rotation"
    elif fit_comps[i][0] == 2:
        comp_vec = External_FDF(
            fit_comps[i][1],
            fit_comps[i][2],
            fit_comps[i][3],
            fit_comps[i][4],
            fit_comps[i][5],
        )
        comp_style = "External"
    elif fit_comps[i][0] == 3:
        comp_vec = Internal_FDF(
            fit_comps[i][1],
            fit_comps[i][2],
            fit_comps[i][3],
            fit_comps[i][4],
            fit_comps[i][5],
        )
        comp_style = "Internal"
    elif fit_comps[i][0] == 4:
        comp_vec = Mixed_FDF(
            fit_comps[i][1],
            fit_comps[i][2],
            fit_comps[i][3],
            fit_comps[i][4],
            fit_comps[i][5],
        )
        comp_style = "Mixed"
    else:
        continue

    plt.plot(
        phiVec, comp_vec * ISQ0, color="tab:orange", label="QUfit comps", alpha=0.7
    )
    title_str = (
        title_str
        + "QUfit_comp "
        + str(i + 1)
        + ": "
        + comp_style
        + ", frac_pol="
        + "{:.2%}".format(fit_comps[i][1])
        + ", PSI="
        + str(fit_comps[i][2])
        + ", RM="
        + str(fit_comps[i][3])
        + ", sigRM="
        + str(fit_comps[i][4])
        + ", delRM="
        + str(fit_comps[i][5])
        + "\n"
    )

plt.xlim([-600, 600])
plt.title(title_str)
plt.legend()
plt.xlabel("RM (rad m^-2)")
plt.ylabel("Flux density at 2 GHz (Jy/beam)")
plt.savefig(rootSname + "_FDF.png", dpi=300, bbox_inches="tight")
plt.close()
