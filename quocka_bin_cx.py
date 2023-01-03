#!/usr/bin/env python

import sys

import numpy as np

sname = sys.argv[1]

data = np.genfromtxt(sname + ".txt")

freq = data[:, 0]
band21 = freq < 4
band55 = np.logical_and(freq > 4, freq < 6.5)
band75 = freq > 6.5

bins55 = np.linspace(freq[band55][0], freq[band55][-1], 26)
bins75 = np.linspace(freq[band75][0], freq[band75][-1], 14)

inds55 = np.digitize(freq[band55], bins55)
data55_bin = [data[band55][inds55 == i].mean(axis=0) for i in range(1, len(bins55))]
data55_bin = np.array(data55_bin)

inds75 = np.digitize(freq[band75], bins75)
data75_bin = [data[band75][inds75 == i].mean(axis=0) for i in range(1, len(bins75))]
data75_bin = np.array(data75_bin)

data_bin = np.concatenate([data[band21], data55_bin, data75_bin])

np.savetxt(sname + "_bin.txt", data_bin, fmt="%s")
