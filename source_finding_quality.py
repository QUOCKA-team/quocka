#!/usr/bin/env python
# -*- coding: utf-8 -*-

import aplpy
import sys
import matplotlib
matplotlib.use('Agg')

sname = sys.argv[1]

comp_file = sname + '_comp.reg'
isle_file = sname + '_isle.reg'

# Reformat the region files generated by Aegean, so they can be used in Aplpy.

f = open(comp_file, 'r')
contents = f.readlines()
contents_reformat = [line.replace('d', '') for line in contents]
contents_reformat.insert(2, "global color=yellow\n")
f.close()

f = open(comp_file+'.reformat.reg', "w")
contents = "".join(contents_reformat)
f.write(contents)
f.close()

f = open(isle_file, 'r')
contents = f.readlines()
contents.insert(2, "global color=white\n")
f.close()

f = open(isle_file+'.reformat.reg', "w")
contents = "".join(contents)
f.write(contents)
f.close()

fig = aplpy.FITSFigure(sname)
fig.show_colorscale(cmap='cubehelix')  # ,vmin=0.002, vmax=0.01)
fig.show_regions(sname+'_isle.reg.reformat.reg')
fig.show_regions(sname+'_comp.reg.reformat.reg')
fig.add_beam()
fig.beam.set(edgecolor='y')
fig.save(sname+'.sf.png')
