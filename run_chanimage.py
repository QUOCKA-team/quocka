#!/usr/bin/env python

# Script to generate channel images from calibrated miriad data
# First version by Philippa Patterson, 10 December 2018

import sys 
sourcename = sys.argv[1]

import argparse, ConfigParser
import glob, os
from subprocess import call
from numpy import unique

for i in range(1,2049): #2049
	#for stokes in ['q','u']:
		#call(['invert','vis=%s'%(sourcename),'map=%s.map.ch%04d.%s'%(sourcename,i,stokes),'beam=%s.beam.ch%04d.%s'%(sourcename,i,stokes),'imsize=1024','cell=1','robust=0.5','stokes=%s'%(stokes),'options=mfs,double','slop=0.4','line=chan,1,'+str(i)],stdin=None, stdout=None, stderr=None, shell=False)
	call(['invert','vis=%s'%(sourcename),'map=%s.dirtymap.ch%04d.i'%(sourcename,i)+',%s.dirtymap.ch%04d.q'%(sourcename,i)+ ',%s.dirtymap.ch%04d.u'%(sourcename,i),'beam=%s.beam.ch%04d'%(sourcename,i),'imsize=1024','cell=1','robust=0.5','stokes=i,q,u','options=mfs,double','line=chan,1,'+str(i)],stdin=None, stdout=None, stderr=None, shell=False)
		
		#print('%s.map.ch%04d.%s'%(sourcename))
                #raw_input('Paused')
	for stokes in ['q','u']:
	               
		 if not os.path.exists('%s.dirtymap.ch%04d.%s'%(sourcename,i,stokes)):
                	    continue
            	 else:
                 	call(['clean','map=%s.dirtymap.ch%04d.%s'%(sourcename,i,stokes),'beam=%s.beam.ch%04d'%(sourcename,i),'out=%s.model.ch%04d.%s'%(sourcename,i,stokes),'cutoff=8e-3','niters=3000'],stdin=None, stdout=None, stderr=None, shell=False)
		 	#print('%s.dirtymap.ch%04d.%s'%(sourcename,i,stokes)+'%s.beam.ch%04d'%(sourcename,i)+'%s.model.ch%04d.%s'%(sourcename,i,stokes))
		   	call(['restor','map=%s.dirtymap.ch%04d.%s'%(sourcename,i,stokes),'beam=%s.beam.ch%04d'%(sourcename,i),'model=%s.model.ch%04d.%s'%(sourcename,i,stokes),'out=%s.restor.ch%04d.%s'%(sourcename,i,stokes)],stdin=None, stdout=None, stderr=None, shell=False)



