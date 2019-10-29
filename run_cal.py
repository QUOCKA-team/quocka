#!/usr/bin/env python

import argparse, ConfigParser
import glob, os
from subprocess import call
from numpy import unique

def logprint(s2p,lf):
	print >>lf, s2p
	print s2p

# Pgflagging lines, following the ATCA users guide. Pgflagging needs to be done on all the calibrators and targets.
def flag(src, logf):
	call(['pgflag','vis=%s'%src,'stokes=i,q,u,v','flagpar=8,5,5,3,6,3','command=<b','options=nodisp'],stdout=logf,stderr=logf)
	call(['pgflag','vis=%s'%src,'stokes=i,v,u,q','flagpar=8,2,2,3,6,3','command=<b','options=nodisp'],stdout=logf,stderr=logf)
	call(['pgflag','vis=%s'%src,'stokes=i,v,q,u','flagpar=8,2,2,3,6,3','command=<b','options=nodisp'],stdout=logf,stderr=logf)

# change nfbin to 4
NFBIN = 4
	
def main(args,cfg):
	# Initiate log file with options used
	logf = open(args.log_file,'w',1) # line buffered
	logprint('Input settings:',logf)
	logprint(args,logf)
	logprint(cfg.items('input'),logf)
	logprint(cfg.items('output'),logf)
	logprint(cfg.items('observation'),logf)
	logprint('',logf)

	gwcp = cfg.get('input','dir')+'/'+cfg.get('input','date')+'*'
	atfiles = sorted(glob.glob(gwcp))
	if_use = cfg.getint('input','if_use')
	outdir = cfg.get('output','dir')
	rawclobber = cfg.getboolean('output','rawclobber')
	outclobber = cfg.getboolean('output','clobber')
	skipcal = cfg.getboolean('output','skipcal')
	prical = cfg.get('observation','primary')
	seccal = cfg.get('observation','secondary')
	polcal = cfg.get('observation','polcal')
	seccal_ext = cfg.get('observation','sec_ext')
	target_ext = cfg.get('observation','ext')

	if not os.path.exists(outdir):
		logprint('Creating directory %s'%outdir,logf)
		os.makedirs(outdir)
	for line in open(args.setup_file):
		if line[0] == '#':
			continue
		sline = line.split()
		for a in atfiles:
			if sline[0] in a:
				logprint('Ignoring setup file %s'%sline[0],logf)
				atfiles.remove(a)
	uvlist = ','.join(atfiles)

	if not os.path.exists(outdir+'/dat.uv') or rawclobber:
		logprint('Running ATLOD...',logf)
		if if_use > 0:
			call(['atlod', 'in=%s'%uvlist, 'out=%s/dat.uv'%outdir, 'ifsel=%s'%if_use, 'options=birdie,noauto,xycorr,rfiflag'],stdout=logf,stderr=logf)
		else:
			call(['atlod', 'in=%s'%uvlist, 'out=%s/dat.uv'%outdir, 'options=birdie,noauto,xycorr,rfiflag'],stdout=logf,stderr=logf)
	else:
		logprint('Skipping atlod step',logf)
	os.chdir(outdir)
	logprint('Running UVSPLIT...',logf)
	if outclobber:
		logprint('Output files will be clobbered if necessary',logf)
		call(['uvsplit','vis=dat.uv','options=mosaic,clobber'],stdout=logf,stderr=logf)
	else:
		call(['uvsplit','vis=dat.uv','options=mosaic'],stdout=logf,stderr=logf)
	slist = sorted(glob.glob('[j012]*.[257]???'))
	logprint('Working on %d sources'%len(slist),logf)
	bandfreq = unique([x[-4:] for x in slist])
	logprint('Frequency bands to process: %s'%(','.join(bandfreq)),logf)

	src_to_plot = []

	for frqb in bandfreq:
		logprint('\n\n##########\nWorking on frequency: %s\n##########\n\n'%(frqb),logf)
		pricalname = '__NOT_FOUND__'
		ext_seccalname = '__NOT_FOUND__'
		seccalnames = []
		polcalnames = []
		targetnames = []
		ext_targetnames = []
		for i,source in enumerate(slist):
			frqid = source[-4:]
			if frqid not in frqb:
				continue
			if prical in source: 
				pricalname = source
			elif any([sc in source for sc in seccal.split(',')]):
				seccalnames.append(source)
			elif seccal_ext in source:
				ext_seccalname = source
			elif any([pc in source for pc in polcal.split(',')]):
				polcalnames.append(source)
			elif any([es in source for es in target_ext.split(',')]):
				ext_targetnames.append(source)
			else:
				targetnames.append(source)
				src_to_plot.append(source[:-5])
		if pricalname == '__NOT_FOUND__':
			logprint('Error: primary cal (%s) not found'%prical,logf)
			logf.close()
			exit(1)
		if len(seccalnames) == 0:
			logprint('Error: secondary cal (%s) not found'%seccal,logf)
			logf.close()
			exit(1)
		if ext_seccalname == '__NOT_FOUND__' and seccal_ext != 'NONE':
			logprint('Error: extended-source secondary cal (%s) not found'%seccal_ext,logf)
			logf.close()
			exit(1)
		elif seccal_ext == 'NONE':
			ext_seccalname = '(NONE)'
		logprint('Identified primary cal: %s'%pricalname,logf)
		logprint('Identified %d secondary cals'%len(seccalnames),logf)
		logprint('Identified %d polarization calibrators'%len(polcalnames),logf)
		logprint('Identified %d compact targets to calibrate'%len(targetnames),logf)
		logprint('Identified secondary cal for extended sources: %s'%ext_seccalname,logf)
		logprint('Identified %d extended targets to calibrate'%len(ext_targetnames),logf)
		if skipcal:
			logprint('Skipping flagging and calibration steps on user request.',logf)
			continue
		logprint('Initial flagging round proceeding...',logf)
		for i,source in enumerate(slist):
			# only flag data corresponding to the data that we're dealing with (resolves issue #4)
			if frqb not in source:
				continue
			logprint('\nFLAGGING: %d / %d = %s'%(i+1,len(slist),source),logf)
			####
			# This part may be largely obsolete with options=rfiflag in ATLOD.
			# However, options=rfiflag doesn't cover all the badchans, so we'll still do this. -XZ
			for line in open('../badchans_%s.txt'%frqid):
				sline=line.split()
				lc,uc=sline[0].split('-')
				dc = int(uc)-int(lc)+1
				call(['uvflag','vis=%s'%source,'line=chan,%d,%s'%(dc,lc),'flagval=flag'],stdout=logf,stderr=logf)
			####
			#call(['pgflag','vis=%s'%source,'stokes=xx,yy,yx,xy','flagpar=20,10,10,3,5,3,20','command=<be','options=nodisp'])
# 			call(['uvflag','vis=%s'%source,'select=amplitude(2),polarization(xy,yx)','flagval=flag'],stdout=logf,stderr=logf)
# 			call(['uvpflag','vis=%s'%source,'polt=xy,yx','pols=xx,xy,yx,yy','options=or'],stdout=logf,stderr=logf)
# 			call(['pgflag','vis=%s'%source,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
			
			# First round of pgflag for all sources.
			flag(source, logf)
	
		# Flagging/calibrating the primary calibrator 1934-638. 
		logprint('Calibration of primary cal (%s) proceeding ...'%prical,logf)
		# Only select data above elevation=40.
		call(['uvflag','vis=%s'%pricalname, 'select=-elevation(40,90)', 'flagval=flag'],stdout=logf,stderr=logf)
		# First round of calibrating. Apply the solutions.
		call(['mfcal','vis=%s'%pricalname,'interval=0.1,1,30'],stdout=logf,stderr=logf)
		call([ 'gpcal', 'vis=%s'%pricalname, 'interval=0.1', 'nfbin=%d'%NFBIN, 'options=xyvary'],stdout=logf,stderr=logf)
		pricalname_c1 = pricalname + '_c1'
		call(['uvaver', 'vis=%s'%pricalname, 'out=%s'%pricalname_c1],stdout=logf,stderr=logf)
		
		# Second round of flagging/calibrating
		
		flag(pricalname_c1, logf)
		call(['mfcal','vis=%s'%pricalname_c1,'interval=0.1,1,30'],stdout=logf,stderr=logf)
		call([ 'gpcal', 'vis=%s'%pricalname_c1, 'interval=0.1', 'nfbin=%d'%NFBIN, 'options=xyvary'],stdout=logf,stderr=logf)
		pricalname_c2 = pricalname + '_c2'
		call(['uvaver', 'vis=%s'%pricalname_c1, 'out=%s'%pricalname_c2],stdout=logf,stderr=logf)
		
# 		call(['pgflag','vis=%s'%pricalname,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
# 		call(['mfcal','vis=%s'%pricalname,'interval=10000','select=elevation(40,90)'],stdout=logf,stderr=logf)
# 		call([ 'gpcal', 'vis=%s'%pricalname, 'interval=0.1', 'nfbin=16', 'options=xyvary','select=elevation(40,90)'],stdout=logf,stderr=logf)
# 		call(['pgflag','vis=%s'%pricalname,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
# 		call([ 'gpcal', 'vis=%s'%pricalname, 'interval=0.1', 'nfbin=16', 'options=xyvary','select=elevation(40,90)'],stdout=logf,stderr=logf)

		# Move on to the secondary calibrator
		for seccalname in seccalnames:
			logprint('Transferring to compact-source secondary %s...'%seccalname,logf)
			call(['gpcopy','vis=%s'%pricalname_c2,'out=%s'%seccalname, 'mode=merge'],stdout=logf,stderr=logf)
			# flag twice, gpcal twice
			flag(seccalname, logf)
			call(['gpcal','vis=%s'%seccalname,'interval=0.1','nfbin=%d'%NFBIN,'options=xyvary,qusolve'],stdout=logf,stderr=logf)
			flag(seccalname, logf)
			call(['gpcal','vis=%s'%seccalname,'interval=0.1','nfbin=%d'%NFBIN,'options=xyvary,qusolve'],stdout=logf,stderr=logf)
			# boot the flux
			call(['gpboot','vis=%s'%seccalname,'cal=%s'%pricalname_c2],stdout=logf,stderr=logf)
			
# 			call(['puthd','in=%s/interval'%seccalname,'value=100000'],stdout=logf,stderr=logf)
# 			call(['pgflag','vis=%s'%seccalname,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
# 			##call(['gpcal','vis=%s'%seccalname,'interval=0.1','nfbin=16','options=xyvary,qusolve'],stdout=logf,stderr=logf)
# 			call(['gpcal','vis=%s'%seccalname,'interval=0.1','nfbin=16','options=nopol,noxy'],stdout=logf,stderr=logf)
# 			call(['gpedit','vis=%s'%seccalname,'options=phase'],stdout=logf,stderr=logf)
# 			call(['pgflag','vis=%s'%seccalname,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
# 			##call(['gpcal','vis=%s'%seccalname,'interval=0.1','nfbin=16','options=xyvary,qusolve'],stdout=logf,stderr=logf)
# 			call(['gpcal','vis=%s'%seccalname,'interval=0.1','nfbin=16','options=nopol,noxy'],stdout=logf,stderr=logf)
# 			call(['gpedit','vis=%s'%seccalname,'options=phase'],stdout=logf,stderr=logf)
# 			call(['gpboot','vis=%s'%seccalname,'cal=%s'%pricalname],stdout=logf,stderr=logf)
		#if len(seccalnames) == 2:
		#	call(['gpcopy','vis=%s'%seccalnames[0],'out=%s'%seccalnames[1],'mode=merge'],stdout=logf,stderr=logf)
		#	seccalname = seccalnames[1]
		#elif len(seccalnames) == 1:
		#	seccalname = seccalnames[0]
		#else:
		#	logprint('Error: too many secondaries, fix me!!',logf)
		#	exit(1)
		while len(seccalnames) > 1:
			logprint('Merging gain table for %s into %s ...'%(seccalnames[-1],seccalnames[0]),logf)
			call(['gpcopy','vis=%s'%seccalnames[-1],'out=%s'%seccalnames[0],'mode=merge'],stdout=logf,stderr=logf)
			del seccalnames[-1]
		seccalname = seccalnames[0]
		logprint('Using gains from %s ...'%(seccalname),logf)
		# For now, we don't worry about extended sources
# 		if seccal_ext != 'NONE':
# 			logprint('Transferring to extended-source secondary...',logf)
# 			call(['gpcopy','vis=%s'%pricalname,'out=%s'%ext_seccalname],stdout=logf,stderr=logf)
# 			call(['puthd','in=%s/interval'%ext_seccalname,'value=100000'],stdout=logf,stderr=logf)
# 			call(['pgflag','vis=%s'%ext_seccalname,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
# 			call(['gpcal','vis=%s'%ext_seccalname,'interval=0.1','nfbin=16','options=xyvary,qusolve'],stdout=logf,stderr=logf)
# 			call(['pgflag','vis=%s'%ext_seccalname,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
# 			call(['gpcal','vis=%s'%ext_seccalname,'interval=0.1','nfbin=16','options=xyvary,qusolve'],stdout=logf,stderr=logf)
# 			call(['gpboot','vis=%s'%ext_seccalname,'cal=%s'%pricalname],stdout=logf,stderr=logf)
# 			logprint('\n\n##########\nApplying calibration to extended sources...\n##########\n\n',logf)
# 			for t in ext_targetnames:
# 				logprint('Working on source %s'%t,logf)
# 				call(['gpcopy','vis=%s'%ext_seccalname,'out=%s'%t],stdout=logf,stderr=logf)
# 				call(['pgflag','vis=%s'%t,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
		logprint('\n\n##########\nApplying calibration to compact sources...\n##########\n\n',logf)
		for t in targetnames:
			logprint('Working on source %s'%t,logf)
			slogname = '%s.log.txt'%t
			slogf = open(slogname,'w',1)
			
			# Move on to the target!
			call(['gpcopy','vis=%s'%seccalname,'out=%s'%t],stdout=logf,stderr=logf)
			flag(t, logf)
# 			call(['pgflag','vis=%s'%t,'stokes=v','flagpar=7,4,12,3,5,3,20','command=<be','options=nodisp'],stdout=logf,stderr=logf)
			
			logprint('Writing source flag and pol info to %s'%slogname,logf)
			call(['uvfstats','vis=%s'%t],stdout=slogf,stderr=slogf)
			call(['uvfstats','vis=%s'%t,'mode=channel'],stdout=slogf,stderr=slogf)
			slogf.close()
			
			# Apply the solutions before we do selfcal
			t_pscal = t + '_pscal'
			call(['uvaver', 'vis=%s'%t, 'out=%s'%t_pscal],stdout=logf,stderr=logf)
			
			# Phase selfcal. First round.
			call(['invert', 'vis=%s'
			
			
			
			
			
	for t in sorted(unique(src_to_plot)):
		logprint('Plotting RMSF for %s'%t,logf)
		if int(bandfreq[0]) < 3500:
			call(['uvspec','vis=%s.????'%t,'axis=rm','options=nobase,avall','nxy=1,2','interval=100000','xrange=-1500,1500','device=junk.eps/vcps'],stdout=logf,stderr=logf)
		else:
			# For CX, the IFs have to be catenated before RM Synthesis
			call(['uvcat','vis=%s.????'%t,'out=%s.cx'%t],stdout=logf,stderr=logf)
			call(['uvspec','vis=%s.cx'%t,'axis=rm','options=nobase,avall','nxy=1,2','interval=100000','xrange=-3500,3500','device=junk.eps/vcps'],stdout=logf,stderr=logf)
		call(['epstool','--copy','--bbox','junk.eps','%s.eps'%t],stdout=logf,stderr=logf)
		os.remove('junk.eps')
	
	logprint('DONE!',logf)
	logf.close()

ap = argparse.ArgumentParser()
ap.add_argument('config_file',help='Input configuration file')
ap.add_argument('-s','--setup_file',help='Name of text file with setup correlator file names included so that they can be ignored during the processing [default setup.txt]',default='setup.txt')
ap.add_argument('-l','--log_file',help='Name of output log file [default log.txt]',default='log.txt')
args = ap.parse_args()

cfg = ConfigParser.RawConfigParser()
cfg.read(args.config_file)

main(args,cfg)

