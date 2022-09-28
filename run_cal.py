#!/usr/bin/env python
# TODO: Remove old code
# TODO: Print mir commands to log file
# TODO: UVflag before uvsplit
# TODO: Switch to pymir?

import argparse
import configparser
import glob
import os
from subprocess import call
from numpy import unique
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
import numpy as np
import shutil
from braceexpand import braceexpand


## GLOBALS
# change nfbin to 2
NFBIN = 2

def logprint(s2p, lf):
    print(s2p, file=lf)
    print(s2p)



def flag(src, logf):
    # Pgflagging lines, following the ATCA users guide. Pgflagging needs to be done on all the calibrators and targets.
    call(['pgflag', 'vis=%s' % src, 'stokes=i,q,u,v', 'flagpar=8,5,5,3,6,3',
          'command=<b', 'options=nodisp'], stdout=logf, stderr=logf)
    call(['pgflag', 'vis=%s' % src, 'stokes=i,v,u,q', 'flagpar=8,2,2,3,6,3',
          'command=<b', 'options=nodisp'], stdout=logf, stderr=logf)
    call(['pgflag', 'vis=%s' % src, 'stokes=i,v,q,u', 'flagpar=8,2,2,3,6,3',
          'command=<b', 'options=nodisp'], stdout=logf, stderr=logf)


def flag_v(src, logf):
    # pgflagging, stokes V only.
    call(['pgflag', 'vis=%s' % src, 'stokes=i,q,u,v', 'flagpar=8,5,5,3,6,3',
          'command=<b', 'options=nodisp'], stdout=logf, stderr=logf)


def get_noise(img_name):
    # Get the noise of an image
    hdu = fits.open(img_name)
    data = hdu[0].data[0, 0]
    rms = np.std(data)
    hdu.close()
    return rms

def main(args, cfg):
    # Initiate log file with options used
    logf = open(args.log_file, 'w', 1)  # line buffered
    logprint('Input settings:', logf)
    logprint(args, logf)
    logprint(cfg.items('input'), logf)
    logprint(cfg.items('output'), logf)
    logprint(cfg.items('observation'), logf)
    logprint('', logf)

    gwcp = cfg.get('input', 'dir')+'/'+cfg.get('input', 'date')+'*'
    atfiles = []
    for g in braceexpand(gwcp):
        atfiles.extend(glob.glob(g))
    atfiles = sorted(atfiles)
    if_use = cfg.getint('input', 'if_use')
    outdir = cfg.get('output', 'dir')
    rawclobber = cfg.getboolean('output', 'rawclobber')
    outclobber = cfg.getboolean('output', 'clobber')
    skipcal = cfg.getboolean('output', 'skipcal')
    prical = cfg.get('observation', 'primary')
    seccal = cfg.get('observation', 'secondary')
    polcal = cfg.get('observation', 'polcal')
    seccal_ext = cfg.get('observation', 'sec_ext')
    target_ext = cfg.get('observation', 'ext')

    if not os.path.exists(outdir):
        logprint('Creating directory %s' % outdir, logf)
        os.makedirs(outdir)
    for line in open(args.setup_file):
        if line[0] == '#':
            continue
        sline = line.split()
        for a in atfiles:
            if sline[0] in a:
                logprint('Ignoring setup file %s' % sline[0], logf)
                atfiles.remove(a)
    uvlist = ','.join(atfiles)

    if not os.path.exists(outdir+'/dat.uv') or rawclobber:
        logprint('Running ATLOD...', logf)
        if if_use > 0:
            call(['atlod', 'in=%s' % uvlist, 'out=%s/dat.uv' % outdir, 'ifsel=%s' % if_use,
                  'options=birdie,noauto,xycorr,rfiflag,notsys'], stdout=logf, stderr=logf)
        else:
            call(['atlod', 'in=%s' % uvlist, 'out=%s/dat.uv' % outdir,
                  'options=birdie,noauto,xycorr,rfiflag'], stdout=logf, stderr=logf)
    else:
        logprint('Skipping atlod step', logf)
    os.chdir(outdir)
    logprint('Running UVSPLIT...', logf)
    if outclobber:
        logprint('Output files will be clobbered if necessary', logf)
        call(['uvsplit', 'vis=dat.uv', 'options=mosaic,clobber'],
             stdout=logf, stderr=logf)
    else:
        call(['uvsplit', 'vis=dat.uv', 'options=mosaic'],
             stdout=logf, stderr=logf)

    slist = sorted(glob.glob('[j012]*.[257]???'))

    logprint('Working on %d sources' % len(slist), logf)
    bandfreq = unique([x[-4:] for x in slist])
    logprint('Frequency bands to process: %s' % (','.join(bandfreq)), logf)

    src_to_plot = []
    for frqb in bandfreq:
        logprint(
            '\n\n##########\nWorking on frequency: %s\n##########\n\n' % (frqb), logf)
        pricalname = '__NOT_FOUND__'
        ext_seccalname = '__NOT_FOUND__'
        seccalnames = []
        polcalnames = []
        targetnames = []
        ext_targetnames = []
        for i, source in enumerate(slist):
            frqid = source[-4:]
            if frqid not in frqb:
                continue
            if prical in source:
                pricalname = source
            elif seccal != '' and any([sc in source for sc in seccal.split(',')]):
                seccalnames.append(source)
            elif seccal_ext in source and seccal_ext != '':
                ext_seccalname = source
            elif polcal != '' and any([pc in source for pc in polcal.split(',')]):
                polcalnames.append(source)
            elif target_ext != '' and any([es in source for es in target_ext.split(',')]):
                ext_targetnames.append(source)
            else:
                targetnames.append(source)
                src_to_plot.append(source[:-5])
        if pricalname == '__NOT_FOUND__':
            logprint('Error: primary cal (%s) not found' % prical, logf)
            logf.close()
            exit(1)
        if len(seccalnames) == 0:
            logprint('Error: secondary cal (%s) not found' % seccal, logf)
            logf.close()
            exit(1)
        if ext_seccalname == '__NOT_FOUND__' and seccal_ext != 'NONE' and len(ext_targetnames) != 0:
            logprint('Error: extended-source secondary cal (%s) not found' %
                     seccal_ext, logf)
            logf.close()
            exit(1)
        elif seccal_ext == 'NONE':
            ext_seccalname = '(NONE)'
        logprint('Identified primary cal: %s' % pricalname, logf)
        logprint('Identified %d secondary cals' % len(seccalnames), logf)
        logprint('Identified %d polarization calibrators' %
                 len(polcalnames), logf)
        logprint('Identified %d compact targets to calibrate' %
                 len(targetnames), logf)
        logprint('Identified secondary cal for extended sources: %s' %
                 ext_seccalname, logf)
        logprint('Identified %d extended targets to calibrate' %
                 len(ext_targetnames), logf)
        if skipcal:
            logprint(
                'Skipping flagging and calibration steps on user request.', logf)
            continue
        logprint('Initial flagging round proceeding...', logf)
        for i, source in enumerate(slist):
            # only flag data corresponding to the data that we're dealing with (resolves issue #4)
            if frqb not in source:
                continue
            logprint('\nFLAGGING: %d / %d = %s' %
                     (i+1, len(slist), source), logf)
            ####
            # This part may be largely obsolete with options=rfiflag in ATLOD.
            # However, options=rfiflag doesn't cover all the badchans, so we'll still do this. -XZ
            for line in open('../badchans_%s.txt' % frqid):
                sline = line.split()
                lc, uc = sline[0].split('-')
                dc = int(uc)-int(lc)+1
                call(['uvflag', 'vis=%s' % source, 'line=chan,%d,%s' %
                      (dc, lc), 'flagval=flag'], stdout=logf, stderr=logf)
            ####

        # Flagging/calibrating the primary calibrator 1934-638.
        logprint('Calibration of primary cal (%s) proceeding ...' %
                 prical, logf)
        # Only select data above elevation=40.
        call(['uvflag', 'vis=%s' % pricalname, 'select=-elevation(40,90)',
              'flagval=flag'], stdout=logf, stderr=logf)
        flag_v(pricalname, logf)
        # XZ: this part is modified to fix the "no 1934" issue on 2019-06-23. Comment the following three lines if used otherwise.
        if pricalname == '2052-474.2100':
            call(['mfcal', 'vis=%s' % pricalname, 'flux=1.6025794,2.211,-0.3699236',
                  'interval=0.1,1,30'], stdout=logf, stderr=logf)
        else:
            call(['mfcal', 'vis=%s' % pricalname, 'interval=0.1,1,30'],
                 stdout=logf, stderr=logf)
        flag(pricalname, logf)
        call(['gpcal', 'vis=%s' % pricalname, 'interval=0.1', 'nfbin=%d' %
              NFBIN, 'options=xyvary'], stdout=logf, stderr=logf)
        flag(pricalname, logf)
        if pricalname == '2052-474.2100':
            call(['mfboot', 'vis=%s' % pricalname,
                  'flux=1.6025794,2.211,-0.3699236'], stdout=logf, stderr=logf)

        # Move on to the secondary calibrator
        for seccalname in seccalnames:
            logprint('Transferring to compact-source secondary %s...' %
                     seccalname, logf)
            call(['gpcopy', 'vis=%s' % pricalname, 'out=%s' %
                  seccalname], stdout=logf, stderr=logf)
            # flag twice, gpcal twice
            flag(seccalname, logf)
            call(['gpcal', 'vis=%s' % seccalname, 'interval=0.1', 'nfbin=%d' %
                  NFBIN, 'options=xyvary,qusolve'], stdout=logf, stderr=logf)
            flag(seccalname, logf)
            # boot the flux
            call(['gpboot', 'vis=%s' % seccalname, 'cal=%s' %
                  pricalname], stdout=logf, stderr=logf)

        while len(seccalnames) > 1:
            logprint('Merging gain table for %s into %s ...' %
                     (seccalnames[-1], seccalnames[0]), logf)
            call(['gpcopy', 'vis=%s' % seccalnames[-1], 'out=%s' %
                  seccalnames[0], 'mode=merge'], stdout=logf, stderr=logf)
            del seccalnames[-1]
        seccalname = seccalnames[0]
        logprint('Using gains from %s ...' % (seccalname), logf)
        logprint(
            '\n\n##########\nApplying calibration to compact sources...\n##########\n\n', logf)
        for t in targetnames:
            logprint('Working on source %s' % t, logf)
            slogname = '%s.log.txt' % t
            slogf = open(slogname, 'w', 1)

            # Move on to the target!
            call(['gpcopy', 'vis=%s' % seccalname, 'out=%s' %
                  t], stdout=logf, stderr=logf)
            flag(t, logf)
            flag(t, logf)
            logprint('Writing source flag and pol info to %s' % slogname, logf)
            call(['uvfstats', 'vis=%s' % t], stdout=slogf, stderr=slogf)
            call(['uvfstats', 'vis=%s' % t, 'mode=channel'],
                 stdout=slogf, stderr=slogf)
            slogf.close()

            # Apply the solutions before we do selfcal
            t_pscal = t + '.pscal'
            call(['uvaver', 'vis=%s' % t, 'out=%s' %
                  t_pscal], stdout=logf, stderr=logf)

    for t in sorted(unique(src_to_plot)):
        logprint('Plotting RMSF for %s' % t, logf)
        if int(bandfreq[0]) < 3500:
            call(['uvspec', 'vis=%s.????' % t, 'axis=rm', 'options=nobase,avall', 'nxy=1,2',
                  'interval=100000', 'xrange=-1500,1500', 'device=junk.eps/vcps'], stdout=logf, stderr=logf)
        else:
            # For CX, the IFs have to be catenated before RM Synthesis
            call(['uvcat', 'vis=%s.????' % t, 'out=%s.cx' %
                  t], stdout=logf, stderr=logf)
            call(['uvspec', 'vis=%s.cx' % t, 'axis=rm', 'options=nobase,avall', 'nxy=1,2',
                  'interval=100000', 'xrange=-3500,3500', 'device=junk.eps/vcps'], stdout=logf, stderr=logf)
        call(['epstool', '--copy', '--bbox', 'junk.eps',
              '%s.eps' % t], stdout=logf, stderr=logf)
        os.remove('junk.eps')

    logprint('DONE!', logf)
    logf.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config_file', help='Input configuration file')
    ap.add_argument('-s', '--setup_file',
                    help='Name of text file with setup correlator file names included so that they can be ignored during the processing [default setup.txt]', default='setup.txt')
    ap.add_argument('-l', '--log_file',
                    help='Name of output log file [default log.txt]', default='log.txt')
    args = ap.parse_args()

    cfg = configparser.RawConfigParser()
    cfg.read(args.config_file)

    main(args, cfg)