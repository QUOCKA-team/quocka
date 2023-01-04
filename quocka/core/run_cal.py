#!/usr/bin/env python3
# TODO: Switch to pymir?

import argparse
import configparser
import glob
import logging
import os
import shutil
import subprocess as sp

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from braceexpand import braceexpand
from numpy import unique

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logger = logging.getLogger(__name__)
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger.setLevel(logging.INFO)


def get_band_from_vis(vis):
    # Get the band from the vis file
    band_lut = {
        3.124: 2100,
        4.476: 5500,
    }
    nband_max = 5  # There are 5 bands at ATCA

    # Run uvindex to get vis metadata
    output = sp.run(
        f"uvindex vis={vis}".split(),
        check=True,
        capture_output=True,
    )
    lines = output.stdout.decode("utf-8").splitlines()

    # Loop over the frequency configurations
    band_list = []
    nbands = 0
    for i in range(nband_max):
        # Find line with 'Frequency Configuration'
        try:
            idx = lines.index(f"Frequency Configuration {i+1}")
        except ValueError:
            continue
        data = "\n".join(lines[idx + 1 : idx + 3])
        df = Table.read(data.replace("GHz", ""), format="ascii").to_pandas()

        # Get the band
        band = band_lut.get(df["Freq(chan=1)"][0], None)
        if band is None:
            raise ValueError(
                f"Could not find band for {vis} (Freq(chan=1)={df['Freq(chan=1)'][0]})"
            )
        band_list.append(band)
        nbands += 1
    return band_list, nbands


def call(*args, **kwargs):
    # Call a subprocess, print the command to stdout
    logger.info(" ".join(args[0]))
    process = sp.Popen(*args, stdout=sp.PIPE, stderr=sp.STDOUT, **kwargs)

    with process.stdout:
        try:
            for line in iter(process.stdout.readline, b""):
                logger.info(line.decode("utf-8").strip())

        except sp.CalledProcessError as e:
            logger.error(f"{str(e)}")


def flag(
    src,
):
    # Pgflagging lines, following the ATCA users guide. Pgflagging needs to be done on all the calibrators and targets.
    call(
        [
            "pgflag",
            "vis=%s" % src,
            "stokes=i,q,u,v",
            "flagpar=8,5,5,3,6,3",
            "command=<b",
            "options=nodisp",
        ],
    )
    call(
        [
            "pgflag",
            "vis=%s" % src,
            "stokes=i,v,u,q",
            "flagpar=8,2,2,3,6,3",
            "command=<b",
            "options=nodisp",
        ],
    )
    call(
        [
            "pgflag",
            "vis=%s" % src,
            "stokes=i,v,q,u",
            "flagpar=8,2,2,3,6,3",
            "command=<b",
            "options=nodisp",
        ],
    )


def flag_v(
    src,
):
    # pgflagging, stokes V only.
    call(
        [
            "pgflag",
            "vis=%s" % src,
            "stokes=i,q,u,v",
            "flagpar=8,5,5,3,6,3",
            "command=<b",
            "options=nodisp",
        ],
    )


def get_noise(img_name):
    # Get the noise of an image
    hdu = fits.open(img_name)
    data = hdu[0].data[0, 0]
    rms = np.std(data)
    hdu.close()
    return rms


def main(
    cfg,
    setup_file,
):
    # Initiate log file with options used
    logger.info(
        "Config settings:",
    )
    logger.info(
        cfg.items("input"),
    )
    logger.info(
        cfg.items("output"),
    )
    logger.info(
        cfg.items("observation"),
    )
    logger.info(
        "",
    )

    gwcp = cfg.get("input", "dir") + "/" + cfg.get("input", "date") + "*"
    atfiles = []
    for g in braceexpand(gwcp):
        atfiles.extend(glob.glob(g))
    atfiles = sorted(atfiles)
    if_use = cfg.getint("input", "if_use")
    outdir = cfg.get("output", "dir")
    rawclobber = cfg.getboolean("output", "rawclobber")
    outclobber = cfg.getboolean("output", "clobber")
    skipcal = cfg.getboolean("output", "skipcal")
    prical = cfg.get("observation", "primary")
    seccal = cfg.get("observation", "secondary")
    polcal = cfg.get("observation", "polcal")
    seccal_ext = cfg.get("observation", "sec_ext")
    target_ext = cfg.get("observation", "ext")

    # Set globals
    global NFBIN
    NFBIN = cfg.getint("output", "nfbin")
    global N_P_ROUNDS
    N_P_ROUNDS = cfg.getint("output", "nprimary")
    global N_S_ROUNDS
    N_S_ROUNDS = cfg.getint("output", "nsecondary")

    if not os.path.exists(outdir):
        logger.info(
            "Creating directory %s" % outdir,
        )
        os.makedirs(outdir)
    for line in open(setup_file):
        if line[0] == "#":
            continue
        sline = line.split()
        for a in atfiles:
            if sline[0] in a:
                logger.info(
                    "Ignoring setup file %s" % sline[0],
                )
                atfiles.remove(a)
    uvlist = ",".join(atfiles)

    if not os.path.exists(outdir + "/dat.uv") or rawclobber:
        logger.info(
            "Running ATLOD...",
        )
        call(
            [
                "atlod",
                "in=%s" % uvlist,
                "out=%s/dat.uv" % outdir,
                "ifsel=1",
                "options=birdie,noauto,xycorr,rfiflag,notsys",
            ],
        )
    else:
        logger.info(
            "Skipping atlod step",
        )

    # Now in outdir...
    os.chdir(outdir)
    # Run uvflagging

    # Check frequency range
    band_list, nbands = get_band_from_vis("dat.uv")

    if nbands == 1:
        for line in open("../badchans_%s.txt" % band_list[0]):
            sline = line.split()
            lc, uc = sline[0].split("-")
            dc = int(uc) - int(lc) + 1
            call(
                ["uvflag", "vis=dat.uv", "line=chan,%d,%s" % (dc, lc), "flagval=flag"],
            )
    else:
        # TODO: implement flagging for multiple bands
        logger.warning(
            "Multiple frequency bands found. Skipping uvflagging",
        )

    logger.info(
        "Running UVSPLIT...",
    )
    logger.info(
        "Output files will be clobbered if necessary",
    )
    call(
        [
            "uvsplit",
            "vis=dat.uv",
            '"select=-shadow(25)"',
            "options=mosaic,clobber" if outclobber else "options=mosaic",
        ],
    )

    slist = sorted(glob.glob("[j012]*.[257]???"))

    logger.info(
        "Working on %d sources" % len(slist),
    )
    bandfreq = unique([x[-4:] for x in slist])
    logger.info(
        "Frequency bands to process: %s" % (",".join(bandfreq)),
    )

    src_to_plot = []
    for frqb in bandfreq:
        logger.info(
            "\n\n##########\nWorking on frequency: %s\n##########\n\n" % (frqb),
        )
        pricalname = "__NOT_FOUND__"
        ext_seccalname = "__NOT_FOUND__"
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
            elif seccal != "" and any([sc in source for sc in seccal.split(",")]):
                seccalnames.append(source)
            elif seccal_ext in source and seccal_ext != "":
                ext_seccalname = source
            elif polcal != "" and any([pc in source for pc in polcal.split(",")]):
                polcalnames.append(source)
            elif target_ext != "" and any(
                [es in source for es in target_ext.split(",")]
            ):
                ext_targetnames.append(source)
            else:
                targetnames.append(source)
                src_to_plot.append(source[:-5])
        if pricalname == "__NOT_FOUND__":
            raise FileNotFoundError(
                "primary cal (%s) not found" % prical,
            )
        if len(seccalnames) == 0:
            raise FileNotFoundError(
                "secondary cal (%s) not found" % seccal,
            )
        if (
            ext_seccalname == "__NOT_FOUND__"
            and seccal_ext != "NONE"
            and len(ext_targetnames) != 0
        ):
            raise FileNotFoundError(
                "extended-source secondary cal (%s) not found" % seccal_ext,
            )
        elif seccal_ext == "NONE":
            ext_seccalname = "(NONE)"

        logger.info(
            "Identified primary cal: %s" % pricalname,
        )
        logger.info(
            "Identified %d secondary cals" % len(seccalnames),
        )
        logger.info(
            "Identified %d polarization calibrators" % len(polcalnames),
        )
        logger.info(
            "Identified %d compact targets to calibrate" % len(targetnames),
        )
        logger.info(
            "Identified secondary cal for extended sources: %s" % ext_seccalname,
        )
        logger.info(
            "Identified %d extended targets to calibrate" % len(ext_targetnames),
        )
        if skipcal:
            logger.info(
                "Skipping flagging and calibration steps on user request.",
            )
            continue
        logger.info(
            "Initial flagging round proceeding...",
        )

        # Flagging/calibrating the primary calibrator 1934-638.
        logger.info(
            "Calibration of primary cal (%s) proceeding ..." % prical,
        )
        # Only select data above elevation=40.
        call(
            [
                "uvflag",
                "vis=%s" % pricalname,
                "select=-elevation(40,90)",
                "flagval=flag",
            ],
        )

        no_1934 = pricalname == "2052-474.2100"
        # Flag / cal loops on primary
        for _ in range(N_P_ROUNDS):
            flag(
                pricalname,
            )
            call(
                [
                    "mfcal",
                    "vis=%s" % pricalname,
                    "interval=0.1,1,30",
                    "flux=1.6025794,2.211,-0.3699236" if no_1934 else "",
                ],
            )
            call(
                [
                    "gpcal",
                    "vis=%s" % pricalname,
                    "interval=0.1",
                    "nfbin=%d" % NFBIN,
                    "options=xyvary",
                ],
            )
        if no_1934:
            call(
                ["mfboot", "vis=%s" % pricalname, "flux=1.6025794,2.211,-0.3699236"],
            )

        # Plot results
        call(
            [
                "uvplt",
                "vis=%s" % pricalname,
                "options=nof,nob,2pass",
                "stokes=i",
                "axis=time,amp",
                "device=%s_time_amp.ps/ps" % (pricalname),
            ],
        )
        call(
            [
                "ps2pdf",
                "%s_time_amp.ps" % (pricalname),
            ],
        )
        call(
            [
                "uvplt",
                "vis=%s" % pricalname,
                "options=nof,nob,2pass",
                "stokes=i",
                "axis=freq,amp",
                "device=%s_freq_amp.ps/ps" % (pricalname),
            ],
        )
        call(
            [
                "ps2pdf",
                "%s_freq_amp.ps" % (pricalname),
            ],
        )

        # Move on to the secondary calibrator
        for seccalname in seccalnames:
            logger.info(
                "Transferring to compact-source secondary %s..." % seccalname,
            )
            call(
                ["gpcopy", "vis=%s" % pricalname, "out=%s" % seccalname],
            )
            # Flag / cal loops on secondary
            for _ in range(N_S_ROUNDS):
                flag(
                    seccalname,
                )
                call(
                    [
                        "gpcal",
                        "vis=%s" % seccalname,
                        "interval=0.1",
                        "nfbin=%d" % NFBIN,
                        "options=xyvary,qusolve",
                    ],
                )

            # Plot results before boot
            call(
                [
                    "uvfmeas",
                    "vis=%s" % seccalname,
                    "stokes=i",
                    "order=2",
                    "options=log,mfflux",
                    "device=%s_uvfmeas_preboot.ps/ps" % (seccalname),
                    "feval=2.1",
                ],
            )
            call(
                [
                    "ps2pdf",
                    "%s_uvfmeas_preboot.ps" % (seccalname),
                ],
            )

            # boot the flux
            call(
                ["gpboot", "vis=%s" % seccalname, "cal=%s" % pricalname],
            )
            # Plot results after boot
            call(
                [
                    "uvfmeas",
                    "vis=%s" % seccalname,
                    "stokes=i",
                    "order=2",
                    "options=log,mfflux",
                    "device=%s_uvfmeas_postboot.ps/ps" % (seccalname),
                    "feval=2.1",
                ],
            )
            call(
                [
                    "ps2pdf",
                    "%s_uvfmeas_postboot.ps" % (seccalname),
                ],
            )

        while len(seccalnames) > 1:
            logger.info(
                "Merging gain table for %s into %s ..."
                % (seccalnames[-1], seccalnames[0]),
            )
            call(
                [
                    "gpcopy",
                    "vis=%s" % seccalnames[-1],
                    "out=%s" % seccalnames[0],
                    "mode=merge",
                ],
            )
            del seccalnames[-1]
        seccalname = seccalnames[0]
        logger.info(
            "Using gains from %s ..." % (seccalname),
        )
        logger.info(
            "\n\n##########\nApplying calibration to compact sources...\n##########\n\n",
        )
        for t in targetnames:
            logger.info(
                "Working on source %s" % t,
            )
            # Move on to the target!
            call(
                ["gpcopy", "vis=%s" % seccalname, "out=%s" % t],
            )
            flag(
                t,
            )
            flag(
                t,
            )
            logger.info("Writing source flag and pol info")
            call(["uvfstats", "vis=%s" % t])
            call(["uvfstats", "vis=%s" % t, "mode=channel"])

            # Apply the solutions before we do selfcal
            t_pscal = t + ".pscal"
            call(["uvaver", "vis=%s" % t, "out=%s" % t_pscal])

    logger.info(
        "DONE!",
    )


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file", help="Input configuration file")
    ap.add_argument(
        "-s",
        "--setup_file",
        help="Name of text file with setup correlator file names included so that they can be ignored during the processing [default setup.txt]",
        default="setup.txt",
    )
    ap.add_argument(
        "-l",
        "--log_file",
        help="Name of output log file [default log.txt]",
        default="log.txt",
    )
    args = ap.parse_args()

    cfg = configparser.RawConfigParser()
    cfg.read(args.config_file)

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(
        "Command-line settings:",
    )
    logger.info(
        args,
    )

    main(cfg, args.setup_file)


if __name__ == "__main__":
    cli()
