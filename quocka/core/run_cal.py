#!/usr/bin/env python3
# TODO: Switch to pymir?

import argparse
import configparser
import glob
import logging
import os
import shutil
import subprocess as sp
from typing import List, NamedTuple, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from braceexpand import braceexpand
from casatasks import importmiriad, importuvfits
from dask import compute, delayed
from dask.distributed import Client
from IPython import embed

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logger = logging.getLogger(__name__)
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger.setLevel(logging.INFO)


class QuockaConfig(NamedTuple):
    atfiles: list
    if_use: int
    outdir: str
    rawclobber: bool
    outclobber: bool
    skipcal: bool
    prical: str
    seccal: str
    polcal: str
    setup_file: str
    NFBIN: int
    N_P_ROUNDS: int
    N_S_ROUNDS: int
    gpaver_interval: float
    convert_ms: bool


class QuockaSources(NamedTuple):
    pricalname: str
    seccalnames: list
    polcalnames: list
    targetnames: list


@delayed()
def convert_to_ms(
    vis: str,
    outdir: str,
) -> str:
    """Convert a uvfits file to a measurement set

    Args:
        vis (str): Visibility file to convert
    """
    # Now in outdir
    os.chdir(outdir)

    logger.critical(f"Converting {vis} to ms")
    logger.critical("This is experimental and may not work as expected!")

    # OPTION 1: Convert vis to uvfits - convert uvfits to ms
    # Convert vis to uvfits
    uvfits = f"{vis}.uv"
    call(
        [
            "fits",
            f"in={vis}",
            f"out={uvfits}",
            "op=uvout",
        ],
    )
    # Convert uvfits to ms
    ms = f"{uvfits}.ms"
    if os.path.exists(ms):
        logger.warning(f"Removing {ms}")
        shutil.rmtree(ms)
    importuvfits(
        fitsfile=uvfits,
        vis=ms,
    )

    # OPTION 2: Convert vis to ms directly
    ms = f"{vis}.ms"
    if os.path.exists(ms):
        logger.warning(f"Removing {ms}")
        shutil.rmtree(ms)

    importmiriad(
        mirfile=vis,
        vis=ms,
    )
    return ms


def single_compute(*args, **kwargs):
    """Compute a single task

    Args:
        *args: Arguments to pass to dask.compute
        **kwargs: Keyword arguments to pass to dask.compute
    """
    return compute(*args, **kwargs)[0]


def get_band_from_vis(vis: str) -> Tuple[List[int], int]:
    """Get the band from the vis file

    Args:
        vis (str): Visibility file

    Raises:
        ValueError: If the band cannot be found

    Returns:
        Tuple[List[int], int]: List of bands and number of bands
    """
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
    """Wrapper for subprocess.Popen to log the command and output

    All arguments are passed to subprocess.Popen

    """
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
    src: str,
) -> None:
    """Flag the data

    Args:
        src (str): Visibility file to flag
    """
    # Pgflagging lines, following the ATCA users guide.
    # Pgflagging needs to be done on all the calibrators and targets.
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
    src: str,
) -> None:
    """Flag the data (stokes V only)

    Args:
        src (str): Visibility file to flag
    """
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


def get_noise(img_name: str) -> float:
    """Extract the noise of an image

    Args:
        img_name (str): FITS image

    Returns:
        float: RMS of the image
    """
    # Get the noise of an image
    hdu = fits.open(img_name)
    data = hdu[0].data[0, 0]
    rms = np.std(data)
    hdu.close()
    return rms


def parse_config(
    config_file: str,
) -> QuockaConfig:
    """Parse the config file

    Args:
        config_file (str): Path to the config file

    Returns:
        QuockaConfig: NamedTuple with the config options
    """
    cfg = configparser.RawConfigParser()
    cfg.read(config_file)
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
    outdir = os.path.abspath(cfg.get("output", "dir"))
    rawclobber = cfg.getboolean("output", "rawclobber")
    outclobber = cfg.getboolean("output", "clobber")
    skipcal = cfg.getboolean("output", "skipcal")
    prical = cfg.get("observation", "primary")
    seccal = cfg.get("observation", "secondary")
    polcal = cfg.get("observation", "polcal")
    setup_file = cfg.get("input", "setup_file")

    NFBIN = cfg.getint("output", "nfbin")
    N_P_ROUNDS = cfg.getint("output", "nprimary")
    N_S_ROUNDS = cfg.getint("output", "nsecondary")
    gpaver_interval = cfg.getfloat("output", "gpaver_interval")
    convert_ms = cfg.getboolean("output", "convert_to_ms")

    return QuockaConfig(
        atfiles=atfiles,
        if_use=if_use,
        outdir=outdir,
        rawclobber=rawclobber,
        outclobber=outclobber,
        skipcal=skipcal,
        prical=prical,
        seccal=seccal,
        polcal=polcal,
        setup_file=setup_file,
        NFBIN=NFBIN,
        N_P_ROUNDS=N_P_ROUNDS,
        N_S_ROUNDS=N_S_ROUNDS,
        gpaver_interval=gpaver_interval,
        convert_ms=convert_ms,
    )


@delayed()
def load_visibilities(
    outdir: str,
    setup_file: str,
    atfiles: list,
    rawclobber: bool,
    if_use: int,
) -> str:
    """Load the visibilities from the correlator files

    Args:
        outdir (str): Output directory
        setup_file (str): Setup file
        atfiles (list): correlator files
        rawclobber (bool): Overwrite existing files
        if_use (int): IF to use

    Returns:
        str: Visibility file
    """
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

    if os.path.exists(outdir + "/dat.uv") and not rawclobber:
        logger.info(
            "Skipping atlod step",
        )
        return

    if os.path.exists(outdir + "/dat.uv") and rawclobber:
        logger.info(
            "Removing existing dat.uv",
        )
        shutil.rmtree(outdir + "/dat.uv")
    logger.info(
        "Running ATLOD...",
    )
    call(
        [
            "atlod",
            "in=%s" % uvlist,
            "out=%s/dat.uv" % outdir,
            f"ifsel={if_use}",
            "options=birdie,noauto,xycorr,rfiflag,notsys",
        ],
    )

    return "dat.uv"


@delayed()
def frequency_split(
    vis_file: str,
    rawclobber: bool,
    outclobber: bool,
    outdir: str,
) -> list:
    """Split the data into frequency bands

    Args:
        rawclobber (bool): Overwrite existing files
        outclobber (bool): Overwrite existing files

    Returns:
        list: List of frequency bands
    """
    # Now in outdir...
    os.chdir(outdir)
    # Now we need a uvsplit into frequency bands
    call(
        [
            "uvsplit",
            f"vis={vis_file}",
            "options=nosource,clobber" if rawclobber else "options=nosource",
        ]
    )
    # Check for double IF in 2100 band
    if os.path.exists("uvsplit.2100.1"):
        if os.path.exists("uvsplit.2100"):
            shutil.rmtree("uvsplit.2100")
        shutil.move("uvsplit.2100.1", "uvsplit.2100")
    if os.path.exists("uvsplit.2100.2"):
        shutil.rmtree("uvsplit.2100.2")

    # Run uvflagging
    # Check frequency range
    band_list, nbands = get_band_from_vis(vis_file)
    logger.info(
        f"Found {nbands} frequency bands: {band_list}",
    )

    for band in band_list:
        for line in open(f"../badchans_{band}.txt"):
            sline = line.split()
            lc, uc = sline[0].split("-")
            dc = int(uc) - int(lc) + 1
            call(
                [
                    "uvflag",
                    f"vis=uvsplit.{band}",
                    "line=chan,%d,%s" % (dc, lc),
                    "flagval=flag",
                ],
            )

    logger.info(
        "Running UVSPLIT...",
    )
    logger.info(
        "Output files will be clobbered if necessary",
    )

    for band in band_list:
        call(
            [
                "uvsplit",
                f"vis=uvsplit.{band}",
                '"select=-shadow(25)"',
                "options=mosaic,clobber" if outclobber else "options=mosaic",
            ],
        )

    return band_list


@delayed()
def find_sources(outdir: str) -> List[str]:
    # Now in outdir...
    os.chdir(outdir)
    slist = sorted(glob.glob("[j012]*.[257]???"))
    logger.info(
        "Working on %d sources" % len(slist),
    )
    return slist


@delayed()
def split_sources(
    prical: str,
    seccal: str,
    polcal: str,
    frqb: int,
    slist: list,
    outdir: str,
) -> QuockaSources:
    """Split the sources into calibrators and targets for a given frequency

    Args:
        prical (str): Primary calibrator
        seccal (str): Secondary calibrator
        polcal (str): Polarization calibrator
        frqb (int): Frequency band
        slist (list): List of sources

    Raises:
        FileNotFoundError: If the primary calibrator is not found
        FileNotFoundError: If the secondary calibrator is not found

    Returns:
        QuockaSources: Named tuple with the names of the calibrators and targets
    """
    # Now in outdir...
    os.chdir(outdir)
    logger.info(
        "\n\n##########\nWorking on frequency: %s\n##########\n\n" % (frqb),
    )
    pricalname = ""
    seccalnames = []
    polcalnames = []
    targetnames = []
    for i, source in enumerate(slist):
        frqid = int(source[-4:])
        if frqid != frqb:
            continue
        if prical in source:
            pricalname = source
        elif seccal != "" and any([sc in source for sc in seccal.split(",")]):
            seccalnames.append(source)
        elif polcal != "" and any([pc in source for pc in polcal.split(",")]):
            polcalnames.append(source)
        else:
            targetnames.append(source)
    if not pricalname:
        raise FileNotFoundError(
            "primary cal (%s) not found" % prical,
        )
    if not seccalnames:
        raise FileNotFoundError(
            "secondary cal (%s) not found" % seccal,
        )

    logger.info(
        "Identified primary cal: %s" % pricalname,
    )
    logger.info(
        "Identified %d secondary cals" % len(seccalnames),
    )
    logger.info(
        "Identified %d polarization calibrators" % len(polcalnames),
    )
    if len(polcalnames) > 1:
        logger.critical(
            """Very accurate circular polarization calibration not supported yet!
            These calibrators will be treated as targets
            """
        )
        targetnames.extend(polcalnames)
    logger.info(
        "Identified %d compact targets to calibrate" % len(targetnames),
    )

    return QuockaSources(
        pricalname=pricalname,
        seccalnames=seccalnames,
        polcalnames=polcalnames,
        targetnames=targetnames,
    )


@delayed()
def primary_cal(
    prical: str,
    pricalname: str,
    N_P_ROUNDS: int,
    NFBIN: int,
    outdir: str,
) -> str:
    """Derive bandpass and gain calibration for the primary calibrator

    Args:
        prical (str): Primary calibrator
        pricalname (str): Source name of the primary calibrator
        N_P_ROUNDS (int): Number of flag / cal rounds
        NFBIN (int): Number of frequency bins

    Returns:
        str: Calibrated primary calibrator name
    """
    # Now in outdir...
    os.chdir(outdir)
    logger.info(
        "Initial flagging round proceeding...",
    )

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
    return pricalname


@delayed()
def secondary_cal(
    pricalname: str,
    seccalname: str,
    N_S_ROUNDS: int,
    NFBIN: int,
    outdir: str,
    gpaver_interval: float = 0,
) -> str:
    """Derive the gain and phase calibration for the secondary calibrator

    Args:
        pricalname (str): Primary calibrator name
        seccalname (str): Secondary calibrator name
        N_S_ROUNDS (int): Number of flag / cal rounds
        NFBIN (int): Number of frequency bins

    Returns:
        str: Calibrated secondary calibrator name
    """
    # Now in outdir...
    os.chdir(outdir)
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
    # Apply averaging to the gain solutions if requested
    if gpaver_interval > 0:
        logger.info(
            f"Averaging secondary cal gain solutions over {gpaver_interval} min interval..."
        )
        call(
            [
                "gpaver",
                f"interval={gpaver_interval}",
                f"vis={seccalname}",
                "options=scalar",
            ],
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

    return seccalname


@delayed()
def merge_secondary_cals(
    seccalnames: List[str],
    outdir: str,
) -> str:
    """Merge secondary calibrator tables

    Args:
        seccalnames (List[str]): List of secondary calibrator names

    Returns:
        str: Merged secondary calibrator name
    """
    # Now in outdir...
    os.chdir(outdir)
    while len(seccalnames) > 1:
        logger.info(
            "Merging gain table for %s into %s ..." % (seccalnames[-1], seccalnames[0]),
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
    return seccalname


@delayed()
def target_cal(
    target: str,
    seccalname: str,
    outdir: str,
    clobber: bool = False,
) -> str:
    """Apply the calibration to the target

    Args:
        target (str): Target name
        seccalname (str): Secondary calibrator name

    Returns:
        str: Calibrated target name
    """
    # Now in outdir...
    os.chdir(outdir)
    logger.info(
        "Working on source %s" % target,
    )
    call(
        ["gpcopy", "vis=%s" % seccalname, "out=%s" % target],
    )
    flag(
        target,
    )
    flag(
        target,
    )
    logger.info("Writing source flag and pol info")
    call(["uvfstats", "vis=%s" % target])
    call(["uvfstats", "vis=%s" % target, "mode=channel"])

    # Apply the solutions before we do selfcal
    t_pscal = target + ".pscal"
    if os.path.exists(t_pscal):
        if clobber:
            logger.info(f"{t_pscal} exists and clobber is False, skipping")
            return t_pscal
        else:
            logger.info(f"{t_pscal} exists and clobber is True, removing")
            shutil.rmtree(t_pscal)

    call(["uvaver", "vis=%s" % target, "out=%s" % t_pscal])

    return t_pscal


def main(
    config_file: str,
):
    # Parse config file
    config = parse_config(config_file)

    # Load visibilities
    vis_file = load_visibilities(
        outdir=config.outdir,
        setup_file=config.setup_file,
        atfiles=config.atfiles,
        rawclobber=config.rawclobber,
        if_use=config.if_use,
    )

    band_list = frequency_split(
        vis_file=vis_file,
        rawclobber=config.rawclobber,
        outclobber=config.outclobber,
        outdir=config.outdir,
    )

    slist = find_sources(config.outdir)

    for frqb in single_compute(band_list):
        sources = split_sources(
            prical=config.prical,
            seccal=config.seccal,
            polcal=config.polcal,
            slist=slist,
            frqb=frqb,
            outdir=config.outdir,
        )
        if config.skipcal:
            logger.warning("Skipping flagging/calibration on user request.")
            continue
        # Flagging/calibrating the primary calibrator 1934-638.
        pricalname_cal = primary_cal(
            prical=config.prical,
            pricalname=sources.pricalname,
            N_P_ROUNDS=config.N_P_ROUNDS,
            NFBIN=config.NFBIN,
            outdir=config.outdir,
        )
        # Move on to the secondary calibrators
        secal_list = []
        for seccalname in single_compute(sources.seccalnames):
            secalname_cal = secondary_cal(
                pricalname=pricalname_cal,
                seccalname=seccalname,
                N_S_ROUNDS=config.N_S_ROUNDS,
                NFBIN=config.NFBIN,
                outdir=config.outdir,
                gpaver_interval=config.gpaver_interval,
            )
            secal_list.append(secalname_cal)

        # Merge the secondary calibrators
        merged_cal = merge_secondary_cals(
            seccalnames=secal_list,
            outdir=config.outdir,
        )
        # Move on to the target!
        logger.info(
            "\n\n##########\nApplying calibration to target sources...\n##########\n\n",
        )
        target_list = []
        for target in single_compute(sources.targetnames):
            targetname_cal = target_cal(
                target=target,
                seccalname=merged_cal,
                outdir=config.outdir,
                clobber=config.outclobber,
            )
            if config.convert_ms:
                targetname_ms = convert_to_ms(
                    targetname_cal,
                    outdir=config.outdir,
                )
                target_list.append(targetname_ms)
            else:
                target_list.append(targetname_cal)

        targets = single_compute(target_list)
        logger.info(
            "\n\n##########\nFinished calibrating target sources!\n##########\n\n",
        )
        logger.info(
            "Calibrated targets:",
        )
        logger.info(
            targets,
        )

    logger.info("DONE!")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file", help="Input configuration file")
    ap.add_argument(
        "-l",
        "--log_file",
        help="Name of output log file [default log.txt]",
        default="log.txt",
    )
    args = ap.parse_args()

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
    with Client(threads_per_worker=1) as client:
        logger.info(
            "Dask settings:",
        )
        logger.info(
            f"Dask dashboard available at: {client.dashboard_link}",
        )
        logger.info(
            client,
        )
        main(args.config_file)


if __name__ == "__main__":
    cli()
