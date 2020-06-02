#!/usr/bin/env python
"""Make QUOCKA cubes"""

import schwimmbad
import sys

def main(pool, args, verbose=False):
    """Main script
    """
    pass

def cli():
    """Command-line interface
    """
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Smooth a field of 2D images to a common resolution.

    Names of output files are 'infile'.sm.fits

    NOTE: Glob is used to parse wildcards. So if you want to run on 
        *.fits, use: python beamcon_2D.py '*.fits'
        i.e. parse the wildcard as a string.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        'datadir',
        metavar='datadir',
        type=str,
        help='Directory containing a single QUOCKA field images.')

    parser.add_argument(
        '-p',
        '--prefix',
        dest='prefix',
        type=str,
        default=None,
        help='Add prefix to output filenames.')

    parser.add_argument(
        '-o',
        '--outdir',
        dest='outdir',
        type=str,
        default=None,
        help='Output directory of smoothed FITS image(s) [./].')

    parser.add_argument(
        "-v", 
        "--verbose", 
        dest="verbose", 
        action="store_true",
        help="verbose output [False].")

    parser.add_argument(
        "-d", 
        "--dryrun", 
        dest="dryrun", 
        action="store_true",
        help="Compute common beam and stop [False].")

    parser.add_argument(
        "-t",
        "--tolerance",
        dest="tolerance",
        type=float,
        default=0.0001,
        help="tolerance for radio_beam.commonbeam.")

    parser.add_argument(
        "-e",
        "--epsilon",
        dest="epsilon",
        type=float,
        default=0.0005,
        help="epsilon for radio_beam.commonbeam.")

    parser.add_argument(
        "-n",
        "--nsamps",
        dest="nsamps",
        type=int,
        default=200,
        help="nsamps for radio_beam.commonbeam.")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    if args.mpi:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    verbose = args.verbose

    main(pool, args, verbose=verbose)
    pool.close()


if __name__ == "__main__":
    cli()