#!/usr/bin/env python3

"""
Set up the QUOCKA directory structure and copy the default config files.
"""

import argparse
import logging
import os
import shutil

import pkg_resources

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
logger.setLevel(logging.INFO)


def try_symlink(src: str, dst: str):
    """Create symlink if it doesn't exist
    Args:
        src (str): Source path
        dst (str): Destination path
    """
    # Create output dir if it doesn't exist
    try:
        os.symlink(src, dst)
        logger.info(f"Made symlink '{dst}'.")
    except FileExistsError:
        logger.info(f"Symlink '{dst}' exists.")


# Stolen from https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            print("{}{}".format(subindent, f))


def main(
    raw_vis_list=[],
    out_dir="",
):
    # Check that the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    logger.info(
        f"Setting up QUOCKA directory structure in {os.path.abspath(out_dir)} ..."
    )

    # Create the output directory structure
    raw_dir = os.path.join(out_dir, "raw")
    cal_dir = os.path.join(out_dir, "cal")
    for d in [raw_dir, cal_dir]:
        os.makedirs(d, exist_ok=True)

    # Copy the raw visbilities to the raw directory
    for vis in raw_vis_list:
        try_symlink(os.path.abspath(vis), os.path.join(raw_dir, os.path.basename(vis)))

    # Copy the default config files to the cal directory
    config_dir = pkg_resources.resource_filename("quocka", "data")
    config_files = [
        "badchans_2100.txt",
        "badchans_5500.txt",
        "badchans_7500.txt",
        "template.cfg",
        "setup.txt",
    ]
    for f in config_files:
        shutil.copy(os.path.join(config_dir, f), cal_dir)

    list_files(out_dir)


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "raw_vis_list",
        nargs="+",
        help="List of raw visibility files to copy to the raw directory",
    )
    parser.add_argument("-o", "--out-dir", default=".", help="Output directory")
    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    cli()
