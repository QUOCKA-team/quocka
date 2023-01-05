# QUOCKA

This repository provides a processing script used to calibrate data from the ATCA QUOCKA survey of polarized radio galaxies.

## Installation

This is a Python 3 package.

Clone this repository and install using `pip`:
```bash
git clone https://github.com/gheald/quocka
cd quocka
pip install .
```

## Usage

The `quocka` pipeline primary provides a calibration strategy for ATCA CABB data in the L and C/X bands. You'll need to obtain data in the CABB format (usually from the ATOA, with the name `*.C1234`).

To initial the directory structure, run `quocka init` (pointing to your raw data):
```bash
❯ quocka_init -h
usage: quocka_init [-h] [-o OUT_DIR] raw_vis_list [raw_vis_list ...]

Set up the QUOCKA directory structure and copy the default config files.

positional arguments:
  raw_vis_list          List of raw visibility files to copy to the raw directory

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_DIR, --out-dir OUT_DIR
                        Output directory
```

This will create a directory structure like:
```
.
├── cal
│   ├── badchans_2100.txt
│   ├── badchans_5500.txt
│   ├── badchans_7500.txt
│   ├── setup.txt
│   └── template.cfg
└── raw
    └── raw_data.C1234
```
Your raw data will be symlinked into the `raw` directory. In the `cal` directory you'll find the default calibration setup files. You'll need to edit these to suit your data.

### Setup files

  - `badchans_*.txt`: A list of bad channels to flag. These are in the format `start-end` where `start` and `end` are the first and last channels to flag. The files are named according to the frequency of the bandpass calibration. The default files are for the L band (2100 MHz), C band (5500 MHz) and X band (7500 MHz).
  - `setup.txt`: A list of setup correlator files that should be ignored by the pipeline.
  - `template.cfg`: A template calibration configuration file. Here you can set the input and output data, calibration steps to run, and which sources to use for calibration. You'll need to ensure observations of the calibration sources are in the `raw` directory.

### Calibration
To run the primary calibration stages, run `quocka_cal`:
```bash
❯ quocka_cal -h
usage: quocka_cal [-h] [-s SETUP_FILE] [-l LOG_FILE] config_file

positional arguments:
  config_file           Input configuration file

optional arguments:
  -h, --help            show this help message and exit
  -s SETUP_FILE, --setup_file SETUP_FILE
                        Name of text file with setup correlator file names included so that they can be ignored during
                        the processing [default setup.txt]
  -l LOG_FILE, --log_file LOG_FILE
                        Name of output log file [default log.txt]
```
This will split the raw visibilities and perform flagging, gain calibration, bandpass calibration, and cross calibration using the primary and secondary calibrator sources. The output data will be written to the directory specified in the configuration file.

### Self-calibration
To run the self-calibration stages, run `quocka_selfcal`:
```bash
❯ quocka_selfcal -h
usage: quocka_selfcal [-h] [--ncores NCORES] [-l LOG_FILE] config_file vislist [vislist ...]

positional arguments:
  config_file           Input configuration file
  vislist

optional arguments:
  -h, --help            show this help message and exit
  --ncores NCORES
  -l LOG_FILE, --log_file LOG_FILE
                        Name of output log file [default log.txt]
```
This will run self-calibration in parallel on the specified visibilities. The output data will be written to the directory specified in the configuration file.

### Imaging
TODO

### Auxiliary scripts
The `quocka` package also provides a number of auxiliary scripts for working with the data.
- cutout_400
- cutout_source_finding
- get_spec_coor
- makebigcube
- qu_fdf
- quocka_bin_cx
- quocka_simulate
- run_chanimage
- selfcal_quality
- source_finding_quality
- uptimes

TODO: Write usage instructions.

TODO: Add these as scripts in `pyproject.toml`.

## License
MIT. See `LICENSE` for details.

## Calibration strategy
The QUOCKA pipeline is mostly based on the recommened strategy outlined in the [ATCA User Guide](https://www.atnf.csiro.au/computing/software/miriad/tutorials.html) and uses the [MIRIAD](https://www.atnf.csiro.au/computing/software/miriad/) software package.

TODO: Add more details.

## Acknowledgements
TODO

## Contributing
TODO