# fetchAZA

Repository for reading pressure data from the [Sonardyne Fetch AZA](https://www.sonardyne.com/product/fetch-aza-2/) into netCDF (xarray) format.


During the reading of files, there is an intermediate step where logging event lines in the `*.csv` data file are separated into individual netCDF files.  In the processing step, the hourly data are combined into a single netCDF file (comprising PIES, DQZ, KLR, etc events, into `*use.nc`) and the AZA sequence events that fit a valid sequence of AZS-AZA-AZA-AZA-AZS patterns are merged into a second netCDF file (`*AZAseq.nc*`)

This is a work in progress as part of the [EPOC](http://epoc-eu.org) project.

## Install

Install from PyPI with

```
python -m pip install fetchAZA
```

## Documentation

For documentation, see http://ocean-uhh.github.io/fetchAZA.

Check out the demo notebook `notebooks/demo.ipynb` for example functionality.

As input, fetchAZA takes Sonardyne Fetch AZA `*.csv` files.  See [8318-FS Issue06.pdf](https://github.com/ocean-uhh/fetchAZA/blob/main/docs/source/_static/8318-FS%20Issue06.pdf) for details of the data format, or check out the snippet in `data/sample_data.csv`.

## Contributing

Contributions welcome!

To install a local, development version of fetchAZA, clone the repo, open a terminal in the root directory (next to this README.md file) and run these commands:

```
git clone https://github.com/ocean-uhh/fetchAZA.git
cd fetchAZA
pip install -r requirements-dev.txt
pip install -e .
```

This installs fetchAZA locally.  `-e` ensures that any edits you make in the files will be picked up by scripts that import functions from fetchAZA.  You can run the example jupyter notebook by launching jupyterlab with `jupyter-lab` and navigating to the `notebooks` directory.

All new functions should include tests.  You can run the tests locally and generate a coverage report with:
```
pytest --cov=fetchAZA --cov-report term-missing tests/
```

