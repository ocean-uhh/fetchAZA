import numpy as np
from numbers import Number
import os
import logging
import glob

_log = logging.getLogger(__name__)


def delete_netcdf_datasets(
    data_path, file_root, keys=["KLR", "DQZ", "PIES", "TMP", "INC"]
):
    """
    Delete netCDF files matching the given file_root and optional keys from the specified data_path.

    Parameters
    ----------
    data_path : str
        The directory path where the netCDF files are located.
    file_root : str
        The root name of the files to match.
    keys : list of str, optional
        A list of keys to filter the files. Only files containing these keys in their names will be deleted.
        If None, all matching files will be deleted.

    Returns
    -------
    int
        The number of files successfully deleted.
    """
    # Find matching netCDF files
    search_dir = os.path.join(data_path, f"{file_root}*.nc")
    matching_files = glob.glob(search_dir)

    # Filter files based on keys if provided
    if keys is not None:
        matching_files = [
            file
            for file in matching_files
            if any(f"{file_root}_{key}" in os.path.basename(file) for key in keys)
        ]

    # Delete the matching netCDF files
    deleted_count = 0
    for file in matching_files:
        try:
            print(f"Deleting file: {file}")
            os.remove(file)
            _log.info(f"Deleted file: {file}")
            deleted_count += 1
        except Exception as e:
            _log.error(f"Failed to delete file: {file}. Error: {e}")

    _log.info(f"Deleted {deleted_count} files matching '{file_root}' with keys {keys}.")

    return deleted_count


def save_dataset(ds, output_file="../data/test.nc", overwrite=None):
    """
    Attempts to save the dataset to a NetCDF file. If a TypeError occurs due to invalid attribute values,
    it converts the invalid attributes to strings and retries the save operation.

    Parameters
    ----------
    ds (xarray.Dataset): The dataset to be saved.
    output_file (str): The path to the output NetCDF file. Defaults to 'test.nc'.
    overwrite (bool, optional): If True, overwrite existing files. If False, skip existing files.
                              If None, prompt the user for input.

    Returns
    -------
    bool: True if the dataset was saved successfully, False otherwise.

    Note
    ----
    Based on: https://github.com/pydata/xarray/issues/3743
    """
    # Check if file exists and handle accordingly
    if os.path.exists(output_file):
        if overwrite is None:
            response = (
                input(f"File {output_file} already exists. Overwrite? (y/n): ")
                .lower()
                .strip()
            )
            overwrite = response in ("y", "yes")

        if overwrite:
            try:
                os.remove(output_file)
                _log.info(f"Deleted existing file: {output_file}")
            except Exception as e:
                _log.error(f"Failed to delete existing file {output_file}: {e}")
                return False
        else:
            _log.info(f"Skipping save - file already exists: {output_file}")
            return False

    valid_types = (str, int, float, np.float32, np.float64, np.int32, np.int64)
    # More general
    valid_types = (str, Number, np.ndarray, np.number, list, tuple)

    # Check for empty strings
    encoding = {}
    for var in ds.variables:
        if ds[var].dtype == "timedelta64[ns]":
            _log.warning(
                f"Variable '{var}' has dtype 'timedelta64[ns]'. Converting to float seconds."
            )
            ds[var] = ds[var].astype("timedelta64[s]").astype(float)
        if ds[var].dtype == "datetime64[ns]":
            ds[var].encoding["units"] = "seconds since 1970-01-01 00:00:00"
            _log.info(
                f"Variable '{var}' has dtype 'datetime64[ns]', and length {str(len(ds[var]))}. Encoding as {ds[var].encoding['units']}."
            )
            if "units" in ds[var].attrs:
                _log.warning(
                    f"Variable '{var}' has attribute 'units'={ds[var].attrs['units']}. Removing attribute"
                )
                del ds[var].attrs["units"]

    try:
        ds.to_netcdf(output_file)
        _log.info(f"Successfully saved dataset to: {output_file}")
        return True
    except TypeError as e:
        print(e.__class__.__name__, e)
        for varname, variable in ds.variables.items():
            for k, v in variable.attrs.items():
                if not isinstance(v, valid_types) or isinstance(v, bool):
                    _log.warning(
                        f"variable '{varname}': Converting attribute '{k}' with value '{v}' to string."
                    )
                    variable.attrs[k] = str(v)
        try:
            ds.to_netcdf(output_file)  # , format='NETCDF4_CLASSIC'
            _log.info(f"Successfully saved dataset to: {output_file}")
            return True
        except Exception as e:
            _log.error("Failed to save dataset:", e)
            datetime_vars = [
                var for var in ds.variables if ds[var].dtype == "datetime64[ns]"
            ]
            _log.info("Variables with dtype datetime64[ns]:", datetime_vars)
            float_attrs = [
                attr for attr in ds.attrs if isinstance(ds.attrs[attr], float)
            ]
            _log.info("Attributes with dtype float64:", float_attrs)
            return False


def save_datasets(data_sets_new, input_fn, overwrite=None):
    """
    Save multiple datasets to NetCDF files with filenames derived from the input filename.

    Parameters
    ----------
    data_sets_new : dict
        A dictionary where keys are dataset identifiers (e.g., strings) and values are the corresponding datasets to be saved.
    input_fn : str
        The input filename used as a base to generate output filenames.
    overwrite : bool, optional
        If True, overwrite existing files. If False, skip existing files.
        If None, prompt the user for input.

    Notes
    -----
    - Iterates through the `data_sets_new` dictionary.
    - For each dataset, constructs an output filename by appending the dataset key to the base name of `input_fn` and adding the `.nc` extension.
    - Calls `save_dataset` to save each dataset to the corresponding output file.
    - Prints the key of each dataset being processed.
    - If saving a dataset fails, prints an error message indicating the failure.

    Assumes the existence of a `save_dataset` function that handles the actual saving of datasets to files.
    """

    for key, ds in data_sets_new.items():
        _log.info(f"------------ Saving dataset for key: {key} ------------")
        output_file = os.path.splitext(input_fn)[0] + f"_{key}.nc"

        if not save_dataset(ds, output_file, overwrite=overwrite):
            _log.error(f"Failed to save dataset for key: {key}")
            print(f"Failed to save dataset for key: {key}")
