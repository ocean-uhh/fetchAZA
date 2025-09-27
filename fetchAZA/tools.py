import numpy as np
import xarray as xr
from fetchAZA import readers, utilities, timetools
import logging

_log = logging.getLogger(__name__)


# Various conversions from the key to units_name with the multiplicative conversion factor
unit_conversion = {
    "cm/s": {"units_name": "m/s", "factor": 0.01},
    "cm s-1": {"units_name": "m s-1", "factor": 0.01},
    "m/s": {"units_name": "cm/s", "factor": 100},
    "m s-1": {"units_name": "cm s-1", "factor": 100},
    "S/m": {"units_name": "mS/cm", "factor": 0.1},
    "S m-1": {"units_name": "mS cm-1", "factor": 0.1},
    "mS/cm": {"units_name": "S/m", "factor": 10},
    "mS cm-1": {"units_name": "S m-1", "factor": 10},
    "dbar": {"units_name": "Pa", "factor": 10000},
    "kPa": {"units_name": "dbar", "factor": 0.1},
    "kPa s-1": {"units_name": "dbar s-1", "factor": 0.1},
    "Pa": {"units_name": "dbar", "factor": 0.0001},
    "deg": {"units_name": "degrees", "factor": 1},
    "dbar": {"units_name": "kPa", "factor": 10},
    "Deg C": {"units_name": "Celcius", "factor": 1},
    "degrees_Celcius": {"units_name": "Celsius", "factor": 1},
    "degrees_Celsius": {"units_name": "Celsius", "factor": 1},
    "Celsius": {"units_name": "degrees_Celsius", "factor": 1},
    "m": {"units_name": "cm", "factor": 100},
    "m": {"units_name": "km", "factor": 0.001},
    "cm": {"units_name": "m", "factor": 0.01},
    "km": {"units_name": "m", "factor": 1000},
    "g m-3": {"units_name": "kg m-3", "factor": 0.001},
    "kg m-3": {"units_name": "g m-3", "factor": 1000},
}

# Specify the preferred units, and it will convert if the conversion is available in unit_conversion
preferred_units = ["m s-1", "dbar", "S m-1", "dbar s-1"]

# String formats for units.  The key is the original, the value is the desired format
unit_str_format = {
    "m/s": "m s-1",
    "hex": "hexadecimal",
    "s": "seconds",
    "cm/s": "cm s-1",
    "S/m": "S m-1",
    "meters": "m",
    "Deg C": "Celsius",
    "kPa/Sec": "kPa s-1",
    "degrees_Celsius": "Celsius",
    "g/m^3": "g m-3",
}


def reformat_units_var(ds, var_name, unit_format=unit_str_format):
    """
    Renames units in the dataset based on the provided dictionary for OG1.

    Parameters
    ----------
    ds (xarray.Dataset): The input dataset containing variables with units to be renamed.
    unit_format (dict): A dictionary mapping old unit strings to new formatted unit strings.

    Returns
    -------
    xarray.Dataset: The dataset with renamed units.
    """
    old_unit = ds[var_name].attrs.get("units")
    if old_unit in unit_format:
        new_unit = unit_format[old_unit]
    else:
        new_unit = old_unit
    return new_unit


def convert_units(ds, key=""):
    """
    Convert the units of variables in an xarray Dataset to preferred units.  This is useful, for instance, to convert cm/s to m/s.

    Parameters
    ----------
    ds (xarray.Dataset): The dataset containing variables to convert.

    Returns
    -------
    xarray.Dataset: The dataset with converted units.
    """

    for var in ds.variables:
        var_values = ds[var].values
        orig_unit = ds[var].attrs.get("units")
        interim_unit = reformat_units_var(ds, var)
        if var in readers.vocab_attrs:
            if "units" in readers.vocab_attrs[var]:
                new_unit = readers.vocab_attrs[var].get("units")
                if orig_unit != new_unit:
                    var_values, new_unit, errstr = convert_units_var(
                        var_values, orig_unit, new_unit
                    )
                    ds[var].values = var_values
                    ds[var].attrs["units"] = new_unit
                new_unit = reformat_units_var(ds, var)
                ds[var].attrs["units"] = new_unit

                if orig_unit != new_unit:
                    _log.info(
                        f"{var}".ljust(25)
                        + f"Converted UNITS {orig_unit} --> {new_unit}"
                    )

    return ds


def convert_units_var(
    var_values, current_unit, new_unit, unit_conversion=unit_conversion
):
    """
    Convert the units of variables in an xarray Dataset to preferred units.  This is useful, for instance, to convert cm/s to m/s.

    Parameters
    ----------
    ds (xarray.Dataset): The dataset containing variables to convert.
    preferred_units (list): A list of strings representing the preferred units.
    unit_conversion (dict): A dictionary mapping current units to conversion information.
    Each key is a unit string, and each value is a dictionary with:
        - 'factor': The factor to multiply the variable by to convert it.
        - 'units_name': The new unit name after conversion.

    Returns
    -------
    xarray.Dataset: The dataset with converted units.
    """
    if (
        current_unit in unit_conversion
        and new_unit in unit_conversion[current_unit]["units_name"]
    ):
        errstr = ""
        conversion_factor = unit_conversion[current_unit]["factor"]
        new_values = var_values * conversion_factor
    elif current_unit == new_unit:
        errstr = ""
        new_values = var_values
    else:
        errstr = f"No conversion info for {current_unit} to {new_unit}"
        new_values = var_values
        new_unit = current_unit
    return new_values, new_unit, errstr


def process_datasets(
    data_path,
    file_root,
    deploy_date,
    recovery_date,
    keys=["KLR", "INC", "DQZ", "TMP", "PIES", "AZAseq"],
):
    """
    Processes datasets by loading, transforming, and combining data from multiple sources.

    Steps performed:
    1. Load netCDF datasets based on provided keys.
    2. Convert units and adjust time formats.
    3. Assign sampling time for the AZA sequence dataset.
    4. Filter datasets to the deployment period.
    5. Reindex datasets on time.
    6. Rename variables in datasets using predefined mappings.
    7. Add dataset-specific attributes to variables.
    8. Combine selected datasets into a single dataset.
    9. Interpolate the combined dataset to an evenly spaced time grid.
    10. Clean and organize dataset attributes and variables.
    11. Process the AZA sequence dataset, including renaming attributes and cleaning variables.

    External utilities and tools such as `readers`, `timetools`, and `utilities` are used for specific operations.
    Attributes with certain prefixes or exact matches are removed from the combined dataset.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the dataset files.
    file_root : str
        Root name of the files to be processed.
    deploy_date : str or datetime
        Deployment start date for filtering the datasets.
    recovery_date : str or datetime
        Recovery end date for filtering the datasets.
    keys : list of str, optional
        List of dataset keys to process. Defaults to ['KLR', 'INC', 'DQZ', 'TMP', 'PIES', 'AZAseq'].

    Returns
    -------
    tuple
        - ds_pressure (xarray.Dataset): Combined and interpolated dataset containing pressure-related data.
        - ds_AZA (xarray.Dataset): Processed AZA sequence dataset with cleaned attributes.
    """

    # Load netCDF interim files
    datasets = readers.load_netcdf_datasets(data_path, file_root, keys)
    _log.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    for key, ds in datasets.items():
        record_time_len = len(ds.get("RECORD_TIME", [])) if "RECORD_TIME" in ds else 0
        _log.info(
            f"Dataset {key}: {len(ds.variables)} variables, RECORD_TIME length: {record_time_len}"
        )
        if record_time_len == 0:
            _log.warning(f"Dataset {key} has no RECORD_TIME data!")

    # Change units to preferred
    for key in datasets:
        ds = datasets[key]
        ds = convert_units(ds)
        ds = timetools.convert_seconds_to_float(ds)

    # Assign sampling time for AZA sequence
    if "AZAseq" in datasets:
        _log.info(
            f"AZAseq dataset found with {len(datasets['AZAseq']['RECORD_TIME'])} records"
        )
        _log.info(
            f"AZAseq pattern: {datasets['AZAseq'].attrs.get('pattern', 'No pattern attribute')}"
        )
        datasets["AZAseq"] = timetools.assign_sample_time(
            datasets["AZAseq"],
            pattern=datasets["AZAseq"].attrs["pattern"],
            adjust_time=15,
        )
        _log.info(
            f"AZAseq after time assignment: {len(datasets['AZAseq']['RECORD_TIME'])} records"
        )
    else:
        _log.warning("No AZAseq dataset found in loaded datasets")

    # Cut dataset to deployment period
    datasets = timetools.cut_to_deployment(datasets, deploy_date, recovery_date)
    _log.info("After cutting to deployment period:")
    for key, ds in datasets.items():
        record_time_len = len(ds.get("RECORD_TIME", [])) if "RECORD_TIME" in ds else 0
        _log.info(f"Dataset {key}: RECORD_TIME length: {record_time_len}")
        if record_time_len == 0:
            _log.warning(
                f"Dataset {key} has no data after deployment period filtering!"
            )
        if key == "AZAseq":
            _log.info(f"AZAseq after deployment filtering: {record_time_len} records")
            if record_time_len > 0:
                _log.info(
                    f"AZAseq time range: {ds['RECORD_TIME'].min().values} to {ds['RECORD_TIME'].max().values}"
                )
            else:
                _log.warning(
                    "AZAseq dataset is empty after deployment period filtering!"
                )

    # Assign attributes to dataset variables using readers.vocab_attrs
    for key in datasets:
        ds = datasets[key]
        for var in ds.variables:
            if var in readers.vocab_attrs:
                for attr_name, attr_value in readers.vocab_attrs[var].items():
                    if attr_name not in ds[var].attrs:
                        ds[var].attrs[attr_name] = attr_value
                    else:
                        _log.warning(
                            f"Variable '{var}' already has attribute '{attr_name}'. Not overwriting."
                        )
            else:
                _log.warning(f"Variable '{var}' not found in vocab_attrs")
        datasets[key] = ds

    # Reindex datasets on time
    for key in datasets:
        ds = datasets[key]
        ds = timetools.reindex_on_time(ds)
        datasets[key] = ds

    time_var = "RECORD_TIME"
    keys_to_combine = ["KLR", "INC", "DQZ", "TMP", "PIES"]

    # Filter keys_to_combine to only include datasets that actually exist
    available_keys = [key for key in keys_to_combine if key in datasets]
    if len(available_keys) < len(keys_to_combine):
        missing_keys = [key for key in keys_to_combine if key not in datasets]
        _log.warning(
            f"Some expected datasets are missing: {missing_keys}. Proceeding with available datasets: {available_keys}"
        )
        keys_to_combine = available_keys

    vars_to_rename = {
        "KLR": {
            "PRESSURE": "PRESSURE_KLR",
            "TEMPERATURE": "TEMPERATURE_KLR",
            "Serial_Number": "Serial_Number_KLR",
            "Index": "Index_KLR",
        },
        "DQZ": {
            "PRESSURE": "PRESSURE_DQZ",
            "TEMPERATURE": "TEMPERATURE_DQZ",
            "Serial_Number": "Serial_Number_DQZ",
            "Index": "Index_DQZ",
        },
        "INC": {"Serial_Number": "Serial_Number_INC", "Index": "Index_INC"},
        "TMP": {
            "TEMPERATURE_DEG_C": "TEMPERATURE",
            "Serial_Number": "Serial_Number_TMP",
            "Index": "Index_TMP",
        },
        "PIES": {
            "PRESSURE": "PRESSURE_PIES",
            "Serial_Number": "Serial_Number_PIES",
            "Index": "Index_PIES",
        },
    }

    # Rename variables in datasets
    for key, rename_dict in vars_to_rename.items():
        if key in datasets:
            ds = datasets[key]
            for old_name, new_name in rename_dict.items():
                if old_name in ds:
                    ds = ds.rename({old_name: new_name})
            datasets[key] = ds
        else:
            print(f"Dataset {key} not found in datasets")

    # For all variables in datasets, add as attribute the key
    for key in datasets:
        ds = datasets[key]
        for var in ds.variables:
            if var not in ["RECORD_TIME", "TIME"]:
                ds[var].attrs["Logging Event"] = key
        datasets[key] = ds

    # Collect all attributes from the datasets
    all_attributes = {}
    # Combine datasets
    combined_datasets = {}
    for key in datasets:
        ds = datasets[key]
        if key in keys_to_combine:
            combined_datasets[key] = ds
            all_attributes[key] = ds.attrs
        else:
            print(f"Dataset {key} not included in combined datasets")

    # Combine datasets into one
    _log.info(
        f"Combining {len(combined_datasets)} datasets: {list(combined_datasets.keys())}"
    )
    for key, ds in combined_datasets.items():
        _log.info(
            f"Dataset {key}: {len(ds.variables)} variables, RECORD_TIME length: {len(ds.get('RECORD_TIME', []))}"
        )

    combined_dataset = xr.merge(combined_datasets.values(), compat="override")
    _log.info(
        f"Combined dataset: {len(combined_dataset.variables)} variables, RECORD_TIME length: {len(combined_dataset.get('RECORD_TIME', []))}"
    )
    # Assign attributes to combined_dataset from all_attributes
    for key, attrs in all_attributes.items():
        for attr_name, attr_value in attrs.items():
            attr_name = attr_name.replace(" ", "_")
            combined_dataset.attrs[f"{attr_name}_{key}"] = attr_value
    # Remove attributes that start with attr_skip = ['UID_']
    attr_skip = ["UID_", "Sensor"]
    for attr_name in list(combined_dataset.attrs.keys()):
        if any(attr_name.startswith(skip) for skip in attr_skip):
            del combined_dataset.attrs[attr_name]
    # Remove attributes that EXACTLY match attr_skip = ['Calculation Version']
    attr_skip = ["Calculation Version", "Index"]
    for attr_name in list(combined_dataset.attrs.keys()):
        if any(attr_name == skip for skip in attr_skip):
            del combined_dataset.attrs[attr_name]
    _, med_sr = timetools.calculate_sample_rate(combined_dataset)

    # Check if combined_dataset has any data
    if len(combined_dataset["RECORD_TIME"]) == 0:
        _log.error("Combined dataset is empty - no RECORD_TIME data found")
        raise ValueError(
            "Combined dataset is empty. No data was loaded from the intermediate netCDF files. Check that the CSV file contains valid data for the specified keys."
        )

    # Create an evenly spaced time grid based on combined_dataset['RECORD_TIME']
    time_start = combined_dataset["RECORD_TIME"].min().values
    time_end = combined_dataset["RECORD_TIME"].max().values
    time_grid = xr.DataArray(
        np.arange(
            time_start,
            time_end,
            med_sr * np.timedelta64(1, "s"),
            dtype="datetime64[ns]",
        ),
        dims="TIME",
        name="TIME",
    )

    # Create a new dataset with the time grid, linearly interpolating data from combined_dataset['RECORD_TIME']
    _log.info(f"Interpolating dataset to {len(time_grid)} time points")
    ds_pressure = combined_dataset.interp(RECORD_TIME=time_grid, method="linear")
    ds_pressure["RECORD_TIME"] = time_grid
    ds_pressure["RECORD_TIME"].values
    ds_pressure["RECORD_TIME"].attrs["description"] = "Interpolated record time"

    # Remove PRESSURE_PIES from the variable ds_pressure
    if "PRESSURE_PIES" in ds_pressure.variables:
        ds_pressure = ds_pressure.drop_vars("PRESSURE_PIES")
        ds_pressure["PRESSURE_DQZ"].attrs[
            "description"
        ] = "Interpolated pressure from DQZ, identical to pressure in PIES logging events"

    # Sort the dataset variables
    ds_pressure = ds_pressure[sorted(ds_pressure.data_vars)]
    # Sort the dataset attributes
    sorted_attrs = dict(sorted(ds_pressure.attrs.items()))
    ds_pressure.attrs = sorted_attrs
    ds_pressure = utilities.set_best_dtype(ds_pressure)
    ds_pressure = ds_pressure.drop_vars("index")

    ds_AZA = datasets["AZAseq"]
    ds_AZA = ds_AZA[sorted(ds_AZA.data_vars)]
    # Replace spaces with underscores in attributes of ds_AZA
    for attr_name in list(ds_AZA.attrs.keys()):
        new_attr_name = attr_name.replace(" ", "_")
        ds_AZA.attrs[new_attr_name] = ds_AZA.attrs.pop(attr_name)
    rename_attr = {
        "Transfer_SN": "Serial_Number_Transfer",
        "Ambient_SN": "Serial_Number_Ambient",
        "Serial Number": "Serial_Number",
    }
    for old_name, new_name in rename_attr.items():
        if old_name in ds_AZA.attrs:
            ds_AZA.attrs[new_name] = ds_AZA.attrs.pop(old_name)
    if "SERIAL_NUMBER" in ds_AZA.variables:
        snvalue = ds_AZA["SERIAL_NUMBER"].mean().item()
        if np.isnan(snvalue):
            _log.warning(
                "SERIAL_NUMBER contains NaN values, setting Serial_Number_Low to 0"
            )
            ds_AZA.attrs["Serial_Number_Low"] = 0
        else:
            ds_AZA.attrs["Serial_Number_Low"] = int(snvalue)
        ds_AZA = ds_AZA.drop_vars("SERIAL_NUMBER")

    ds_AZA = utilities.set_best_dtype(ds_AZA)

    # Rename 'description' attribute to 'comment' if 'comment' does not exist
    for var in ds_pressure.variables:
        if (
            "description" in ds_pressure[var].attrs
            and "comment" not in ds_pressure[var].attrs
        ):
            ds_pressure[var].attrs["comment"] = ds_pressure[var].attrs.pop(
                "description"
            )

    for var in ds_AZA.variables:
        if "description" in ds_AZA[var].attrs and "comment" not in ds_AZA[var].attrs:
            ds_AZA[var].attrs["comment"] = ds_AZA[var].attrs.pop("description")

    ds_pressure = ds_pressure.set_index(TIME="RECORD_TIME")
    return ds_pressure, ds_AZA
