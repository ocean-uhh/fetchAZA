from fetchAZA import tools, readers, writers

import os
import datetime
import logging

_log = logging.getLogger(__name__)


def convertAZA(
    data_path,
    fn,
    STN="sample",
    deploy_date="2000-01-01",
    recovery_date="2099-01-01",
    latitude="0",
    longitude="0",
    water_depth="0",
    keys=["DQZ", "PIES", "INC", "TMP", "KLR"],
    cleanup=True,
    overwrite=None,
):
    """
    Processes and converts AZA data from CSV to netCDF format.

    Parameters
    ----------
    data_path : str
        Path to the data directory.
    fn : str
        Filename of the input CSV file.
    STN : str
        Station identifier.
    deploy_date : str
        Deployment date in 'YYYY-MM-DD' format.
    recovery_date : str
        Recovery date in 'YYYY-MM-DD' format.
    latitude : float
        Latitude of the station.
    longitude : float
        Longitude of the station.
    water_depth : float
        Water depth at the station.
    keys : list of str
        Keys for intermediate files to process.
    cleanup : bool
        Whether to delete intermediate files after processing.
    overwrite : bool, optional
        If True, overwrite existing files. If False, skip existing files.
        If None, prompt the user for input.

    Returns
    -------
    tuple
        ds_pressure and ds_AZA datasets.
    """
    # Process filename
    file_path = os.path.join(data_path, fn)
    file_root = fn.split(".")[0]
    platform_id = file_root
    today = datetime.datetime.now()
    start_time = today.strftime("%Y%m%dT%H")

    # Create a log file
    log_file = os.path.join(data_path, f"{platform_id}_{start_time}_read.log")
    logf_with_path = os.path.join(data_path, log_file)
    # Create the log file
    logging.basicConfig(
        filename=logf_with_path,
        encoding="utf-8",
        format="%(asctime)s %(levelname)-8s %(funcName)s %(message)s",
        filemode="w",  # 'w' to overwrite, 'a' to append
        level=logging.INFO,
        datefmt="%Y%m%dT%H%M%S",
        force=True,
    )
    _log.info("Reading AZA from CSV to netCDF")
    _log.info("Processing data from: %s", file_path)

    # Convert the data
    datasets = readers.read_csv_to_xarray(file_path)
    # Save intermediate files
    writers.save_datasets(datasets, file_path, overwrite=overwrite)
    # Process the data
    ds_pressure, ds_AZA = tools.process_datasets(
        data_path, file_root, deploy_date, recovery_date
    )
    # Save the datasets
    # Add attributes to ds_pressure
    ds_pressure.attrs.update(
        {
            "Station": STN,
            "Latitude": latitude,
            "Longitude": longitude,
            "Water_Depth": water_depth,
            "Start_Time": deploy_date,
            "End_Time": recovery_date,
        }
    )

    # Add attributes to ds_AZA
    ds_AZA.attrs.update(
        {
            "Station": STN,
            "Latitude": latitude,
            "Longitude": longitude,
            "Water_Depth": water_depth,
            "Start_Time": deploy_date,
            "End_Time": recovery_date,
        }
    )

    output_file = os.path.join(
        data_path, f"{STN}_{deploy_date.replace('-','').replace('/','')}_use.nc"
    )
    writers.save_dataset(ds_pressure, output_file, overwrite=overwrite)

    output_file = os.path.join(
        data_path, f"{STN}_{deploy_date.replace('-','').replace('/','')}_AZAseq.nc"
    )
    writers.save_dataset(ds_AZA, output_file, overwrite=overwrite)

    if cleanup:
        # Delete the intermediate files
        _log.info("Deleting intermediate files")
        writers.delete_netcdf_datasets(data_path, file_root, keys)

    return ds_pressure, ds_AZA
