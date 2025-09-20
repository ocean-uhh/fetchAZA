import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_record_type(key):
    """
    Retrieve the record type description for a given key or a list of keys.

    Parameters
    ----------
    key : str or list
        The key or list of keys to look up in the `record_types` dictionary.

    Returns
    -------
    str or None
        The record type description if the key is found, otherwise None.
    """
    if isinstance(key, str):
        return record_types.get(key, None)
    elif isinstance(key, list):
        for k in key:
            print(f"{k}: {record_types.get(k, 'Unknown key')}")
    else:
        raise TypeError("Input must be a string or a list of strings")


def plot_AZA_pressure(ds_AZA, variables, demean=False):
    """
    Plot a comparison of pressure-related variables within a single dataset over a specified time range.

    Parameters
    ----------
    ds_AZA : xarray.Dataset
        The dataset containing the variables to plot.
    variables : list
        List of variable names (strings) to plot from the dataset.
    start_date : str, optional
        Start date for the time range to filter the data (format: 'YYYY-MM-DD'). Defaults to None.
    end_date : str, optional
        End date for the time range to filter the data (format: 'YYYY-MM-DD'). Defaults to None.
    demean : bool, optional
        If True, demean the data by subtracting the mean of each variable before plotting. Defaults to False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plots.
    axs : numpy.ndarray
        Array of axes objects corresponding to the subplots.

    Notes
    -----
    - If the number of data points in a variable is less than or equal to 20, the data points are plotted with symbols.
    - The function automatically adjusts the number of columns in the subplot grid based on the number of variables.
    - Unused subplot axes are removed for cleaner visualization.
    - The y-axis is inverted for all plots.
    - Units for each variable are extracted from the dataset attributes and displayed on the y-axis label.
    """

    time_var = next(
        (var for var in ["TIME", "SAMPLE_TIME", "RECORD_TIME"] if var in ds_AZA.coords),
        None,
    )
    if time_var is None:
        raise ValueError("No valid time coordinate found in the dataset.")

    if demean:
        ds_AZA = ds_AZA - ds_AZA.mean()

    # Check the number of data points
    num_data_points = ds_AZA.sizes[time_var]
    plot_with_symbol = num_data_points <= 20

    # Determine the number of columns needed
    num_vars = len(variables)
    num_cols = (num_vars + 2) // 3  # Default to 3 rows

    # Create the plot
    fig, axs = plt.subplots(3, num_cols, figsize=(9, 2 * 3), sharex=True)

    axs = axs.flatten()

    for i, var in enumerate(variables):
        axs[i].plot(ds_AZA[time_var], ds_AZA[var], label=f"{var}", color="blue")
        if plot_with_symbol:
            axs[i].plot(
                ds_AZA[time_var], ds_AZA[var], "o", label=f"{var}", color="blue"
            )

        if demean:
            axs[i].set_title(f"{var} (demeaned)")
        else:
            axs[i].set_title(f"{var}")
        units = ds_AZA[var].attrs.get("units", "units")
        axs[i].set_ylabel(f"{var} ({units})")
        # axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[i].invert_yaxis()

    # Remove unused axes
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()
    return fig, axs


def compare_pressure(ds_pressure, keys, STN):
    """
    Compare pressure data between multiple time series in a dataset over a specified time range.

    This function interpolates the pressure data from the specified time series onto a common time axis,
    calculates the differences between them, and plots the pressure data and their differences.

    Parameters
    ----------
    ds_pressure : xarray.Dataset
        The dataset containing pressure data. It must have a time coordinate ('TIME', 'SAMPLE_TIME', or 'RECORD_TIME')
        and the specified variable for each key in `keys`.
    keys : list of str
        A list of strings identifying the pressure time series to compare in the dataset.
    variable : str
        The name of the variable in the dataset to compare (e.g., 'PRESSURE').
    deploy_date : str, optional
        The deployment date in 'YYYY-MM-DD' format. Data before this date will be excluded.
        Default is '2000-01-01'.
    recovery_date : str, optional
        The recovery date in 'YYYY-MM-DD' format. Data after this date will be excluded.
        Default is '2099-01-01'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    axs : numpy.ndarray
        An array of axes objects corresponding to the subplots.
    """
    # Determine the time variable in the dataset
    if "TIME" in ds_pressure.coords:
        time_var = "TIME"
    elif "SAMPLE_TIME" in ds_pressure.coords:
        time_var = "SAMPLE_TIME"
    elif "RECORD_TIME" in ds_pressure.coords:
        time_var = "RECORD_TIME"
    else:
        raise ValueError("No valid time coordinate found in the dataset.")

    # Plot the pressure data and their differences
    num_keys = len(keys)
    fig, axs = plt.subplots(num_keys + 1, 1, figsize=(14, 7.5), sharex=True)

    # Plot each key's pressure data
    for i, key in enumerate(keys):
        axs[i].plot(
            ds_pressure[time_var],
            ds_pressure[key],
            label=f"{key} Pressure",
            color=f"C{i}",
        )
        axs[i].set_title(f"{key} Pressure over Time")
        axs[i].set_ylabel("Pressure (dbar)")
        axs[i].legend()
        axs[i].invert_yaxis()

    # Calculate and plot the differences between the first key and the others
    reference_key = keys[0]
    for key in keys[1:]:
        pressure_difference = ds_pressure[reference_key] - ds_pressure[key]
        axs[-1].plot(
            ds_pressure[time_var],
            pressure_difference,
            label=f"Difference ({reference_key} - {key})",
        )

    axs[-1].set_title("Pressure Differences over Time")
    axs[-1].set_xlabel("Time")
    axs[-1].set_ylabel("Pressure Difference (dbar)")
    axs[-1].legend()

    # Add station info to the top right of the figure
    fig.text(
        0.99, 0.99, f"STN: {STN}", ha="right", fontsize=12, verticalalignment="top"
    )
    return fig, axs


# Example usage
def plot_histograms(ds, STN):
    """
    Plot histograms of all data variables in the dataset.

    This function creates a grid of histograms showing the distribution of values
    for each data variable in the dataset. The histograms are arranged in a grid
    with up to 4 columns, and unused subplot positions are removed.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing variables to plot. All data variables in the dataset
        will have histograms created.
    STN : str
        Station identifier that will be displayed in the top right corner of the figure.

    Returns
    -------
    None
        The function displays the plot but does not return any values.

    Notes
    -----
    - NaN values are automatically excluded from the histograms.
    - Each histogram uses 30 bins by default.
    - The grid layout uses up to 4 columns with the number of rows calculated
      based on the number of variables.
    - Unused subplot axes are removed for cleaner visualization.
    """
    num_vars = len(ds.data_vars)
    ncols = 4
    nrows = (num_vars + ncols - 1) // ncols  # Calculate rows needed for up to 4 columns

    fig, axs = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier iteration

    for i, var in enumerate(ds.data_vars):
        data = ds[var].values.flatten()
        axs[i].hist(data[~np.isnan(data)], bins=30, alpha=0.7, label=f"{var}")
        axs[i].set_title(f"Histogram of {var}")
        axs[i].set_xlabel("Value")
        axs[i].set_ylabel("Frequency")
        # axs[i].legend()

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Add station info to the top right of the figure
    fig.text(
        0.99, 0.99, f"STN: {STN}", ha="right", fontsize=12, verticalalignment="top"
    )

    plt.tight_layout()
    plt.show()


def plot_temperature_variables(ds_pressure, keys, STN, fig=None, axs=None):
    """
    Plot temperature variables from a dataset against time.

    This function creates time series plots of temperature variables, automatically
    determining the appropriate time coordinate and creating subplots for each
    specified variable key.

    Parameters
    ----------
    ds_pressure : xarray.Dataset
        The dataset containing temperature variables. Should have a time coordinate
        such as 'RECORD_TIME' or 'TIME'.
    keys : str or list of str
        A single key (str) or a list of keys to match variable names for plotting.
        Each key will generate a separate subplot.
    STN : str
        Station identifier for annotation on the plots.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot into. If None, a new figure is created.
        Default is None.
    axs : numpy.ndarray or matplotlib.axes._axes.Axes, optional
        Existing axes to plot into. If None, new axes are created.
        Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    axs : numpy.ndarray or list
        The array or list of axes objects corresponding to the subplots.

    Notes
    -----
    - The function prioritizes 'TIME' over 'RECORD_TIME' as the time coordinate.
    - If neither is found, it uses the first coordinate in the dataset.
    - Each key gets its own subplot with shared x-axis for time comparison.
    """
    time_var = (
        "RECORD_TIME"
        if "RECORD_TIME" in ds_pressure.coords
        else list(ds_pressure.coords.keys())[0]
    )
    time_var = (
        "TIME" if "TIME" in ds_pressure.coords else list(ds_pressure.coords.keys())[0]
    )

    if isinstance(keys, str):
        keys = [keys]

    # Create new figure and axes if not provided
    if fig is None or axs is None:
        fig, axs = plt.subplots(len(keys), 1, figsize=(12, 6 * len(keys)), sharex=True)

    # Ensure axs is iterable even if there's only one subplot
    if len(keys) == 1:
        axs = [axs]

    # Define marker styles and line styles to cycle through
    markers = ["o", "v", "s"]
    line_styles = ["-", "--", ":"]

    for i, key in enumerate(keys):
        # Find all variables in ds_pressure that include the key in their names
        temperature_vars = [
            var for var in ds_pressure.data_vars if key.upper() in var.upper()
        ]

        # Plot each temperature variable
        for j, var in enumerate(temperature_vars):
            axs[i].plot(
                ds_pressure[time_var],
                ds_pressure[var],
                label=f"{var} (°C)",
                marker=markers[j % len(markers)],
                linestyle=line_styles[j % len(line_styles)],
                markerfacecolor=(
                    "none" if j > 0 else None
                ),  # Solid for the first, open for subsequent
            )

        # Customize the subplot
        axs[i].set_title(f'Temperature Variables Matching "{key}" in ds_pressure')
        axs[i].set_ylabel("Temperature (°C)")
        axs[i].legend()
        axs[i].grid(True)

    # Customize the shared x-axis
    axs[-1].set_xlabel("Record Time")

    # Add station info to the top right of the figure
    fig.text(
        0.99, 0.99, f"STN: {STN}", ha="right", fontsize=12, verticalalignment="top"
    )

    # Adjust layout
    plt.tight_layout()

    return fig, axs


def plot_temperatures(ds, fig=None, ax=None, ylim=None):
    """
    Plot transfer and ambient temperature time series from a dataset.

    This function creates a time series plot showing both transfer temperature and
    ambient temperature variables from the dataset. The y-axis limits are automatically
    calculated based on the data range (mean ± 2 standard deviations) unless specified.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing temperature variables. Must have 'TRANSFER_TEMPERATURE',
        'AMBIENT_TEMPERATURE', and 'SAMPLE_TIME' variables, and a 'Station' attribute.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot into. If None, a new figure is created.
        Default is None.
    ax : matplotlib.axes._axes.Axes, optional
        Existing axes to plot into. If None, new axes are created.
        Default is None.
    ylim : tuple of float, optional
        Y-axis limits as (lower_limit, upper_limit). If None, limits are automatically
        calculated from the data. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._axes.Axes
        The axes object containing the plot.

    Notes
    -----
    - The function expects specific variable names: 'TRANSFER_TEMPERATURE',
      'AMBIENT_TEMPERATURE', and 'SAMPLE_TIME'.
    - Y-axis limits are auto-calculated as mean ± 2 standard deviations if not provided.
    - Station information is displayed in the bottom right corner of the plot.
    - X-axis labels are rotated 45 degrees for better readability.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    transfer_temp_filtered = ds["TRANSFER_TEMPERATURE"].values
    ambient_temp_filtered = ds["AMBIENT_TEMPERATURE"].values

    if ylim is None:
        combined_temps = np.concatenate([transfer_temp_filtered, ambient_temp_filtered])
        upper_limit = np.nanmean(combined_temps) + 2 * np.nanstd(combined_temps)
        lower_limit = np.nanmean(combined_temps) - 2 * np.nanstd(combined_temps)
        ylim = (lower_limit, upper_limit)

    ax.set_ylim(ylim)

    ax.plot(ds["SAMPLE_TIME"], transfer_temp_filtered, label="Transfer Temperature")
    ax.plot(
        ds["SAMPLE_TIME"],
        ambient_temp_filtered,
        label="Ambient Temperature",
        markerfacecolor="none",
    )

    ax.set_xlabel("Sample Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Transfer and Ambient Temperature over Time")
    ax.legend()
    ax.grid(True)

    ax.tick_params(axis="x", rotation=45)

    ax.text(
        0.99,
        0.05,
        "STN:" + ds.attrs["Station"],
        transform=ax.transAxes,
        fontsize=12,
        horizontalalignment="right",
    )

    if fig is None:
        plt.show()
    return fig, ax


def plot_hist_fig(
    data_sets_new, types_to_plot, deploy_date, recovery_date, data_path, STN
):
    """
    Generate and save histogram plots for specified data types within a given time range.

    This function creates histograms for numeric variables in the provided datasets,
    filtered by the deployment and recovery dates. Each histogram is saved as an image
    file in the specified data path.

    Parameters
    ----------
    data_sets_new : dict
        A dictionary containing xarray datasets, where keys are data types and values
        are the corresponding datasets.
    types_to_plot : list
        A list of data types to generate histograms for.
    deploy_date : str or datetime
        The start date for filtering the datasets.
    recovery_date : str or datetime
        The end date for filtering the datasets.
    data_path : str
        The directory path where the histogram images will be saved. If None, the images
        will not be saved.
    STN : str
        The station identifier to include in the plot titles and saved filenames.

    Returns
    -------
    None

    Notes
    -----
    - The function assumes that the datasets are xarray objects with a 'TIME' dimension.
    - Only numeric variables in the datasets are considered for plotting.
    - If a variable has a timedelta64 dtype, it is converted to seconds before plotting.
    - The function creates subplots with up to 12 histograms per figure.
    - If there are fewer variables than subplot slots, the extra slots are removed.
    - The mean of each variable is marked on the histogram with a red dashed line.
    - The saved filenames follow the format: '{STN}_{data_type}_histograms.png'.

    Example
    -------
    plot_hist_fig(
        data_sets_new=my_datasets,
        types_to_plot=['type1', 'type2'],
        deploy_date='2023-01-01',
        recovery_date='2023-12-31',
        data_path='/path/to/save',
        STN='Station123'
    )
    """

    def plot_histogram(ax, data, var, data_type, units, STN):
        ax.hist(data, bins=50, label=var)
        ax.set_title(f"{var}")
        ax.set_ylabel("Frequency")
        ax.set_xlabel(units)
        #        ax.axvline(np.median(data), color='red', linestyle='--', linewidth=1, label='Median')
        ax.axvline(
            np.mean(data), color="red", linestyle="--", linewidth=1, label="Mean"
        )
        ax.figure.text(
            0.95,
            0.95,
            f"STN: {STN} ({data_type})",
            ha="right",
            fontsize=12,
            verticalalignment="bottom",
        )

    for data_type in types_to_plot:
        if data_type in data_sets_new:
            dataset = data_sets_new[data_type].copy()
            if "SAMPLE_TIME" in dataset:
                time_var = "SAMPLE_TIME"
            elif "RECORD_TIME" in dataset:
                time_var = "RECORD_TIME"
            else:
                raise ValueError(
                    "Dataset must contain either 'RECORD_TIME' or 'SAMPLE_TIME'."
                )
            dataset = dataset.sel({time_var: slice(deploy_date, recovery_date)})
            numeric_vars = [
                var
                for var in dataset.data_vars
                if np.issubdtype(dataset[var].dtype, np.number)
            ]
            print(f"{data_type} has numeric variables {numeric_vars}")
            fig, axs = plt.subplots(3, 4, figsize=(14, 8.5))
            fig.subplots_adjust(hspace=0.4)
            axs = axs.flatten()
            for i, var in enumerate(numeric_vars):
                d1 = dataset[var]
                if np.issubdtype(d1.dtype, np.timedelta64):
                    d1 = d1 / np.timedelta64(1, "s")

                plot_histogram(
                    axs[i], d1, var, data_type, dataset[var].attrs.get("units", ""), STN
                )

                if i % 4 != 0:
                    axs[i].set_ylabel("")

            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            if data_path is not None:
                fig.savefig(
                    os.path.join(data_path, f"{STN}_{data_type}_histograms.png")
                )
                plt.close(fig)
        else:
            print(f"No data found for type {data_type}")


def plot_data_type(
    data_sets_new, data_type, STN, deploy_date, recovery_date, data_path=None
):
    """
    Plot and save figures for a specified data type.

    Parameters
    ----------
    data_type : str
        The type of data to plot.
    STN : str
        The station identifier.
    last_4_digits : str
        The last four digits to include in the filename.

    Returns
    -------
    None

    Notes
    -----
    - This function checks if the specified data type exists in the `data_sets_new` dictionary.
    - If the data type exists, it selects the dataset for the given time range (`deploy_date` to `recovery_date`),
      identifies numeric variables, and creates figures to plot these variables over time.
    - Each figure contains up to 4 variables, and the figures are saved with filenames that
      include the data type, station identifier, and part number.
    """

    xlbl_orig = "Time"
    xlbl = xlbl_orig
    if data_type in data_sets_new:
        dataset = data_sets_new[data_type]
        if "SAMPLE_TIME" in dataset:
            time_var = "SAMPLE_TIME"
        elif "RECORD_TIME" in dataset:
            time_var = "RECORD_TIME"
        else:
            raise ValueError(
                "Dataset must contain either 'RECORD_TIME' or 'SAMPLE_TIME'."
            )
        dataset = dataset.sel({time_var: slice(deploy_date, recovery_date)})
        numeric_vars = [
            var
            for var in dataset.data_vars
            if np.issubdtype(dataset[var].dtype, np.number)
        ]

        # Determine the number of figures needed
        num_figures = (len(numeric_vars) + 5) // 6

        for fig_num in range(num_figures):
            start_idx = fig_num * 6
            end_idx = min(start_idx + 6, len(numeric_vars))
            vars_to_plot = numeric_vars[start_idx:end_idx]

            # Create a figure for each set of variables
            fig, axs = plt.subplots(3, 2, figsize=(14, 8))
            fig.suptitle(
                f"{data_type} Variables Over Time (Part {fig_num + 1})", fontsize=16
            )
            axs = axs.flatten()

            for i, var in enumerate(vars_to_plot):
                if np.issubdtype(dataset[var].dtype, np.timedelta64):
                    dataset[var] = dataset[var] / np.timedelta64(1, "s")
                    dataset[var].attrs["units"] = "s"
                axs[i].plot(dataset[time_var], dataset[var], ".-", label=var)
                axs[i].set_title(f"{var} in {data_type}")
                units = dataset[var].attrs.get("units", "")
                axs[i].set_ylabel(f"{var} ({units})")
                # axs[i].legend()

                # Check the difference between x-ticks
                x_ticks = axs[i].get_xticks()
                if len(x_ticks) > 1:
                    min_year = pd.to_datetime(x_ticks[0], unit="D").year
                    max_year = pd.to_datetime(x_ticks[-1], unit="D").year
                    if min_year == max_year:
                        # Format without %Y and append the year to xlbl
                        xlbl = f"{xlbl_orig} ({min_year})"
                        if (
                            pd.to_datetime(x_ticks[-1]) - pd.to_datetime(x_ticks[0])
                        ).total_seconds() < 86400:
                            # Format as Y/m/d HH if the difference is less than 1 day
                            axs[i].xaxis.set_major_formatter(
                                plt.matplotlib.dates.DateFormatter("%d-%b %Hh")
                            )
                        else:
                            axs[i].xaxis.set_major_formatter(
                                plt.matplotlib.dates.DateFormatter("%m/%d")
                            )

                    elif (
                        pd.to_datetime(x_ticks[1]) - pd.to_datetime(x_ticks[0])
                    ).total_seconds() < 86400:
                        # Format as Y/m/d HH if the difference is less than 1 day
                        axs[i].xaxis.set_major_formatter(
                            plt.matplotlib.dates.DateFormatter("%Y/%m/%d %Hh")
                        )
                    else:
                        xlbl = xlbl_orig
                        # Format as Y/m/d with at least 10 ticks
                        axs[i].xaxis.set_major_formatter(
                            plt.matplotlib.dates.DateFormatter("%Y/%m/%d")
                        )
                        axs[i].xaxis.set_major_locator(plt.MaxNLocator(10))
                axs[i].tick_params(axis="x", rotation=45)

                if i < len(vars_to_plot) - 2:
                    axs[i].set_xticklabels([])

            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            axs[-1].set_xlabel(xlbl)
            axs[-2].set_xlabel(xlbl)

            # Add station info to the top right of the figure
            fig.text(
                0.99,
                0.99,
                f"STN: {STN}",
                ha="right",
                fontsize=12,
                verticalalignment="top",
            )

            # Save the figure using data_type and part number in the filename
            if data_path is not None:
                fig.savefig(
                    os.path.join(
                        data_path, f"{STN}_{data_type}_p{fig_num + 1}_tseries.png"
                    )
                )
                plt.close(fig)
            else:
                plt.show()
    else:
        print(f"No data found for type {data_type}")

    return fig, axs


def plot_all_variables(ds_pressure, STN, data_path=None):
    """
    Plot and save figures for all variables in a given xarray dataset.

    Parameters
    ----------
    ds_pressure : xarray.Dataset
        The dataset containing the variables to plot.
    STN : str
        The station identifier.
    data_path : str, optional
        The directory path where the figures will be saved. If None, the figures will be displayed instead.

    Returns
    -------
    None

    Notes
    -----
    - This function cycles through all numeric variables in the dataset and creates time series plots.
    - Each figure contains up to 6 variables, and the figures are saved with filenames that
      include the station identifier and part number.
    """

    xlbl_orig = "Time"
    xlbl = xlbl_orig
    time_var = next(
        (var for var in ["TIME", "SAMPLE_TIME", "RECORD_TIME"] if var in ds_pressure),
        None,
    )
    if time_var is None:
        raise ValueError(
            "Dataset must contain one of the following time variables: 'TIME', 'SAMPLE_TIME', or 'RECORD_TIME'."
        )

    if time_var not in ds_pressure:
        raise ValueError(
            "Dataset must contain either 'RECORD_TIME' or 'SAMPLE_TIME' as the time variable."
        )

    numeric_vars = [
        var
        for var in ds_pressure.data_vars
        if np.issubdtype(ds_pressure[var].dtype, np.number)
    ]

    # Determine the number of figures needed
    CC = 4
    RR = 3
    num_figures = (len(numeric_vars) + 5) // (RR * CC)

    for fig_num in range(num_figures):
        start_idx = fig_num * RR * CC
        end_idx = min(start_idx + RR * CC, len(numeric_vars))
        vars_to_plot = numeric_vars[start_idx:end_idx]

        # Create a figure for each set of variables
        fig, axs = plt.subplots(RR, CC, figsize=(14, 8))
        fig.suptitle(f"Variables Over Time (Part {fig_num + 1})", fontsize=16)
        axs = axs.flatten()

        for i, var in enumerate(vars_to_plot):
            if np.issubdtype(ds_pressure[var].dtype, np.timedelta64):
                ds_pressure[var] = ds_pressure[var] / np.timedelta64(1, "s")
                ds_pressure[var].attrs["units"] = "s"
            axs[i].plot(ds_pressure[time_var], ds_pressure[var], ".-", label=var)
            axs[i].set_title(f"{var}")
            units = ds_pressure[var].attrs.get("units", "")
            axs[i].set_ylabel(f"{var} ({units})")

            # Check the difference between x-ticks
            x_ticks = axs[i].get_xticks()
            if len(x_ticks) > 1:
                min_year = pd.to_datetime(x_ticks[0], unit="D").year
                max_year = pd.to_datetime(x_ticks[-1], unit="D").year
                if min_year == max_year:
                    xlbl = f"{xlbl_orig} ({min_year})"
                    if (
                        pd.to_datetime(x_ticks[-1]) - pd.to_datetime(x_ticks[0])
                    ).total_seconds() < 86400:
                        axs[i].xaxis.set_major_formatter(
                            plt.matplotlib.dates.DateFormatter("%d-%b %Hh")
                        )
                    else:
                        axs[i].xaxis.set_major_formatter(
                            plt.matplotlib.dates.DateFormatter("%m/%d")
                        )
                elif (
                    pd.to_datetime(x_ticks[1]) - pd.to_datetime(x_ticks[0])
                ).total_seconds() < 86400:
                    axs[i].xaxis.set_major_formatter(
                        plt.matplotlib.dates.DateFormatter("%Y/%m/%d %Hh")
                    )
                else:
                    xlbl = xlbl_orig
                    axs[i].xaxis.set_major_formatter(
                        plt.matplotlib.dates.DateFormatter("%Y/%m/%d")
                    )
                    axs[i].xaxis.set_major_locator(plt.MaxNLocator(10))
            axs[i].tick_params(axis="x", rotation=45)

            if i < len(vars_to_plot) - CC:
                axs[i].set_xticklabels([])

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        axs[-1].set_xlabel(xlbl)
        axs[-2].set_xlabel(xlbl)
        axs[-3].set_xlabel(xlbl)
        axs[-4].set_xlabel(xlbl)

        # Add station info to the top right of the figure
        fig.text(
            0.99, 0.99, f"STN: {STN}", ha="right", fontsize=12, verticalalignment="top"
        )

        plt.tight_layout()

        # Save the figure using part number in the filename
        if data_path is not None:
            fig.savefig(os.path.join(data_path, f"{STN}_p{fig_num + 1}_tseries.png"))
            plt.close(fig)
        else:
            plt.show()
    return fig, axs


# -----------------------------------------------------------------------------------


##------------------------------------------------------------------------------------
## Views of the ds or nc file
##------------------------------------------------------------------------------------
def show_contents(data, content_type="variables"):
    """
    Display the contents of an xarray Dataset or a netCDF file.

    Parameters
    ----------
    data : str or xr.Dataset
        The input data, either a file path to a netCDF file or an xarray Dataset.
    content_type : str
        The type of content to display. Options are:
        - 'variables' or 'vars': Show details about the variables.
        - 'attributes' or 'attrs': Show details about the attributes.
        Default is 'variables'.

    Returns
    -------
    pandas.io.formats.style.Styler or pandas.DataFrame
        A styled DataFrame with details about the variables or attributes.
    """
    if content_type in ["variables", "vars"]:
        if isinstance(data, str):
            return show_variables(data)
        elif isinstance(data, xr.Dataset):
            return show_variables(data)
        else:
            raise TypeError("Input data must be a file path (str) or an xarray Dataset")
    elif content_type in ["attributes", "attrs"]:
        if isinstance(data, str):
            return show_attributes(data)
        elif isinstance(data, xr.Dataset):
            return show_attributes(data)
        else:
            raise TypeError("Attributes can only be shown for netCDF files (str)")
    else:
        raise ValueError(
            "content_type must be either 'variables' (or 'vars') or 'attributes' (or 'attrs')"
        )


def show_variables(data):
    """
    Extract variable information from an xarray Dataset or a netCDF file.

    Parameters
    ----------
    data : str or xr.Dataset
        The input data, either a file path to a netCDF file or an xarray Dataset.

    Returns
    -------
    pandas.io.formats.style.Styler
        A styled DataFrame containing the following columns:
        - dims : str
            The dimension of the variable (or "string" if it is a string type).
        - name : str
            The name of the variable.
        - units : str
            The units of the variable (if available).
        - comment : str
            Any additional comments about the variable (if available).
    """
    from pandas import DataFrame
    from netCDF4 import Dataset

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        dataset = Dataset(data)
        variables = dataset.variables
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        variables = data.variables
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(variables):
        var = variables[key]
        if isinstance(data, str):
            dims = var.dimensions[0] if len(var.dimensions) == 1 else "string"
            units = "" if not hasattr(var, "units") else var.units
            comment = "" if not hasattr(var, "comment") else var.comment
        else:
            dims = var.dims[0] if len(var.dims) == 1 else "string"
            units = var.attrs.get("units", "")
            comment = var.attrs.get("comment", "")

        info[i] = {
            "name": key,
            "dims": dims,
            "units": units,
            "comment": comment,
            "standard_name": var.attrs.get("standard_name", ""),
            "dtype": str(var.dtype) if isinstance(data, str) else str(var.data.dtype),
        }

    vars = DataFrame(info).T

    dim = vars.dims
    dim[dim.str.startswith("str")] = "string"
    vars["dims"] = dim

    vars = (
        vars.sort_values(["dims", "name"])
        .reset_index(drop=True)
        .loc[:, ["dims", "name", "units", "comment", "standard_name", "dtype"]]
        .set_index("name")
        .style
    )

    return vars


def show_attributes(data):
    """
    Extract and display attribute information from an xarray Dataset or a netCDF file.

    Parameters
    ----------
    data : str or xr.Dataset
        The input data, either a file path to a netCDF file or an xarray Dataset.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the following columns:
        - Attribute: The name of the attribute.
        - Value: The value of the attribute.
        - DType: The data type of the attribute value.

    Notes
    -----
    - If the input is a file path, the function reads the attributes from the netCDF file.
    - If the input is an xarray Dataset, the function reads the attributes directly from the Dataset.
    """
    from pandas import DataFrame
    from netCDF4 import Dataset

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        rootgrp = Dataset(data, "r", format="NETCDF4")
        attributes = rootgrp.ncattrs()
        get_attr = lambda key: getattr(rootgrp, key)
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        attributes = data.attrs.keys()
        get_attr = lambda key: data.attrs[key]
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(attributes):
        dtype = type(get_attr(key)).__name__
        info[i] = {"Attribute": key, "Value": get_attr(key), "DType": dtype}

    attrs = DataFrame(info).T

    return attrs


def show_variables_by_dimension(data, dimension_name="trajectory"):
    """
    Extract variable information filtered by a specific dimension from an xarray Dataset or a netCDF file.

    Parameters
    ----------
    data : str or xr.Dataset
        The input data, either a file path to a netCDF file or an xarray Dataset.
    dimension_name : str
        The name of the dimension to filter variables by.

    Returns
    -------
    pandas.io.formats.style.Styler
        A styled DataFrame containing the following columns:
        - dims : str
            The dimension of the variable (or "string" if it is a string type).
        - name : str
            The name of the variable.
        - units : str
            The units of the variable (if available).
        - comment : str
            Any additional comments about the variable (if available).
    """

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        dataset = Dataset(data)
        variables = dataset.variables
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        variables = data.variables
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(variables):
        var = variables[key]
        if isinstance(data, str):
            dims = var.dimensions[0] if len(var.dimensions) == 1 else "string"
            units = "" if not hasattr(var, "units") else var.units
            comment = "" if not hasattr(var, "comment") else var.comment
        else:
            dims = var.dims[0] if len(var.dims) == 1 else "string"
            units = var.attrs.get("units", "")
            comment = var.attrs.get("comment", "")

        if dims == dimension_name:
            info[i] = {
                "name": key,
                "dims": dims,
                "units": units,
                "comment": comment,
            }

    vars = DataFrame(info).T

    dim = vars.dims
    dim[dim.str.startswith("str")] = "string"
    vars["dims"] = dim

    vars = (
        vars.sort_values(["dims", "name"])
        .reset_index(drop=True)
        .loc[:, ["dims", "name", "units", "comment"]]
        .set_index("name")
        .style
    )

    return vars


def plot_age_values(ds_AZA, fig_path, time_var="SAMPLE_TIME", fig=None, ax=None):
    """
    Plots AGE values for AZA and AZS datasets.

    Parameters:
        ds_AZA (xarray.Dataset): The AZA dataset.
        age_values_aza (xarray.DataArray): AGE values for AZA.
        age_values_azs (xarray.DataArray): AGE values for AZS.
        STN (str): Station identifier.
        fig_path (str): Path to save the figure.
        time_var (str): Time variable name. Default is 'SAMPLE_TIME'.
        ax (matplotlib.axes._axes.Axes, optional): Axes to plot into. If None, a new figure and axes are created.

    Returns:
        matplotlib.axes._axes.Axes: The axes containing the plot.
    """
    # Initialize the figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    STN = ds_AZA.attrs["Station"]
    time_var = "SAMPLE_TIME"
    # Extract the AGE values from the AZA and AZS datasets
    age_values_aza = ds_AZA["AGE"].where(
        ds_AZA["SEQUENCE_NUM"].isin([2, 3, 4]), drop=True
    )
    age_values_azs = ds_AZA["AGE"].where(ds_AZA["SEQUENCE_NUM"].isin([1, 5]), drop=True)

    # Convert age values from ns to seconds
    if age_values_aza.dtype == "timedelta64[ns]":
        age_values_aza = age_values_aza / np.timedelta64(1, "s")
        age_values_azs = age_values_azs / np.timedelta64(1, "s")
        units = "seconds"
        age_values_aza.attrs["units"] = units
        age_values_azs.attrs["units"] = units

    # Plot the AGE values for AZA and AZS datasets
    ax.plot(age_values_aza[time_var], age_values_aza, "o", label="AGE AZA")
    ax.plot(
        age_values_azs[time_var],
        age_values_azs,
        "s",
        label="AGE AZS",
        markerfacecolor="none",
    )

    # Set labels, title, and grid
    ax.set_xlabel("Time")
    ax.set_ylabel(f"AGE ({units})")
    ax.set_title("AGE values in AZA sequence")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Fix the axes limits
    ax.set_xlim(
        [age_values_aza[time_var].min().values, age_values_aza[time_var].max().values]
    )
    ax.set_ylim([0, max(age_values_aza.max().values, age_values_azs.max().values) + 50])

    # Add station information
    fig.text(
        0.99,
        0.99,
        f"STN: {STN}",
        ha="right",
        fontsize=12,
        verticalalignment="top",
        transform=ax.transAxes,
    )

    # Rotate the x-axis tick labels by 45 degrees
    ax.tick_params(axis="x", rotation=45)

    # Save the plot if a new figure was created
    if ax is None:
        plt.savefig(os.path.join(fig_path, f"{STN}_AGE_plot_fixed_axes.png"))
        plt.show()

    return fig, ax


def plot_pressure_sequence(ds, key1, STN):
    """
    Plot pressure data for different sequence numbers with comparisons and differences.

    This function creates a three-panel plot showing pressure data for different
    sequence numbers. It compares AZA (sequences 2 and 4) with AZS (sequences 1 and 5)
    data and displays the differences between paired sequences.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing pressure data with 'SEQUENCE_NUM' and 'SAMPLE_TIME'
        coordinates, and the specified pressure variable.
    key1 : str
        The name of the pressure variable to plot (e.g., 'AMBIENT_PRESSURE').
        Must be present in the dataset with 'units', 'standard_name', and
        'long_name' attributes.
    STN : str
        Station identifier for annotation (currently unused in the function).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing all three subplots.
    axs : numpy.ndarray
        Array of three axes objects corresponding to the subplots.

    Notes
    -----
    - The function filters data for sequence numbers 1, 2, 4, and 5.
    - Left panel: Comparison of sequences 1 (AZS) and 2 (AZA).
    - Middle panel: Comparison of sequences 4 (AZA) and 5 (AZS).
    - Right panel: Differences between paired sequences (2-1 and 5-4).
    - AZA sequences are shown in blue solid lines, AZS in red dashed lines.
    - The function expects the variable to have 'units', 'standard_name',
      and 'long_name' attributes.
    """
    # Filter the dataset for SEQUENCE_NUM 1, 2, 3, 4, and 5
    ds_seq1 = ds.where(ds["SEQUENCE_NUM"].isin([1]), drop=True)
    ds_seq2 = ds.where(ds["SEQUENCE_NUM"].isin([2]), drop=True)
    ds_seq3 = ds.where(ds["SEQUENCE_NUM"].isin([3]), drop=True)
    ds_seq4 = ds.where(ds["SEQUENCE_NUM"].isin([4]), drop=True)
    ds_seq5 = ds.where(ds["SEQUENCE_NUM"].isin([5]), drop=True)

    # Plot the ambient pressure and its difference for sequence 1 and 2
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    # Plot the ambient pressure for sequence 1 and 2
    axs[0].plot(ds_seq2["SAMPLE_TIME"], ds_seq2[key1], "b-", label="Seq 2 (AZA)")
    axs[0].plot(ds_seq1["SAMPLE_TIME"], ds_seq1[key1], "r--", label="Seq 1 (AZS)")
    units = ds[key1].attrs["units"]
    standard_name = ds[key1].attrs["standard_name"]
    if "long_name" in ds[key1].attrs:
        long_name = ds[key1].attrs["long_name"]
    else:
        long_name = standard_name
    long_name = ds[key1].attrs["long_name"]
    axs[0].set_ylabel(f"{standard_name} ({units})")
    axs[0].set_title(f"{long_name}\nSequence 1 and 2")
    axs[0].set_xlabel("Time")
    axs[0].legend()
    axs[0].grid(True)

    # Plot the ambient pressure for sequence 4 and 5
    axs[1].plot(ds_seq4["SAMPLE_TIME"], ds_seq4[key1], "b-", label="Seq 4 (AZA)")
    axs[1].plot(ds_seq5["SAMPLE_TIME"], ds_seq5[key1], "r--", label="Seq 5 (AZS)")
    axs[1].set_ylabel(f"{standard_name} ({units})")
    axs[1].set_title(f"{long_name}\nSequence 4 and 5")
    axs[1].legend()
    axs[1].grid(True)

    # Plot the difference in ambient pressure
    diff_2_1 = ds_seq2[key1].values - ds_seq1[key1].values
    diff_5_4 = ds_seq5[key1].values - ds_seq4[key1].values
    axs[2].plot(
        ds_seq2["SAMPLE_TIME"],
        diff_2_1,
        color="black",
        linestyle="--",
        label="Seq 2 - Seq 1",
    )
    axs[2].plot(
        ds_seq5["SAMPLE_TIME"], diff_5_4, color="darkgray", label="Seq 5 - Seq 4"
    )
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel(rf"$\Delta${standard_name} ({units})")
    axs[2].set_title(f"Difference in {long_name}")
    axs[2].legend(loc="best")
    axs[2].grid(True)
    fig.text(
        0.99, 0.99, f"STN: {STN}", ha="right", fontsize=12, verticalalignment="top"
    )

    # Set x-ticks every 2 months
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axs[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axs[0].tick_params(axis="x", rotation=45)
    axs[1].tick_params(axis="x", rotation=45)
    axs[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_pressure_diff_and_age(ds, key1, key2, seqnum):
    """
    Plots the pressure difference, AGE, and temperatures for specified sequences in a dataset.

    Parameters
    -----------
    ds : xarray.Dataset
        The dataset containing the data to be plotted. It should include variables such as
        'SEQUENCE_NUM', 'SAMPLE_TIME', 'AGE', 'AMBIENT_TEMPERATURE', and 'TRANSFER_TEMPERATURE'.
    key1 : str
        The name of the first pressure variable in the dataset.
    key2 : str
        The name of the second pressure variable in the dataset.
    seqnum : list of int
        A list of sequence numbers to filter and plot.

    Returns
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    axs : numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
        The array of axes objects for the subplots.

    Notes
    ------
    - The function calculates the difference between the two pressure variables (`key1` and `key2`)
      and plots it against time for each sequence in `seqnum`.
    - The AGE variable is converted to seconds and plotted against time.
    - Ambient and transfer temperatures are plotted on the third subplot.
    - The x-axis is formatted to display time in 2-month intervals.
    - The temperature subplot's y-axis limits are determined based on the median and standard
      deviation of the combined temperature data.
    - The function assumes that the dataset variables have attributes `standard_name` and `units`
      for proper labeling.
    - A station identifier (`STN`) is expected to be available in the global scope for annotation.

    Raises
    -------
    KeyError:
        If any of the required variables (`key1`, `key2`, 'AGE', 'AMBIENT_TEMPERATURE',
        'TRANSFER_TEMPERATURE', or 'SEQUENCE_NUM') are missing from the dataset.
    """

    # Plot the difference and AGE of sequence 3
    fig, axs = plt.subplots(3, 1, figsize=(8, 9))

    STN = ds.attrs["Station"]
    for sn in seqnum:
        ds_seq3 = ds.where(ds["SEQUENCE_NUM"].isin([sn]), drop=True)

        # Calculate the difference between the transfer and low pressure
        pressure_diff = ds_seq3[key1] - ds_seq3[key2]

        # Plot the pressure difference
        name1 = ds_seq3[key1].attrs["standard_name"]
        name2 = ds_seq3[key2].attrs["standard_name"]
        units = ds_seq3[key1].attrs["units"]
        axs[0].plot(
            ds_seq3["SAMPLE_TIME"],
            pressure_diff,
            label=f"Sequence {str(sn)}): {name1} - {name2}",
        )

        Age1 = ds_seq3["AGE"]
        # Convert Age1 to units of seconds
        Age1_seconds = Age1.dt.total_seconds()
        ds_seq3["AGE"].attrs["units"] = "seconds"
        # Plot the AGE of sequence 3
        axs[1].plot(ds_seq3["SAMPLE_TIME"], Age1_seconds, label="AGE")

    # Plot the temperatures
    axs[2].plot(
        ds_seq3["SAMPLE_TIME"],
        ds_seq3["AMBIENT_TEMPERATURE"],
        label="Ambient Temperature",
        color="magenta",
    )
    axs[2].plot(
        ds_seq3["SAMPLE_TIME"],
        ds_seq3["TRANSFER_TEMPERATURE"],
        label="Transfer Temperature",
        color="green",
    )

    axs[0].set_ylabel(f"Pressure Difference ({units})")
    axs[0].set_title(f"Difference between {name1} and {name2}")
    axs[0].legend()
    axs[0].grid(True)
    # Set x-ticks every 2 months
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axs[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    units = ds_seq3["AGE"].attrs["units"]
    axs[1].set_ylabel(f"AGE ({units})")
    axs[1].set_title(f"AGE for Sequence ({str(seqnum)}")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Temperature (°C)")
    axs[2].set_title("Ambient and Low Temperature")
    axs[2].legend()
    axs[2].grid(True)

    combined_temps = np.concatenate(
        [ds["TRANSFER_TEMPERATURE"].values, ds["AMBIENT_TEMPERATURE"].values]
    )
    upper_limit = np.median(combined_temps) + 2 * np.std(combined_temps)
    lower_limit = np.floor(np.min(combined_temps))

    axs[2].set_ylim(lower_limit, upper_limit)

    fig.text(
        0.99, 0.99, f"STN: {STN}", ha="right", fontsize=12, verticalalignment="top"
    )

    plt.tight_layout()
    plt.show()
    return fig, axs
