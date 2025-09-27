import os
import tempfile
import pandas as pd
import xarray as xr
import numpy as np
import pathlib
import sys
import pytest

script_dir = pathlib.Path(__file__).parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

from fetchAZA import readers


@pytest.fixture
def sample_csv_path():
    """Get the path to the real sample CSV file."""
    return pathlib.Path(__file__).parent.parent / "data" / "sample_data.csv"


def test_parse_column_name():
    assert readers.parse_column_name("Pressure (kPa)") == ("Pressure", "kPa")
    assert readers.parse_column_name("UsedPercentage %") == (
        "UsedPercentage",
        "percent",
    )
    assert readers.parse_column_name("Volts V") == ("Volts", "V")
    assert readers.parse_column_name("UID") == ("UID", None)


def test_parse_header_data_basic():
    lines = [
        "TMP,Record Time,Temperature (Deg C)",
        "# Data",
        "TMP,2025/03/29 12:00:00,22.4",
    ]
    header, data = readers.parse_header_data(lines)
    assert header == ["TMP,Record Time,Temperature (Deg C)"]
    assert data == ["TMP,2025/03/29 12:00:00,22.4"]


def test_process_header_lines():
    headers = ["TMP,Time,Temperature (Deg C),Temperature (Deg C)"]
    event_headers, event_units = readers.process_header_lines(headers)
    assert event_headers["TMP"] == ["Time", "Temperature"]
    assert event_units["TMP"] == {"Temperature": "Deg C"}


def test_standardise_dataset_constant_and_datetime():
    ds = xr.Dataset(
        {
            "UID": ("time", ["007217"] * 3),
            "Node Ref": ("time", [""] * 3),
            "Record Time": (
                "time",
                ["2025/02/18 15:14:43", "2025/02/18 15:14:22", "2025/02/18 15:14:23"],
            ),
            "IntString": ("time", ["1", "2", "3"]),
            "FloatString": ("time", ["1.1", "2.2", "3.3"]),
        }
    )

    ds_std = readers.standardise_dataset(ds)
    assert "UID" not in ds_std
    assert "Node Ref" not in ds_std
    assert "UID" in ds_std.attrs
    assert pd.api.types.is_datetime64_any_dtype(ds_std["RECORD_TIME"].dtype)
    assert np.issubdtype(ds_std["INTSTRING"].dtype, np.integer)
    assert np.issubdtype(ds_std["FLOATSTRING"].dtype, np.floating)


def test_csv_to_xarray_single_event():
    # Prepare a simple CSV with one event type: BAT
    csv_content = """BAT,Record Time,Retrieval Time,Node Ref,UID,UsedPercentage %,Volts V
# Data
BAT,2025/02/18 15:14:43,2025/02/18 15:14:22,,007217,1,14.0
BAT,2023/01/18 12:30:00,2025/02/18 15:14:23,,007217,2,14.1
BAT,2023/01/18 14:00:00,2025/02/18 15:14:23,,007217,1,14.1
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write(csv_content)
        tmp_filename = tmp.name
    try:
        datasets = readers.csv_to_xarray(tmp_filename)
        # Expect one key in the returned dictionary
        assert "BAT" in datasets
        ds = datasets["BAT"]

        # Check that the expected variables (columns) are present.
        expected_vars = [
            "Record Time",
            "Retrieval Time",
            "Node Ref",
            "UID",
            "UsedPercentage",
            "Volts",
        ]
        for var in expected_vars:
            assert (
                var in ds.data_vars
            ), f"Expected variable {var} not found in the BAT dataset."
    finally:
        os.remove(tmp_filename)


def test_csv_to_xarray_AZA():
    # Create a CSV snippet for event type AZA.
    csv_content = """AZA,Record Time,Retrieval Time,Node Ref,UID,Index,Age(s),Age(s),Report,Report,Status(Hex),Status(Hex),Transfer Pressure(kPa),Transfer Pressure(kPa),Transfer Temperature(Deg C),Transfer Temperature(Deg C),Transfer SN,Transfer SN,Ambient Pressure(kPa),Ambient Pressure(kPa),Ambient Temperature(Deg C),Ambient Temperature(Deg C),Ambient SN,Ambient SN,Low Pressure(kPa),Low Pressure(kPa),Low Temperature(Deg C),Low Temperature(Deg C),Serial Number,Serial Number,Mean Square Error(kPa),Mean Square Error(kPa),Rate of Change from settling(kPa/Sec),Rate of Change from settling(kPa/Sec),Samples Remaining,Samples Remaining
# Data
AZA,2023/01/18 14:10:11,2025/02/18 15:14:23,,007217,2,31.5,4023,8019,59753.914,16.614,155399,59750.695,16.224,1262636,111.264,-9999.000,0,0.0007,-0.06,88
AZA,2023/01/18 14:10:11,2025/02/18 15:14:23,,007217,2,163.5,4053,8016,100.620,16.636,155399,59745.406,16.263,1262636,101.465,-9999.000,0,0.0002,0.02,88
"""
    # Write CSV content to a temporary file.
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write(csv_content)
        tmp_filename = tmp.name

    try:
        # Process the CSV file.
        datasets = readers.csv_to_xarray(tmp_filename)

        # Check that the output includes an AZA dataset.
        assert "AZA" in datasets, "AZA event type not found in the returned datasets."
        ds = datasets["AZA"]

        # Check that 'Record Time' is present
        assert "Record Time" in ds.data_vars, "Record Time variable missing."

        # Check that Temperature Deg C exists and is numeric.
        assert (
            "Transfer Temperature" in ds.data_vars
        ), "Transfer Temperature variable missing."

        # Check that 'Transfer Temperature' has the attribute 'units'.
        assert (
            "units" in ds["Transfer Temperature"].attrs
        ), "Transfer Temperature variable is missing the 'units' attribute."

    finally:
        os.remove(tmp_filename)


def test_csv_to_xarray(tmp_path):
    content = """TMP,Record Time,Temperature (Deg C)
                # Data
                TMP,2025/03/29 12:00:00,22.4
                 """
    file = tmp_path / "test.csv"
    file.write_text(content)

    datasets = readers.csv_to_xarray(str(file))
    assert "TMP" in datasets
    assert "Temperature" in datasets["TMP"]
    assert datasets["TMP"]["Temperature"].attrs["units"] == "Deg C"


def test_read_csv_to_xarray_real_sample_data(sample_csv_path):
    """Test reading the real sample_data.csv file."""
    if not sample_csv_path.exists():
        pytest.skip("Real sample_data.csv not available")

    # Use the main function that processes the real CSV file
    datasets = readers.read_csv_to_xarray(str(sample_csv_path))

    # Check that we get datasets back
    assert isinstance(datasets, dict)
    assert len(datasets) > 0

    # Check for expected event types that should be in the sample data
    # Based on the CSV header we saw earlier, these should be present
    expected_event_types = [
        "BAT",
        "TMP",
        "KLR",
        "INC",
        "DQZ",
        "PIES",
        "STP",
        "MOD",
        "REP",
        "BAS",
    ]

    # At least some of these should be present
    found_types = [
        event_type for event_type in expected_event_types if event_type in datasets
    ]
    assert (
        len(found_types) > 0
    ), f"No expected event types found. Available: {list(datasets.keys())}"

    # Check that each dataset has the expected structure
    for event_type, ds in datasets.items():
        assert isinstance(ds, xr.Dataset)
        # Should have some data variables
        assert len(ds.data_vars) > 0
        # Should have RECORD_TIME if it was processed
        if "RECORD_TIME" in ds.data_vars:
            assert len(ds["RECORD_TIME"]) > 0
