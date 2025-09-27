import pathlib
import sys

script_dir = pathlib.Path(__file__).parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

import numpy as np
import xarray as xr
from fetchAZA import tools


def test_convert_units_var():
    values = np.array([100, 200])
    new_vals, new_unit, err = tools.convert_units_var(values, "cm/s", "m/s")
    np.testing.assert_array_almost_equal(new_vals, [1, 2])
    assert new_unit == "m/s"
    assert err == ""


def test_reformat_units_var():
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

    for original_unit, expected_format in unit_str_format.items():
        ds = xr.Dataset({"x": ("t", [1])})
        ds["x"].attrs["units"] = original_unit
        assert tools.reformat_units_var(ds, "x") == expected_format


# Additional tests for enhanced functionality could be added here
# Currently focusing on core unit conversions and formatting
