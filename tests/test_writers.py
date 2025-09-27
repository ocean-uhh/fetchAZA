import os
import pathlib
import sys
import tempfile
import shutil
from unittest.mock import patch
import pytest

script_dir = pathlib.Path(__file__).parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

import numpy as np
import xarray as xr
from fetchAZA import writers


@pytest.fixture
def sample_dataset():
    """Load a real sample dataset from the project's test data."""
    # Use the AZAseq sample file since it's a core sensor (not optional like PIES)
    sample_file = (
        pathlib.Path(__file__).parent.parent / "data" / "sample_data_AZAseq.nc"
    )
    if sample_file.exists():
        return xr.open_dataset(sample_file)
    else:
        # Fallback to simple test dataset if sample file doesn't exist
        data = {"PRESSURE": (["time"], np.random.rand(10) + 1000)}
        coords = {"time": np.arange(10)}
        ds = xr.Dataset(data, coords=coords)
        ds.attrs["title"] = "Test Dataset"
        ds["PRESSURE"].attrs["units"] = "kPa"
        return ds


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def real_sample_datasets():
    """Load real sample datasets from the project's data directory."""
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    datasets = {}

    # Load existing sample netCDF files for core sensors
    for key in ["DQZ", "AZAseq"]:
        sample_file = data_dir / f"sample_data_{key}.nc"
        if sample_file.exists():
            try:
                datasets[key] = xr.open_dataset(sample_file)
            except Exception:
                # If file can't be read, create a minimal placeholder
                datasets[key] = xr.Dataset(
                    {
                        "RECORD_TIME": (["time"], np.arange(5)),
                        "PRESSURE": (["time"], np.random.rand(5) + 1000),
                    }
                )
        else:
            # Create placeholder if file doesn't exist
            datasets[key] = xr.Dataset(
                {
                    "RECORD_TIME": (["time"], np.arange(5)),
                    "PRESSURE": (["time"], np.random.rand(5) + 1000),
                }
            )

    return datasets


@pytest.fixture
def sample_csv_path():
    """Get the path to the real sample CSV file."""
    return pathlib.Path(__file__).parent.parent / "data" / "sample_data.csv"


class TestSaveDataset:
    """Test cases for save_dataset function."""

    def test_save_dataset_new_file(self, sample_dataset, temp_dir):
        """Test saving a dataset to a new file."""
        output_file = os.path.join(temp_dir, "test_output.nc")
        result = writers.save_dataset(sample_dataset, output_file, overwrite=True)

        assert result is True
        assert os.path.exists(output_file)

        # Verify the file can be read back
        ds_loaded = xr.open_dataset(output_file)
        # Check for variables that should be in the AZAseq dataset
        assert "RECORD_TIME" in ds_loaded.variables or "PRESSURE" in ds_loaded.variables
        ds_loaded.close()

    def test_save_dataset_overwrite_true(self, sample_dataset, temp_dir):
        """Test overwriting an existing file with overwrite=True."""
        output_file = os.path.join(temp_dir, "test_overwrite.nc")

        # Create initial file
        writers.save_dataset(sample_dataset, output_file, overwrite=True)
        initial_mtime = os.path.getmtime(output_file)

        # Wait a moment to ensure different modification time
        import time

        time.sleep(0.1)

        # Overwrite the file
        result = writers.save_dataset(sample_dataset, output_file, overwrite=True)

        assert result is True
        assert os.path.exists(output_file)
        assert os.path.getmtime(output_file) > initial_mtime

    def test_save_dataset_overwrite_false(self, sample_dataset, temp_dir):
        """Test skipping save when file exists and overwrite=False."""
        output_file = os.path.join(temp_dir, "test_skip.nc")

        # Create initial file
        writers.save_dataset(sample_dataset, output_file, overwrite=True)
        initial_mtime = os.path.getmtime(output_file)

        # Try to save with overwrite=False
        result = writers.save_dataset(sample_dataset, output_file, overwrite=False)

        assert result is False
        assert os.path.getmtime(output_file) == initial_mtime

    @patch("builtins.input", return_value="y")
    def test_save_dataset_overwrite_none_yes(
        self, mock_input, sample_dataset, temp_dir
    ):
        """Test user prompt with 'yes' response when overwrite=None."""
        output_file = os.path.join(temp_dir, "test_prompt_yes.nc")

        # Create initial file
        writers.save_dataset(sample_dataset, output_file, overwrite=True)
        initial_mtime = os.path.getmtime(output_file)

        # Wait a moment to ensure different modification time
        import time

        time.sleep(0.1)

        # Save with overwrite=None (should prompt)
        result = writers.save_dataset(sample_dataset, output_file, overwrite=None)

        assert result is True
        assert os.path.getmtime(output_file) > initial_mtime
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="n")
    def test_save_dataset_overwrite_none_no(self, mock_input, sample_dataset, temp_dir):
        """Test user prompt with 'no' response when overwrite=None."""
        output_file = os.path.join(temp_dir, "test_prompt_no.nc")

        # Create initial file
        writers.save_dataset(sample_dataset, output_file, overwrite=True)
        initial_mtime = os.path.getmtime(output_file)

        # Save with overwrite=None (should prompt)
        result = writers.save_dataset(sample_dataset, output_file, overwrite=None)

        assert result is False
        assert os.path.getmtime(output_file) == initial_mtime
        mock_input.assert_called_once()

    def test_save_dataset_with_logging(self, sample_dataset, temp_dir):
        """Test that save_dataset logs success messages."""
        output_file = os.path.join(temp_dir, "test_logging.nc")

        with patch("fetchAZA.writers._log") as mock_log:
            result = writers.save_dataset(sample_dataset, output_file, overwrite=True)

            assert result is True
            mock_log.info.assert_called_with(
                f"Successfully saved dataset to: {output_file}"
            )


class TestSaveDatasets:
    """Test cases for save_datasets function."""

    def test_save_datasets_success(self, temp_dir, real_sample_datasets):
        """Test saving multiple datasets successfully using real sample data."""
        input_fn = os.path.join(temp_dir, "test_input.csv")
        writers.save_datasets(real_sample_datasets, input_fn, overwrite=True)

        # Check that all files were created
        for key in real_sample_datasets.keys():
            expected_file = os.path.join(temp_dir, f"test_input_{key}.nc")
            assert os.path.exists(expected_file)

    def test_save_datasets_with_overwrite_false(self, temp_dir, real_sample_datasets):
        """Test save_datasets with overwrite=False using real sample data."""
        # Use just one dataset for this test
        datasets = {"DQZ": real_sample_datasets["DQZ"]}
        input_fn = os.path.join(temp_dir, "test_input.csv")

        # Save once
        writers.save_datasets(datasets, input_fn, overwrite=True)

        # Try to save again with overwrite=False
        with patch("fetchAZA.writers._log") as mock_log:
            writers.save_datasets(datasets, input_fn, overwrite=False)
            # Should log an error about failing to save
            mock_log.error.assert_called()


class TestDeleteNetcdfDatasets:
    """Test cases for delete_netcdf_datasets function."""

    def test_delete_existing_files(self, temp_dir):
        """Test deleting existing netCDF files."""
        # Create test files
        file_root = "test_data"
        keys = ["DQZ", "PIES", "KLR"]
        created_files = []

        for key in keys:
            filename = os.path.join(temp_dir, f"{file_root}_{key}.nc")
            # Create dummy netCDF files
            ds = xr.Dataset({"data": (["x"], [1, 2, 3])})
            ds.to_netcdf(filename)
            created_files.append(filename)

        # Verify files exist
        for filename in created_files:
            assert os.path.exists(filename)

        # Delete files
        deleted_count = writers.delete_netcdf_datasets(temp_dir, file_root, keys)

        assert deleted_count == len(keys)

        # Verify files are deleted
        for filename in created_files:
            assert not os.path.exists(filename)

    def test_delete_nonexistent_files(self, temp_dir):
        """Test deleting files that don't exist."""
        file_root = "nonexistent"
        keys = ["DQZ", "PIES"]

        deleted_count = writers.delete_netcdf_datasets(temp_dir, file_root, keys)

        # Should return 0 since no files were actually deleted
        assert deleted_count == 0

    def test_delete_mixed_existing_nonexisting(self, temp_dir):
        """Test deleting a mix of existing and non-existing files."""
        file_root = "mixed_test"
        keys = ["DQZ", "PIES", "KLR"]

        # Create only some of the files
        existing_keys = ["DQZ", "KLR"]
        for key in existing_keys:
            filename = os.path.join(temp_dir, f"{file_root}_{key}.nc")
            ds = xr.Dataset({"data": (["x"], [1, 2, 3])})
            ds.to_netcdf(filename)

        deleted_count = writers.delete_netcdf_datasets(temp_dir, file_root, keys)

        # Should only count actually deleted files
        assert deleted_count == len(existing_keys)


if __name__ == "__main__":
    pytest.main([__file__])
