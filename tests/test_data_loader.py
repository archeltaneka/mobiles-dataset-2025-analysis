import os

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.data.mobiles import MobilesData


def test_load_data(sample_csv_path, sample_raw_dataframe):
    loader = MobilesData(data_dir=sample_csv_path.parent, fname=sample_csv_path.name)
    df = loader.load_data()

    assert not df.empty, 'Loaded data is empty'
    assert os.path.getsize(os.path.join(sample_csv_path.parent, sample_csv_path.name)) > 0, 'Loaded data is empty'
    assert list(df.columns) == list(sample_raw_dataframe.columns), f'Loaded data must have the following columns: {sample_raw_dataframe.columns}'
    assert_frame_equal(df, sample_raw_dataframe), 'Loaded data does not have the same values with the expected data'


def test_load_data_file_not_found():
    loader = MobilesData(data_dir='non_existent_dir', fname='non_existent_file.csv')
    with pytest.raises(FileNotFoundError):
        loader.load_data()


