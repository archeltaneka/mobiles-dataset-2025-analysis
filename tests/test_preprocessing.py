import os

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.data.mobiles import MobilesData
from src.cleaning.preprocessing import *
from src.cleaning.feature_extraction import *


def load_mock_data(sample_csv_path):
    loader = MobilesData(data_dir=sample_csv_path.parent, fname=sample_csv_path.name)
    df = loader.load_data()
    return df


def test_preprocessing(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = standardize_col_names(df)
    expected_cols = [
        'company_name', 'model_name', 'mobile_weight', 'ram', 'front_camera', 'back_camera', 'processor', 
        'battery_capacity', 'screen_size', 'launched_price_pakistan', 'launched_price_india', 'launched_price_china', 
        'launched_price_usa', 'launched_price_dubai', 'launched_year']
    assert list(df.columns) == expected_cols


def test_lowercase_string_cols(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = lowercase_string_cols(df)
    for col in df.columns:
        if df[col].dtypes == 'object':
            assert df[col].str.islower().all()


def test_fill_null_values(sample_csv_path):
    pass


def test_cast_column_types(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = standardize_col_names(df)
    df = lowercase_string_cols(df)
    df = split_model_series_rom(df)
    df = extract_features_from_model_name(df)
    df = extract_features_from_processor(df)
    df = normalize_processor_names(df)
    df = normalize_processor_types(df)
    df = cast_column_types(df)

    assert df['company_name'].dtypes == 'object'
    assert df['model'].dtypes == 'object'
    assert df['series'].dtypes == 'object'
    assert df['processor_name'].dtypes == 'object'
    assert df['processor_type'].dtypes == 'object'

    assert df['mobile_weight'].dtypes == 'float64'
    assert df['screen_size'].dtypes == 'float64'
    assert df['launched_price_pakistan'].dtypes == 'float64'
    assert df['launched_price_india'].dtypes == 'float64'
    assert df['launched_price_china'].dtypes == 'float64'
    assert df['launched_price_usa'].dtypes == 'float64'
    assert df['launched_price_dubai'].dtypes == 'float64'

    assert df['ram'].dtypes == 'int64'
    assert df['front_camera'].dtypes == 'int64'
    assert df['back_camera'].dtypes == 'int64'
    assert df['internal_memory'].dtypes == 'int64'
    assert df['battery_capacity'].dtypes == 'int64'
    assert df['launched_year'].dtypes == 'int64'