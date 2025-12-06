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
    assert list(df.columns) == expected_cols, f'Standardized column names should have the following columns: {expected_cols}'


def test_lowercase_string_cols(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = lowercase_string_cols(df)
    for col in df.columns:
        if df[col].dtypes == 'object':
            assert df[col].str.islower().all(), f'Column {col} values should be in lowercase'


def test_fill_null_values(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = standardize_col_names(df)
    df = lowercase_string_cols(df)
    df = split_model_series_rom(df)
    df = extract_features_from_model_name(df)
    df = extract_features_from_processor(df)
    df = normalize_processor_names(df)
    df = normalize_processor_types(df)
    df = cast_column_types(df)
    df = convert_internal_memory_to_gb(df)
    df = fill_null_values(df)

    # These values must not have null values
    assert df['internal_memory'].isnull().sum() == 0, 'internal_memory column must not have null values'
    assert df['launched_price_pakistan'].isnull().sum() == 0, 'launched_price_pakistan column must not have null values'
    assert df['launched_price_india'].isnull().sum() == 0, 'launched_price_india column must not have null values'
    assert df['launched_price_china'].isnull().sum() == 0, 'launched_price_china column must not have null values'
    assert df['launched_price_usa'].isnull().sum() == 0, 'launched_price_usa column must not have null values'
    assert df['launched_price_dubai'].isnull().sum() == 0, 'launched_price_dubai column must not have null values'


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


def test_clip_price_outliers(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = standardize_col_names(df)
    df = lowercase_string_cols(df)
    df = split_model_series_rom(df)
    df = extract_features_from_model_name(df)
    df = extract_features_from_processor(df)
    df = normalize_processor_names(df)
    df = normalize_processor_types(df)
    df = cast_column_types(df)
    df = convert_internal_memory_to_gb(df)
    df = fill_null_values(df)
    df = clip_price_outliers(df)

    for col in df.columns:
        if col.endswith('bound'):
            assert df[col].isnull().sum() == 0, 'lower_bound column must not have null values'
            assert df[col].isnull().sum() == 0, 'upper_bound column must not have null values'
        
    for col in df.columns:
        if col.startswith('launched_price') and not col.endswith('bound'):
            assert df[col].between(df[f'{col}_lower_bound'], df[f'{col}_upper_bound']).all(), \
                f'{col} values must be between its lower bound and upper bound'