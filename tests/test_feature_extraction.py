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
    df = standardize_col_names(df)
    df = lowercase_string_cols(df)
    df = split_model_series_rom(df)
    return df


def test_extract_launched_price(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_launched_price(df)
    expected_df = pd.DataFrame(
        {'launched_price_pakistan': ['320000', '480000', '520000', '600000', '650000', '700000'],
         'launched_price_india': ['120000', '180000', '220000', '300000', '350000', '400000'],
         'launched_price_china': ['5000', '5600', '5200', '6000', '6500', '5500'],
         'launched_price_usa': ['700', '780', '800', '990', '1100', '800'],
         'launched_price_dubai': ['3500', '3600', '3700', '3800', '3900', '4000']}
    )

    for col in df.columns:
        if col.startswith('launched_price'):
            assert df[col].str.contains(r'(\d+)$').all(), f'{col} values must only contain numbers'

    assert_frame_equal(df.filter(like='launched_price'), expected_df, check_index_type=False)


def test_extract_screen_size(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_screen_size(df)
    expected_df = pd.DataFrame({'screen_size': ['6.4', '6.1', '6.2', '6.3', '6.4', '6.5']})

    assert df['screen_size'].str.contains(r'(\d+\.?\d*)$').all(), 'screen_size values must only contain numbers'
    assert_frame_equal(df.filter(like='screen_size'), expected_df, check_index_type=False)


def test_extract_battery_capacity(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_battery_capacity(df)
    expected_df = pd.DataFrame({'battery_capacity': ['5000', '4500', '4000', '3900', '5200', '4800']})

    assert df['battery_capacity'].str.contains(r'(\d+\.?\d*)$').all(), 'battery_capacity values must only contain numbers'
    assert_frame_equal(df.filter(like='battery_capacity'), expected_df, check_index_type=False)


def test_extract_camera_reso(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_camera_reso(df)
    expected_df = pd.DataFrame({
        'front_camera': ['8', '6', '6', '4', '10', '12'],
        'back_camera': [48, 60, 24, 8, 62, 46]
    })

    assert df['front_camera'].str.contains(r'(\d+\.?\d*)$').all(), 'front_camera values must only contain numbers' 
    assert (df['back_camera'] > 0).all(), 'back_camera values must only contain numbers'
    assert_frame_equal(df.filter(like='camera'), expected_df, check_index_type=False)


def test_extract_ram(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_ram(df)
    expected_df = pd.DataFrame({'ram': ['6', '8', '12', '6', '8', '12']})

    assert df['ram'].str.contains(r'(\d+\.?\d*)$').all(), 'ram values must only contain numbers'
    assert_frame_equal(df.filter(like='ram'), expected_df, check_index_type=False)


def test_extract_weight(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_weight(df)
    expected_df = pd.DataFrame({'mobile_weight': ['144', '129', '156', '129', '144', '129']})

    assert df['mobile_weight'].str.contains(r'(\d+\.?\d*)$').all(), 'mobile_weight values must only contain numbers'
    assert_frame_equal(df.filter(like='mobile_weight'), expected_df, check_index_type=False)


def test_split_model_series_rom(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    expected_df = pd.DataFrame({
        'model': ['model', 'model', 'model', 'model', 'model', 'model'],
        'series': ['a', 'b', 'c', 'd', 'e', 'f'],
        'internal_memory': ['256', '128', '1', '64', '2', '512']
    })

    assert 'model' in df, 'model column must exist'
    assert 'series' in df, 'series column must exist'
    assert 'internal_memory' in df, 'internal_memory column must exist'
    assert df['internal_memory'].str.contains(r'\d+$').all(), 'internal_memory values must only contain numbers'
    assert_frame_equal(df[['model', 'series', 'internal_memory']], expected_df, check_index_type=False)



def test_extract_features_from_processor(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_features_from_processor(df)
    expected_df = pd.DataFrame({
        'processor_name': ['snapdragon', 'snapdragon', 'a17', 'exynos', 'mediatek', 'snapdragon'],
        'processor_type': ['8 gen 1', '685', 'bionic', '1350', 'helio p35', '8 gen 2']
    })

    assert 'processor_name' in df, 'processor_name column must exist'
    assert 'processor_type' in df, 'processor_type column must exist'
    assert 'processor' not in df, 'processor column must be dropped'
    assert_frame_equal(df[['processor_name', 'processor_type']], expected_df, check_index_type=False)


def test_normalize_processor_types(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_features_from_processor(df)
    df = normalize_processor_names(df)
    expected_df = pd.DataFrame({
        'processor_name': ['qualcomm', 'qualcomm', 'a17', 'exynos', 'mediatek', 'qualcomm'],
        'processor_type': ['snapdragon 8 gen 1', 'snapdragon 685', 'bionic', '1350', 'helio p35', 'snapdragon 8 gen 2']
    })

    assert_frame_equal(df[['processor_name', 'processor_type']], expected_df, check_index_type=False)


def test_convert_internal_memory_to_gb(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_features_from_model_name(df)
    df = extract_features_from_processor(df)
    df = normalize_processor_names(df)
    df = normalize_processor_types(df)
    df = cast_column_types(df)
    df = convert_internal_memory_to_gb(df)
    expected_df = pd.DataFrame({'internal_memory': [256.0, 128.0, 1000.0, 64.0, 2000.0, 512.0]})

    assert_frame_equal(df[['internal_memory']], expected_df, check_index_type=False)


def test_extract_model_series(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = extract_features_from_model_name(df)
    expected_df = pd.DataFrame({'model_series': ['samsung model a', 'apple model b', 'google model c', 'oppo model d', 'vivo model e', 'xiaomi model f']})

    assert_frame_equal(df[['model_series']], expected_df, check_index_type=False)
