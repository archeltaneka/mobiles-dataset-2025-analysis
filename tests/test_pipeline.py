import os

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.data.mobiles import MobilesData
from src.cleaning.pipeline import clean_data


def load_mock_data(sample_csv_path):
    loader = MobilesData(data_dir=sample_csv_path.parent, fname=sample_csv_path.name)
    df = loader.load_data()
    return df


def test_pipeline(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = clean_data(df)
    expected_df = pd.DataFrame({
        'company_name': {
            0: 'samsung',
            1: 'apple',
            2: 'google',
            3: 'oppo',
            4: 'vivo',
            5: 'xiaomi'
        },
        'mobile_weight': {0: 144.0, 1: 129.0, 2: 156.0, 3: 129.0, 4: 144.0, 5: 129.0},
        'ram': {0: 6.0, 1: 8.0, 2: 12.0, 3: 6.0, 4: 8.0, 5: 12.0},
        'front_camera': {0: 8, 1: 6, 2: 6, 3: 4, 4: 10, 5: 12},
        'back_camera': {0: 48, 1: 60, 2: 24, 3: 8, 4: 62, 5: 46},
        'battery_capacity': {0: 5000, 1: 4500, 2: 4000, 3: 3900, 4: 5200, 5: 4800},
        'screen_size': {0: 6.4, 1: 6.1, 2: 6.2, 3: 6.3, 4: 6.4, 5: 6.5},
        'launched_price_pakistan': {
            0: 360000.0,
            1: 480000.0,
            2: 520000.0,
            3: 600000.0,
            4: 650000.0,
            5: 687500.0
        },
        'launched_price_india': {
            0: 135000.0,
            1: 180000.0,
            2: 220000.0,
            3: 300000.0,
            4: 350000.0,
            5: 387500.0
        },
        'launched_price_china': {   
            0: 5050.0,
            1: 5600.0,
            2: 5200.0,
            3: 6000.0,
            4: 6375.0,
            5: 5500.0
        },
        'launched_price_usa': {
            0: 720.0,
            1: 780.0,
            2: 800.0,
            3: 990.0,
            4: 1072.5,
            5: 800.0
        },
        'launched_price_dubai': {
            0: 3525.0,
            1: 3600.0,
            2: 3700.0,
            3: 3800.0,
            4: 3900.0,
            5: 3975.0
        },
        'launched_year': {0: 2015, 1: 2018, 2: 2019, 3: 2020, 4: 2016, 5: 2017},
        'model': {
            0: 'model',
            1: 'model',
            2: 'model',
            3: 'model',
            4: 'model',
            5: 'model'
        },
        'internal_memory': {
            0: 256.0,
            1: 128.0,
            2: 1000.0,
            3: 64.0,
            4: 2000.0,
            5: 512.0
        },
        'series': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'},
        'processor_name': {
            0: 'qualcomm',
            1: 'qualcomm',
            2: 'a17',
            3: 'exynos',
            4: 'mediatek',
            5: 'qualcomm'
        },
        'processor_type': {
            0: 'snapdragon 8 gen 1',
            1: 'snapdragon 685',
            2: 'bionic',
            3: '1350',
            4: 'helio p35',
            5: 'snapdragon 8 gen 2'
        }
    })

    assert_frame_equal(df[list(expected_df.columns)], expected_df, check_index_type=False)