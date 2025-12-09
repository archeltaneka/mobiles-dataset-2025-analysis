import os

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.data.mobiles import MobilesData
from src.cleaning.pipeline import clean_data
from src.analytics.clustering import assign_phone_clusters


def load_mock_data(sample_csv_path):
    loader = MobilesData(data_dir=sample_csv_path.parent, fname=sample_csv_path.name)
    df = loader.load_data()
    df = clean_data(df)

    return df


def test_clustering(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = assign_phone_clusters(df)
    expected_df = pd.DataFrame({'phone_cluster': ['Flagship', 'Flagship', 'Flagship', 'Flagship', 'Flagship', 'Flagship']})
    expected_df['phone_cluster'] = pd.Categorical(expected_df['phone_cluster'], categories=['Budget', 'Mid', 'Flagship'], ordered=True)

    assert 'phone_cluster' in df
    assert_frame_equal(df[['phone_cluster']], expected_df, check_index_type=False)