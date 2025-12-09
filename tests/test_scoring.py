import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
import pytest

from src.data.mobiles import MobilesData
from src.cleaning.pipeline import clean_data
from src.analytics.scoring import calculate_value_score


def load_mock_data(sample_csv_path):
    loader = MobilesData(data_dir=sample_csv_path.parent, fname=sample_csv_path.name)
    df = loader.load_data()
    df = clean_data(df)
    expected_df = df.copy()
    df = calculate_value_score(df)
    
    return df


def test_scoring(sample_csv_path):
    df = load_mock_data(sample_csv_path)
    df = calculate_value_score(df)

    expected_df = pd.DataFrame({'Value_Score': [3.011481e+02, 1.432780e+00, 2.621273e+00, 5.222222e-02, 7.244444e-01, 2.824857e+00]})

    assert 'Value_Score' in df
    assert_allclose(df[['Value_Score']], expected_df, rtol=1e-05, atol=1e-08)