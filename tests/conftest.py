import pandas as pd
import pytest


@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame({
        "Company Name": ["Samsung", "Apple"],
        "Model Name": ["Samsung A", "Iphone B"],
        "Mobile Weight": [999, 1299],
        "RAM": [999, 1299],
        "Front Camera": [999, 1299],
        "Back Camera": [8, 6],
        "Processor": [128, 256],
        "Battery Capacity": [5000, 4500],
        "Screen Size": [6.4, 6.1],
        "Launched Price (Pakistan)": [108, 48],
        "Launched Price (India)": [108, 48],
        "Launched Price (China)": [108, 48],
        "Launched Price (USA)": [108, 48],
        "Launched Price (Dubai)": [108, 48],
        "Launched Year": [2015, 2020],
    })

@pytest.fixture
def sample_csv_path(tmp_path, sample_raw_dataframe):
    csv_path = tmp_path / "sample_data.csv"
    sample_raw_dataframe.to_csv(csv_path, index=False)
    return csv_path