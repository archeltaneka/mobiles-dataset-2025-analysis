import pandas as pd
import pytest


@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame({
        'Company Name': ['Samsung', 'Apple', 'Google', 'Oppo', 'Vivo', 'Xiaomi'],
        'Model Name': ['Model A 256GB', 'Model B 128GB', 'Model C 1TB', 'Model D 64GB', 'Model E 2TB', 'Model F 512GB'],
        'Mobile Weight': ['144g', '129g', '156g', '129g', '144g', '129g'],
        'RAM': ['6GB', '8GB', '12GB', '6GB', '8GB', '12GB'],
        'Front Camera': ['8MP', '6MP', '6MP', '4MP', '10MP', '12MP'],
        'Back Camera': ['48MP', '48MP + 12MP', '16MP + 8MP', '8MP', '50MP + 12MP', '30MP + 16MP'],
        'Processor': ['Snapdragon 8 Gen 1', 'Snapdragon 685', 'A17 Bionic', 'Exynos 1350', 'Mediatek Helio P35', 'Snapdragon 8 Gen 2'],
        'Battery Capacity': ['5,000mAh', '4,500mAh', '4,000mAh', '3,900mAh', '5,200mAh', '4,800mAh'],
        'Screen Size': ['6.4 inches', '6.1 inches', '6.2 inches', '6.3 inches', '6.4 inches', '6.5 inches'],
        'Launched Price (Pakistan)': ['PKR 320,000', 'PKR 480,000', 'PKR 520,000', 'PKR 600,000', 'PKR 650,000', 'PKR 700,000'],
        'Launched Price (India)': ['INR 120,000', 'INR 180,000', 'INR 220,000', 'INR 300,000', 'INR 350,000', 'INR 400,000'],
        'Launched Price (China)': ['CNY 5,000', 'CNY 5,600', 'CNY 5,200', 'CNY 6,000', 'CNY 6,500', 'CNY 5,500'],
        'Launched Price (USA)': ['USD 700', 'USD 780', 'USD 800', 'USD 990', 'USD 1,100', 'USD 800'],
        'Launched Price (Dubai)': ['AED 3,500', 'AED 3,600', 'AED 3,700', 'AED 3,800', 'AED 3,900', 'AED 4,000'],
        'Launched Year': [2015, 2018, 2019, 2020, 2016, 2017],
    })

@pytest.fixture
def sample_csv_path(tmp_path, sample_raw_dataframe):
    csv_path = tmp_path / 'sample_data.csv'
    sample_raw_dataframe.to_csv(csv_path, index=False)
    return csv_path