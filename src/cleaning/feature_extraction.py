import numpy as np


def extract_launched_price(df):
    for col in df.columns:
        if col.startswith('launched_price'):
            df[col] = df[col].str.replace(',', '')
            df[col] = df[col].str.extract(r'(\d+)$')
            
    return df


def extract_screen_size(df):
    df['screen_size'] = df['screen_size'].str.extract(r'(\d+\.?\d*)')
    return df


def extract_battery_capacity(df):
    df['battery_capacity'] = df['battery_capacity'].str.replace(',', '')
    df['battery_capacity'] = df['battery_capacity'].str.replace('mah', '')
    return df


def sum_back_camera_resolutions(s):
    nums = s.str.extractall(r'(\d+)mp')[0].astype(float)
    return nums.groupby(level=0).sum()


def extract_camera_reso(df):
    df['front_camera'] = df['front_camera'].str.extract('(\d+)mp')
    df['back_camera'] = sum_back_camera_resolutions(df['back_camera']) # Some phones have multiple back cameras. In this case, we can just sum them up
    return df


def extract_ram(df):
    df['ram'] = df['ram'].str.replace('gb', '')
    df['ram'] = df['ram'].str.split('/').str[-1].str.strip()
    
    return df


def extract_weight(df):
    df['mobile_weight'] = df['mobile_weight'].str.replace('g', '')
    return df


def split_model_series_rom(df):
    df['model_name'] = df['model_name'].str.split()
    df['model'] = df['model_name'].str[0]
    df['internal_memory'] = df['model_name'].str[-1]
    df['internal_memory'] = np.where(df['internal_memory'].str.contains('gb|tb', na=False), df['internal_memory'], np.nan)
    df['internal_memory'] = df['internal_memory'].str.extract(r'(\d+)[gb|tb]')
    df['series'] = np.where(df['internal_memory'].isnull(), df['model_name'].str[1:].str.join(' '), df['model_name'].str[1:-1].str.join(' '))

    return df


def extract_features_from_processor(df):
    df['processor'] = df['processor'].str.split()
    df['processor_name'] = df['processor'].str[0]
    df['processor_type'] = df['processor'].str[1:].str.join(' ')
    df = df.drop('processor', axis=1)

    return df


def normalize_processor_names(df):
    df['processor_type'] = np.where(df['processor_name']=='snapdragon', 'snapdragon ' + df['processor_type'], df['processor_type'])
    df['processor_name'] = np.where(df['processor_name']=='snapdragon', 'qualcomm', df['processor_name'])
    df['processor_name'] = np.where(df['processor_name'].isin(['dimensity', 'helio']), 'mediatek', df['processor_name'])

    return df


def normalize_processor_types(df):
    processor_name_list = df['processor_name'].unique().tolist()
    df['processor_type'] = np.where(df['processor_name'].str.split().str[0].isin(processor_name_list), 
                                    'snapdragon ' + df['processor_type'], 
                                    df['processor_type'])

    return df


def convert_internal_memory_to_gb(df):
    df['internal_memory'] = np.where(df['internal_memory'] < 16, df['internal_memory'] * 1000, df['internal_memory']) # 1TB is approx 1,000 GB
    return df


def extract_model_series(df):
    df['model_series'] = df['company_name'] + ' ' + df['model'] + ' ' + df['series']
    return df


def extract_features_from_model_name(df):
    df = extract_launched_price(df)
    df = extract_screen_size(df)
    df = extract_battery_capacity(df)
    df = extract_camera_reso(df)
    df = extract_ram(df)
    df = extract_weight(df)
    df = df.drop('model_name', axis=1)
    df = extract_model_series(df)

    return df