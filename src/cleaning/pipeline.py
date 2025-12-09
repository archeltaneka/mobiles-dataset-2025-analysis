from .preprocessing import *
from .feature_extraction import *


def clean_data(df, fill_na=True):
    df = standardize_col_names(df)
    df = lowercase_string_cols(df)
    df = split_model_series_rom(df)
    df = extract_features_from_model_name(df)
    df = extract_features_from_processor(df)
    df = normalize_processor_names(df)
    df = cast_column_types(df)
    df = convert_internal_memory_to_gb(df)

    if fill_na:
        df = fill_null_values(df)
    
    df = clip_price_outliers(df)

    return df
