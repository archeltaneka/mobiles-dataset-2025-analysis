def standardize_col_names(df):
    col_names = df.columns.values
    for idx, col in enumerate(col_names):
        col_names[idx] = '_'.join([word for word in col.lower().replace('(', '').replace(')', '').split(' ')])
    df.columns = col_names

    return df


def lowercase_string_cols(df):
    for col in df.columns:
        if df[col].dtypes == 'object':
            df[col] = df[col].str.lower()

    return df


def fill_null_values(df):
    df['internal_memory'] = df['internal_memory'].fillna(df['internal_memory'].mean())
    df['launched_price_pakistan'] = df['launched_price_pakistan'].fillna(df['launched_price_pakistan'].median())
    df['launched_price_china'] = df['launched_price_china'].fillna(df['launched_price_china'].median())
    
    return df
    

def cast_column_types(df):
    df['mobile_weight'] = df['mobile_weight'].astype('float')
    df['ram'] = df['ram'].astype('float')
    df['front_camera'] = df['front_camera'].astype('int')
    df['back_camera'] = df['back_camera'].astype('int')
    df['battery_capacity'] = df['battery_capacity'].astype('int')
    df['screen_size'] = df['screen_size'].astype('float')
    df['internal_memory'] = df['internal_memory'].astype('float')

    for col in df.columns:
        if col.startswith('launched_price'):
            df[col] = df[col].astype('float')
    
    return df


def clip_price_outliers(df):
    for col in df.columns:
        if col.startswith('launched_price'):
            df[f'{col}_lower_bound'] = df[col].quantile(0.05)
            df[f'{col}_upper_bound'] = df[col].quantile(0.95)
            df[col] = df[col].clip(lower=df[f'{col}_lower_bound'], upper=df[f'{col}_upper_bound'])
            
    return df
