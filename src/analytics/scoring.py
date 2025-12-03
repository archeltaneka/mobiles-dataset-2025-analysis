from sklearn.preprocessing import MinMaxScaler


def calculate_value_score(df):
    features = ['ram', 'screen_size', 'battery_capacity', 'mobile_weight', 'internal_memory', 'front_camera', 'back_camera', 'launched_price_usa']
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df[features])
    
    for i, col in enumerate(features):
        df[f'{col}_scaled'] = features_scaled[:, i]

    # Try with different weight values!
    df['spec_score'] = (
                            # Bigger weight values for internal memory and RAM
                            df['internal_memory_scaled'] * 0.3 +
                            df['ram_scaled'] * 0.3 +
                            # Smaller weight values for the rest
                            df['screen_size_scaled'] * 0.08 +
                            df['battery_capacity_scaled'] * 0.08 +
                            df['mobile_weight_scaled'] * 0.08 +
                            df['front_camera_scaled'] * 0.08 +
                            df['back_camera_scaled'] * 0.08
    )

    df['launched_price_usa_scaled'] = df['launched_price_usa_scaled'].replace(0, 0.001)  # Avoid division by zero
    df['Value_Score'] = df['spec_score'] / df['launched_price_usa_scaled']

    return df