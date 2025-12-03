import numpy as np
import pandas as pd


def assign_phone_clusters(df):
    df['phone_cluster'] = pd.cut(
        df['launched_price_usa'],
        bins=[-np.inf, 250, 600, np.inf], # < 250: Budget, 250-600: Mid-range, > 600: Flagship
        labels=['Budget', 'Mid', 'Flagship']
    )

    return df