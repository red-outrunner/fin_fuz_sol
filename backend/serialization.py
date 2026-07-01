"""JSON-safe serialization helpers shared across the data and analytics layers."""
import numpy as np
import pandas as pd


def clean_data(data):
    """Recursively replace NaN and Infinity with None for JSON serialization."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if isinstance(k, (np.integer, np.int64, np.int32)):
                k = str(int(k))
            elif isinstance(k, pd.Timestamp) or isinstance(k, np.datetime64):
                k = str(k)
            cleaned[k] = clean_data(v)
        return cleaned
    elif isinstance(data, list):
        return [clean_data(v) for v in data]
    elif isinstance(data, (float, np.float64, np.float32)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.int64, np.int32)):
        return int(data)
    elif isinstance(data, pd.Series):
        return clean_data(data.to_dict())
    elif isinstance(data, pd.DataFrame):
        return clean_data(data.to_dict(orient='records'))
    return data
