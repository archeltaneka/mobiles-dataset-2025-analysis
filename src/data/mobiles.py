from pathlib import Path
import pandas as pd


class MobilesData:
    def __init__(self, data_dir, fname):
        self.data_dir = Path(data_dir)
        self.fname = fname

    def load_data(self):
        return pd.read_csv(self.data_dir/self.fname, encoding='unicode_escape') 

