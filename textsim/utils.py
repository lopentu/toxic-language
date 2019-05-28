from pathlib import Path
import numpy as np

def get_data_path(filename):
    fpath = Path(__file__).parent / f"data/{filename}"
    return fpath

def get_resource_path(filename):
    fpath = Path(__file__).parent / f"resources/{filename}"
    return fpath