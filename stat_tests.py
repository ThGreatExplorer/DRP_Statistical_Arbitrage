# TODO Logic of checking the co-integration of two pairs

import numpy as np
import pandas as pd
from pandas import Series

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations

import matplotlib.pyplot as plt

class BondPairAnalysis:
    """
    Class to handle all the data processing. Main responsibility is to return the viable pairs of stocks which are cointegrated 
    (and whose linear combination also passes stationarity). Auxiliary reponsibilities include analyzing and visualizing the 
    time series.
    """

    # TODO fields

    def __init__(self) -> None:
        pass
    
    def process_csv(csv_file_path :str) -> Series:
        raw_csv = pd.read_csv(csv_file_path)
        


