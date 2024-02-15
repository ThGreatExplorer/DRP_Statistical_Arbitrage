# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import random
import quantstats
from data import Data

# %%
class PairTrading:
    """
    Given a Data object containing potentially stationary data, determine if the data is O(1) stationary, check if the time series are cointegrated and find the linear combination that 
    creates cointegration. Then confirm, if their linear combination is O(0) stationary.  
    """

    def __init__(self, data :Data) -> None:
        self.data = data.data
        self.cointegration = None
        self.X, self.y = self.__split__()
        self.__run_ols__()
        self.__check_for_stationary__()
        
    def __split__(self):
        """X/y split, where 0th column is the dependent variable, for cointegration search"""
        col = self.data.columns[0]
        self.dependent = col
        y = self.data[col]
        X = self.data.drop(columns = [col])
        return X, y
    
    def __check_for_stationary__(self):
        "checks the linear combination for stationarity based on 95% confidence interval"
        if self.cointegration == None:
            self.adf = adfuller(self.stationary)
            p_val = self.adf[1]
            if (p_val < 0.05):
                print('p-value = ' + str(p_val) + ' The series ' + self.stationary_series +' is likely stationary.')
                self.cointegration = True
            else: 
                print('p-value = ' + str(p_val) + ' The series ' + self.stationary_series +' is likely not stationary.')
                self.cointegration = False
        else:
            return self.cointegration

    
    def __run_ols__(self):
        """ runs OLS on the groups """
        model = sm.OLS(self.y, self.X).fit()
        self.model = model
        self.predictions = model.predict()
        self.stationary = self.y - self.predictions
        self.stationary_series = f"{self.y.name} - {model.params.iloc[0]} * {self.X.columns.tolist()}" 
    
    def __str__(self) -> str:
        return f"dependent: {self.dependent} \nstationary_series: {self.stationary_series}\ncointegration: {self.cointegration}"



# %%
class Arbitrage:
    """
    An object that caches and holds data about some (potentially)
    stationary data. The idea of this object is that given some I(1) 
    data, it will abitrarily look to find cointegrated group. Then, it 
    allows methods to view the plotted differenced series, displays
    information about the stationarity test, etc...
    """
    
    def __init__(self, data, method = 'ols'):
        self.data = data
        self.X, self.y = self.__split__()
        self.test_results = None
        self.method = method
        if self.method == 'ols':
            self.__run_ols__()
        if self.method == 'wls':
            self.__run_wls__()
        
    def __split__(self):
        """ arbitrary X/y split for cointegration search """
        col = random.choice(self.data.columns)
        self.dependent = col
        y = self.data[col]
        X = self.data.drop(columns = [col])
        return X, y
        
    def __run_ols__(self):
        """ runs OLS on the groups """
        model = sm.OLS(self.y, self.X).fit()
        self.model = model
        self.predictions = model.predict()
        self.stationary = self.y - self.predictions
        
    def __run_wls__(self):
        """ runs WLS with exponential decay on the groups """
        decay_factor = 0.01
        decay = np.exp(-decay_factor * (self.data.index.max() - self.data.index).days)

        model = sm.WLS(self.y, self.X, weights = decay).fit()
        self.model = model
        self.predictions = model.predict()
        self.stationary = self.y - self.predictions
        
    def show_plot(self):
        """ plots the (potentially) stationary differenced series """
        self.stationary.plot()
    
    def get_test_results(self):
        """ returns test results of the stationarity test """
        if self.test_results == None:
            self.adf = adfuller(self.stationary)
            return self.adf
        else:
            return self.adf

# %%
print("Imported class `Arbitrage(data : pd.DataFrame)` with methods `show_plot()`, `get_test_results()`")

# %%



