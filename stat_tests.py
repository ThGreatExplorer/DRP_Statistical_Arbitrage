import pandas as pd
from pandas import Series, DataFrame

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations

import matplotlib.pyplot as plt

def check_for_stationarity(X, cutoff=0.01) -> bool:
         # H_0 in adfuller is unit root exists (non-stationary)
        # We must observe significant p-value to convince ourselves that the series is stationary
        # adfuller (augmented dicky fuller test) is a statistically significant test for stationary dataset
        pvalue = adfuller(X)[1]
        if pvalue < cutoff:
            print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely stationary.')
            return True
        else:
            print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely non-stationary.')
            return False
        
class StationaryTimeSeries:
    """
    Class to handle the data processing of Bond csv data. Main responsibility is to store the processed data and calculate the orders 
    of integration of the series.
    """

    def __init__(self, csv_file :DataFrame, col_name :str, series_name :str) -> None:
        self.raw_csv = csv_file.dropna(inplace=False)
        self.t_series = self.raw_csv[col_name]
        self.t_series.name = series_name
        # will be in Date, Col_name format, where date is the index
        self.t_series.index = self.raw_csv['Date']
        self.series_name = series_name
        self.order_of_integration = self.orders_of_integration()
    
    def __str__(self):
        return f"Series {self.series_name}\n" + f"t_series: {self.t_series}\n" + f"orders_of_integration: {self.order_of_integration}\n"
    
    def orders_of_integration(self) -> int:
        stationary = False
        orders = 0
        series = self.t_series.copy()
        series.name = self.t_series.name + "_0"
        while (not stationary):
            stationary = check_for_stationarity(series)
            if not stationary:
                series = series.diff().dropna()  # Handle NaN values after differencing
                series.name = series.name[:-1] + str(int(series.name[-1]) + 1) 
                orders += 1
        return orders

        

class PairAnalysis:
    """
    Given two StationaryTimeSeries who are O(1) stationary, check if they are cointegrated and find the linear combination that 
    creates cointegration. Then confirm, if their linear combination is O(0) stationary.  
    """

    def __init__(self, t_series1 :StationaryTimeSeries, t_series2 :StationaryTimeSeries):
        if (t_series1.order_of_integration != 1 or t_series2.order_of_integration != 1):
             raise Exception(f"Series 1: {t_series1.series_name} is {t_series1.order_of_integration}\n" 
                             + f"Series 2: {t_series2.series_name} is {t_series2.order_of_integration}")
        self.t_series1 = t_series1
        self.t_series2 = t_series2
        self.merged_series =  pd.merge(self.t_series1.t_series, self.t_series2.t_series, how='inner', on="Date", suffixes=(f"{self.t_series1.series_name}", f"{self.t_series2.series_name}"))
        self.modified_series1 = self.merged_series[self.t_series1.series_name]
        self.modified_series2 = self.merged_series[self.t_series2.series_name]
        self.cointegrated_series = self.calculate_cointegrated_series()
        #self.cointegrated_stationary = check_for_stationarity(self.cointegrated_series) value storing if the linear combination passes stationarity

    def calculate_cointegrated_series(self) -> DataFrame:
        cointegrated = coint(self.modified_series1, self.modified_series2)
        p_value = cointegrated[1]
        if (p_value < 0.05):
             model = sm.OLS(self.modified_series1, self.modified_series2).fit()
             beta = model.params.iloc[0]
             series_diff = self.modified_series1 - beta * self.modified_series2
             series_diff.name = f"{self.modified_series1.name} - {beta:.2f} * {self.modified_series2.name}"
             print(series_diff)
             return series_diff
        else:
             print(f"p_value = {p_value}. The series {self.modified_series1.name} and {self.modified_series2.name} is not cointegrated")
             return None

def main():
    GER_5yr = StationaryTimeSeries(pd.read_csv("Data/csv/GER_10yr_Daily(1_5)_5yr.csv"), "Last Price", "GER_Daily_5yr")
    GB_5yr = StationaryTimeSeries(pd.read_csv("Data/csv/GB_10yr_Daily(1_5yr)_5yr.csv"), "Last Price", "GB_Daily_5yr")
    GER_GB_5yr = PairAnalysis(GER_5yr, GB_5yr)
    plt.plot(GER_GB_5yr.cointegrated_series)
    plt.show()
    print("\n")

    US_5yr = StationaryTimeSeries(pd.read_csv("Data/csv/US_10yr_Daily(1_5yr)_5yr.csv"), "Last Price", "US_Daily_5yr")
    JPY_5yr = StationaryTimeSeries(pd.read_csv("Data/csv/JPY_10yr_Daily(1_5)_5yr.csv"), "Last Price", "JPY_Daily_5yr")
    # note this says JPY is O(0) stationary as in, it is stationary as is?

    GER_US_5yr = PairAnalysis(GER_5yr, US_5yr)
    GB_US_5yr = PairAnalysis(GB_5yr, US_5yr)
    # not even co-integrated

main()