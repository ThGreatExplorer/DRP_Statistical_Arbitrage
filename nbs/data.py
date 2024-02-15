# %%
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import matplotlib.pyplot as plt


# %%
def clean(location):
    """
    function that will clean a dataframe according to
    a file location 
    """
    def read(location):
        """ read the csv and error out if it fails """
        try:
            df = pd.read_csv(location)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except:
            print(f'file not found at {location}')
            return
        
    def subset_columns(dataframe, select = ['Date', 'PX_LAST']):
        """ removes irrelevant columns imported from bloomberg """
        if ('PX_LAST' not in dataframe.columns):
            select = ['Date', 'Last Price']
        df = dataframe.loc[:, select]
        return df
    
    def initialize_columns(dataframe):
        """ renames columns and sets time as index """
        df = dataframe.copy()
        df.columns = ['Date', 'Value']
        df.set_index('Date', inplace = True)
        return df

    
    return initialize_columns(subset_columns(read(location)))

# %%
def get_value_data(files, columns):
    """
    calling this will get all the csv in a directory, read and attempt to 
    clean them all, then outer join them along their dates, providing the final
    values themselves.
    """
    dfs = [clean(f) for f in files]
    df = pd.concat(dfs, axis = 1, join='outer').sort_index()
    df.columns = columns
    return df

def to_stationary(series):
    """ custom pct change / difference function according to yield / total return """
    if max(series.dropna()) < 10:
        return (series - series.shift(1)) / 100
    else:
        return (series / series.shift(1)) - 1

# %%
class Data:
    
    def __init__(self, value_data):
        self.data = value_data

    def get_ffill_data(self):
        return self.data.fillna(method = 'ffill').dropna()
        
    def get_pct_return(self):
        return self.data / self.data.shift(-1) - 1
    
    def get_standard_return(self):
        return self.data.apply(to_stationary, axis = 0)

    def get_true_return(self):
        filled = self.data.fillna(method = 'ffill')
        return filled.apply(to_stationary, axis = 0)
                
    def __mle_normal_params__(series):
        mu = np.mean(series.dropna())
        var = np.var(series.dropna(), ddof=0)
        return mu, var

    def __bounded_normal__(mu, var):
        sd = np.sqrt(var)
        sample =  np.random.normal(mu, sd)
        return min(sample, mu + 1.5*sd) if sample > 0 else max(sample, mu - 1.5*sd)

    def get_mle_interpolate_return(self):
        def interpolate(series):
            mu, var = mle_normal_params(series)
            return np.where(series.isna(),  np.random.normal(mu, np.sqrt(var)), series)
        return self.data.apply(interpolate, axis = 0)

    def get_mle_interpolate_bounded_return(self):
        def interpolate(series):
            mu, var = mle_normal_params(series)
            return np.where(series.isna(),  np.random.normal(mu, np.sqrt(var)), series)
        return self.data.apply(interpolate, axis = 0)

# %%
def initialize_data(**series) -> Data:
    """ retrieves csv files and intializes variables for time series processing """
    directory = r'../data/final_csv'
    files = [f'{directory}/{f}' for f in os.listdir(directory) if f in series.keys() and os.path.isfile(os.path.join(directory, f))]
    columns = series.values()
    value_data = get_value_data(files, columns)
    return Data(value_data.dropna())

# %%
def find_coint(y0, y1, alpha=0.05):
    """
    y0 and y1 are the array-like 1d elements to compare for cointegration. Assuming y0 and y1 are I(1) stationary.

    Parameters:
    - y0: Array-like, 1d
    - y1: Array-like, 1d
    - alpha: the significance level is set at 0.5
    """
    # Fit OLS model
    model = sm.OLS(y0, y1)
    results = model.fit()
    
    # Get the estimated coefficient (beta) and standard error
    beta = results.params.iloc[0]
    std_err = results.bse.iloc[0]

    
    # Perform cointegration test
    _, p_value, _ = coint(y0, y1)

    print(p_value)
    
    # Check if there's cointegration
    cointegrated = p_value < alpha
    
    # Calculate the range
    lower_bound = beta - std_err
    upper_bound = beta + std_err
    
    return cointegrated, (lower_bound,beta, beta, upper_bound)


class PairTradingTest:
    """
    Fields:

    - coint_timeframe: represents the past x days which the two assets will be checked for cointegration. Updated whenever a trading signal is given

    - beta_vals: the cointegration relationship between the two assets which we will assume holds until a new trading signal is given. The starting beta will be intialized as the beta of the first x datapoints (where x = coint_timeframe) with cointegration relationship. 
        - stored as (lower_bound, beta, upper_bound)

    - beta_history: the history of past beta values over the timeframe of the backtest 
    - trading_signals_history: where -1 and 1 is sell, None and None is exit, and 1 and 1 is buy
    - data: the processed timeseries data
    - test_idx: the index for the train/test split for the starting beta value
    - threshold: the threshold for trading signals (i.e. if the spread is + or - within this standard deviation)
    - portfolio: dataframe containing position sizes of the portfolio for each asset over time, the default is 1 million
    - asset1_name: name of asset1
    - asset2_name: name of asset2
    """

    def __init__(self, data :Data, coint_timeframe :int, threshold=1, portfolio=1000000) -> None:
        """
        Given the processed data of two time series which are cointegrated intialize an instance of this class for testing the trading strategy.
        """

        if data.data.shape[1] != 2:
            raise ValueError(f"Data must be processed to have two Time Series for each Asset in the Pair {data.data}")
        self.coint_timeframe = coint_timeframe
        self.beta_vals = None
        self.data = data
        i = 1
        while (True):
            if (i > 20 or i * self.coint_timeframe > len(self.data.data)):
                raise ValueError("Can't find any cointegration relationship between the two time series")
            # intialize the cointegration relationship as some starting beta based on cointegration of the first y days where y = i * coint_timeframes
            cointegrated, beta_vals = find_coint(
                self.data.data.iloc[:i * self.coint_timeframe,0], 
                self.data.data.iloc[:i * self.coint_timeframe,1])
            if (cointegrated):
                # split the dataset into train/test
                self.beta_vals = beta_vals
                self.test_idx = i * self.coint_timeframe
                break
            i+=1
        self.threshold = threshold
        self.beta_history = []
        self.beta_history.append(self.beta_vals[1])
        self.asset1_name = data.data.iloc[:,0].name
        self.asset2_name = data.data.iloc[:,1].name

         # Initialize trading signals history DataFrame with pre-allocated memory
        """
        signals_columns = ['Date', 'Signal']
        signals_data = np.zeros((len(data.data), len(signals_columns)), dtype=object)
        signals_data[:, 0] = data.data.index
        self.trading_signals_history = pd.DataFrame(signals_data, columns=signals_columns)
        self.trading_signals_history.set_index('Date', inplace=True)
        """
        self.portfolio = []  # List to hold portfolio data
        self.trading_signals_history = []  # List to hold trading signals data

        # Calculate initial positions based on the specified beta
        initial_asset2_position = portfolio / (1 + self.beta_vals[1])  # Allocate asset2 with the remaining portfolio
        initial_asset1_position = self.beta_vals[1] * initial_asset2_position  # Calculate asset1 position based on beta

        # Allocate initial portfolio
        initial_asset2_position = portfolio / (1 + self.beta_vals[1])  # Allocate asset2 with the remaining portfolio
        initial_asset1_position = self.beta_vals[1] * initial_asset2_position  # Calculate asset1 position based on beta
        self.portfolio.append([data.data.index[self.test_idx - 1], initial_asset1_position, initial_asset2_position, 0])

        """
        portfolio_columns = ['Date', f'{self.asset1_name}_Position', f'{self.asset2_name}_Position', 'Cash_Position']
        portfolio_data = np.zeros((len(data.data) - self.test_idx + 1, len(portfolio_columns)), dtype=float)
        portfolio_data[:, 0] = data.data.index[self.test_idx - 1:]
        self.portfolio = pd.DataFrame(portfolio_data, columns=portfolio_columns)
        self.portfolio['Date'] = pd.to_datetime(self.portfolio['Date'])
        self.portfolio.set_index('Date', inplace=True)
        

        # allocate intial portfolio
        self.portfolio.loc[self.portfolio.index[0], f'{self.asset1_name}_Position'] = initial_asset1_position
        self.portfolio.loc[self.portfolio.index[0], f'{self.asset2_name}_Position'] = initial_asset2_position
        self.portfolio.loc[self.portfolio.index[0], 'Cash_Position'] = 0
        """
    
    def signal_generation(self, past_data) -> int:
        """
        Checks for a trading signal to adjust the portfolio Called at every timestep. 

        Params:
        - past_data, the past coint_timeframe days of data

        Calls `find_coint` from on `past_data` and checks first for cointegration. If no cointegration is found, then the portfolio is "frozen" until cointregation is found again at one of the future timesteps.

        If cointegration is found, then checks if the beta is within the range of 1 std. error of the beta. If the beta is not within range, then update the beta with the new variable and range.

        Finally, generates a trading signal 1 (buy), -1 (sell), 0 (hold), None (Liquidate positions) based on the standard deviation of the spread.
        """

        cointegrated, beta_vals = find_coint(past_data.iloc[:,0], past_data.iloc[:,1])
        
        if not cointegrated:
            # set the beta to 0 representing that the past relationship doesn't hold
            self.beta_vals = (0,0,0)
            # return none
            return None,None
        else:
            # if the self.beta_vals was previously set to 0, meaning cointegration failed, then set a new cointegration relationship
            if self.beta_vals[1] == 0:
                self.beta_vals = beta_vals
                self.beta_history.append(self.beta_vals[1])

            # not within the same beta bounds, update history and beta, or 
            if not (self.beta_vals[0] <= beta_vals[1] <= self.beta_vals[2]):
                self.beta_vals = beta_vals
                self.beta_history.append(self.beta_vals[1])

            # generate trading signal
            # spread = asset1 - beta * asset2
            spread = past_data.iloc[:, 0] - self.beta_vals[1] * past_data.iloc[:, 1]
            
            # Calculate mean and standard deviation of the spread
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            
            # Calculate trading signals for each asset
            if spread.iloc[-1] > spread_mean + self.threshold * spread_std:
                signal_asset1 = -1  
                signal_asset2 = 1   
            elif spread.iloc[-1] < spread_mean - self.threshold * spread_std:
                signal_asset1 = 1   
                signal_asset2 = -1 
            else:
                signal_asset1 = 0   
                signal_asset2 = 0   
            
            return signal_asset1, signal_asset2
    
    def update_portfolio_positions(self, timestep, signal_asset1, signal_asset2):
        """
        Update portfolio positions based on trading signals.

        Parameters:
        - timestep: the current timestep i.e. index in the time series
        - signal_asset1: Trading signal for asset 1 (-1 for sell, 1 for buy, 0 for hold)
        - signal_asset2: Trading signal for asset 2 (-1 for sell, 1 for buy, 0 for hold)
        """

        asset1_last_px = self.data.data.iloc[timestep-1,0]
        asset2_last_px = self.data.data.iloc[timestep-1,1]

        asset1_curr_px = self.data.data.iloc[timestep,0]
        asset2_curr_px = self.data.data.iloc[timestep,1]

        curr_date = self.data.data.index[timestep]
    
        if signal_asset1 == None or signal_asset2 == None:
            entry = [curr_date, 0, 0, self.portfolio[-1][3] + self.portfolio[-1][1] + self.portfolio[-1][2]]
            self.portfolio.append(entry)
            if len(self.trading_signals_history) == 0 or self.trading_signals_history[-1][1] != 'exit':
                # liquidate the portfolio when it's not already been liquidated, i.e the previous signal was also exit since then we never bought or sell and re-entered
                self.trading_signals_history.append([curr_date, 'exit'])

        else:
            # Calculate position sizes for assets based on portfolio value
            asset1_value = self.portfolio[-1][1]
            asset2_value = self.portfolio[-1][2]
            cash_value = self.portfolio[-1][3]


            curr_asset1_value = asset1_value * (1 + asset1_curr_px - asset1_last_px)
            curr_asset2_value = asset2_value * (1 + asset2_curr_px - asset2_last_px)

            #TODO implement some sort of stop signal?

            asset1_position = curr_asset1_value
            asset2_position = curr_asset2_value

            # Now buy and sell based on the beta and based on the trading signals
            if signal_asset1 == 0 and signal_asset2 == 0:
                # hold everything, allocated cash to position 1 and position 2 based on beta
                if cash_value > 0:
                    asset2_allocation = cash_value / (1 + self.beta_vals[1])  
                    asset1_allocation = self.beta_vals[1] * asset2_allocation  
                    asset1_position += asset1_allocation
                    asset2_position += asset2_allocation
                    cash_value = 0

            #TODO same behavior for signals
            elif signal_asset1 == -1 and signal_asset2 == 1:
                total_position = asset1_position + asset2_position + cash_value

                # set cash to 0
                cash_value = 0

                # reallocate portfolio so that asset1 position = beta * asset2 position
                asset2_position = total_position / (1 + self.beta_vals[1]) 
                asset1_position = total_position - asset2_position
                self.trading_signals_history.append([curr_date, 'sell'])
                
            elif signal_asset1 == 1 and signal_asset2 == -1:
                total_position = asset1_position + asset2_position + cash_value

                # set cash to 0
                cash_value = 0

                # reallocate portfolio so that asset1 position = beta * asset2 position
                asset2_position = total_position / (1 + self.beta_vals[1]) 
                asset1_position = total_position - asset2_position
                
                self.trading_signals_history.append([curr_date, 'buy'])

            else:
                raise ValueError(f"Improper signal given: {signal_asset1}, {signal_asset2}")
            
            entry = [curr_date, asset1_position, asset2_position, cash_value]
            self.portfolio.append(entry)


    def __backtest__(self):
        """
        Backtest the pair trading strategy. Converts the portfolio and trading_signals_history fields to dataframes
        """
        #print(self.test_idx)
        for i in range(self.test_idx, len(self.data.data)):
            past_data = self.data.data.iloc[i - self.coint_timeframe:i]
            signal_asset1, signal_asset2 = self.signal_generation(past_data)

            self.update_portfolio_positions(i, signal_asset1, signal_asset2)

        # Convert lists to DataFrames
        columns = ['Date', f'{self.asset1_name}_Position', f'{self.asset2_name}_Position', 'Cash_Position']
        self.portfolio = pd.DataFrame(self.portfolio, columns=columns)
        self.portfolio['Date'] = pd.to_datetime(self.portfolio['Date'])
        self.portfolio.set_index('Date', inplace=True)

        signals_columns = ['Date', 'Signal']
        self.trading_signals_history = pd.DataFrame(self.trading_signals_history, columns=signals_columns)
        self.trading_signals_history['Date'] = pd.to_datetime(self.trading_signals_history['Date'])
        self.trading_signals_history.set_index('Date', inplace=True)

        print(self.portfolio)

    def plot_portfolio_over_time(self):
        """
        Plot the total portfolio size over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio.index, self.portfolio.sum(axis=1), label='Total Portfolio Value', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.title('Total Portfolio Size Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_time_series_with_signals(self):
        """
        Plot the time series data with trading signals marked.
        """

        # Plot the time series data
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.data.index, self.data.data.iloc[:, 0], label=f"{self.asset1_name} Time Series", color='blue')
        plt.plot(self.data.data.index, self.data.data.iloc[:, 1], label=f"{self.asset2_name} Time Series", color='orange')

        # Mark trading signals if any
        for signal in self.trading_signals_history['Signal'].unique():
            if signal == 'buy':
                plt.scatter([], [], color='green', marker='^', label='Buy Signal')
            elif signal == 'sell':
                plt.scatter([], [], color='red', marker='v', label='Sell Signal')
            elif signal == 'exit':
                plt.scatter([], [], color='black', marker='o', label='Exit Signal')

        for date, signal in self.trading_signals_history.iterrows():
            if signal['Signal'] == 'buy':
                plt.scatter(date, self.data.data.loc[date, self.asset1_name], color='green', marker='^')
                plt.scatter(date, self.data.data.loc[date, self.asset2_name], color='green', marker='^')
            elif signal['Signal'] == 'sell':
                plt.scatter(date, self.data.data.loc[date, self.asset1_name], color='red', marker='v')
                plt.scatter(date, self.data.data.loc[date, self.asset2_name], color='red', marker='v')
            elif signal['Signal'] == 'exit':
                plt.scatter(date, self.data.data.loc[date, self.asset1_name], color='black', marker='o')
                plt.scatter(date, self.data.data.loc[date, self.asset2_name], color='black', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Time Series Data with Trading Signals')
        plt.legend()
        plt.grid(True)
        plt.show()

        


# %%
print("Imported function `get_value_data(files, columns)` -> dataframe of values")
print("Imported function `get_value_data(files, columns)` -> dataframe of returns")
print("Imported function `initialize_data()` -> tuple of value and returns data")
print("Imported PairTradingTest")

def main():
    series = {"US_10yr_2000.csv": "US_10yr", "GB_10yr_2000.csv": "GB_10yr"}
    data = initialize_data(**series)
    coint(data.data.iloc[:,0],data.data.iloc[:,1])
    data.data = data.data.iloc[:500]
    strategy :PairTradingTest = PairTradingTest(data,30,1)
    strategy.__backtest__()
    print(strategy.beta_history)
    print(len(strategy.beta_history))
    strategy.plot_portfolio_over_time()
    strategy.plot_time_series_with_signals()

main()