# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics
from pathlib import Path  

# Functions used in the seminar
import pypfopt_FEseminar
from pypfopt_FEseminar import risk_models
from pypfopt_FEseminar import plotting
from pypfopt_FEseminar import EfficientFrontier
from pypfopt_FEseminar import DiscreteAllocation 
from pypfopt_FEseminar import base_optimizer
from pypfopt_FEseminar import expected_returns
np.seterr(invalid='ignore')

def normalize_weights(discrete_alloc, latest_prices):
    """"Utility function used to transform discrete allocation weights back so that weights sum to 1"""
    
    alloc_portfolio_val = 0 # calculat the portfolio value of the allocation (not equal to budget because of potential leftover)
    for ticker, num in discrete_alloc.items():
        alloc_portfolio_val += num * latest_prices[ticker]
    
    for ticker in discrete_alloc:
        discrete_alloc[ticker] = float(discrete_alloc[ticker]*latest_prices[ticker]/alloc_portfolio_val)
        
    weights = discrete_alloc
    return weights


def discrete_allocations(weights, P, P_latest, S, B, SR):
    """ Finds discrete alocation given desired weights, prices and a budget constraint """
    # Calculated discrete allocations - budget constrainded portfolio
    ###latest_prices = P_latest  # prices as of the day you are allocating, counting backwards from today
    ###da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=B, short_ratio=SR)
    
    da = DiscreteAllocation(weights, P_latest, total_portfolio_value=B, short_ratio=SR)
    alloc, leftover, rmse = da.lp_portfolio(verbose=False)
    
    #print(f"Discrete allocation performed with ${leftover:.2f} leftover")
    
    # Number of assets with non zero weight
    n_assets = sum(float(i) >= 0.0001 for i in alloc.values())

    # Evaluate performance of discrete allocation
    ###alloc_normalized = normalize_weights(alloc, latest_prices) # Transform weights to sum to 1 for proper evaluation
    
    
    alloc_normalized = normalize_weights(alloc, P_latest) # Transform weights to sum to 1 for proper evaluation
    
    
    alloc_performance = base_optimizer.portfolio_performance(weights = alloc_normalized, expected_returns = None, cov_matrix = S, verbose = False)
    annual_volatility = alloc_performance[1]*100 # annual volatility of the discrete portfolio in pct
    return alloc, leftover, rmse, annual_volatility, n_assets


def evaluate_discrete_allocations(budget_array, weights, P, P_latest, S, SR, unconstrained_vol):
    """ Evaluates the discrete allocations for a a range of budget sizes """
    # Empty arrays for results
    annual_vols_array = []
    n_assets_array = []
    rmse_array = []
    risk_deviation_array = []
    
    # Loop over budgets and gather results
    for B in budget_array:
        
        alloc, leftover, rmse, annual_volatility, n_assets = discrete_allocations(
            weights, P, P_latest, S, B, SR)

        annual_vols_array.append(annual_volatility)
        n_assets_array.append(n_assets)
        rmse_array.append(rmse)

    # Gather deviations in discretized portfolio volatility from unconstrained volatility
    for vol in annual_vols_array:
        risk_deviation = (vol / unconstrained_vol - 1)*100 #vol - unconstrained_vol #
        risk_deviation_array.append(risk_deviation)

    return annual_vols_array, n_assets_array, rmse_array, risk_deviation_array


def full_sim(nstocks, ncountries, risk_method, countrycds, exchrates, pf, start_dates, end_dates, budget_array,
            WB, SR, rf, inflation):
    """ Runs the estimations for the select countries given the number of stocks (top N by market cap),
    risk method, threshold)
    Returns are saved to CSV 
    file path = '{pf}/{method}/{df}_RESULTS_{countrycd}_TOP_{nstocks}.csv'
    df = RDA, NAA, AVA, RMSE, P_Latest
    - RDA = risk deviation array - allocation std.dev deviation from threshold (relative to MVP)
    - NAA = number of assets array - number of assets in allocation
    - AVA = annual vols array - standard deviation of allocation
    - RMSE = root mean squared error array - RMSE of allocations vs MVP
    - P_Latest = average price of stocks in that year"""

    # Chose covariance matrix estimator
    method = risk_method

    # Simulation countries and corresponding exch. rate to loop over:
    pars_ = zip(countrycds,exchrates)

    # loop over parameters
    for par in pars_:    

        # List of estimation windows
        start_date = start_dates
        end_date = end_dates
        estimation_windows = zip(start_dates, end_dates)

        # Set country and exchange rate
        countrycd = par[0]
        forex = par[1]

        # For gathering results
        annual_vols_array = np.zeros([len(budget_array),len(end_date)])
        n_assets_array = np.zeros([len(budget_array),len(end_date)])
        rmse_array = np.zeros([len(budget_array),len(end_date)])
        risk_deviation_array = np.zeros([len(budget_array),len(end_date)])
        p_latest_avg = np.zeros([len(budget_array),len(end_date)])

        # Read dataset for country
        data_ = pd.read_csv (f'{pf}/{countrycd}.csv')
        data_WIC = pd.read_csv (f'{pf}/WIC_{countrycd}.csv')
        data_['date'] =  pd.to_datetime(data_['date'], format='%Y-%m-%d')
        data_WIC['junedate'] =  pd.to_datetime(data_WIC['junedate'], format='%Y-%m-%d')

        # Loop over estimation windows
        for i, est_window in enumerate(estimation_windows):
            #try:
            # Estimation_window 
            start = pd.to_datetime(est_window[0])
            end = pd.to_datetime(est_window[1])
            
            # Adjust budget for inflation:
            b = budget_array.copy()
            budget_array_infadj = np.multiply(b,(1+inflation/100)**i)
            

            # Status message
            #print(f'Country: {countrycd}, Iter:{i}, estimation window:{start} to {end}') 

            P = data_.copy() # Copy of data
            P["gvkey"]=P["gvkey"].apply(str)
            P = P[P.date >= start] # Select start date
            P = P[P.date < end] # Select end date

            # 2. List of companies by market cap within year
            P_mktcap = P.copy()
            P_mktcap = P_mktcap.drop(columns=['cshoc', 'adjclose'])
            P_mktcap = P_mktcap.pivot_table('mktcap', 'date', 'gvkey') 
            P_mktcap = P_mktcap.fillna(method='ffill')
            P_mktcap = P_mktcap.iloc[:1] # Select first row
            P_mktcap = P_mktcap.dropna(axis=1, how='any').reset_index().drop(columns=['date'])
            P_mktcap = P_mktcap.sort_values(by=P_mktcap.index[0],ascending=False, axis=1) # sort by Market CAP
            P_mktcap = pd.DataFrame(P_mktcap.columns)

            # 3. Pivot data 
            P = P.pivot_table('adjclose', 'date', 'gvkey') 

            # 5. Fill NA prices with price from previous trading day
            P = P.fillna(method='ffill')
            P = P.multiply(1/forex) #convert to usd and filter outliers
            P = P[P<5000]

            # 6. Drop missing observations
            P = P.dropna(axis=0, how='all')
            P = P.dropna(axis=1, how='any')

            # 7. Select top N gvkey by market cap
            P = P.drop(columns=[col for col in P if col not in P_mktcap['gvkey'][:nstocks].values]) # Drop columns not in top N companies by market cap

            # 8. Latest prices - Transform to USD
            P_latest = P.iloc[-1]
            p_latest_avg[:,i]= statistics.mean(P_latest) # Save for descriptive stats

            # 10. Calculate the covariance matrix, looping over risk methods
            S = risk_models.risk_matrix(P, method=method)

            mu = None # expected_returns.mean_historical_return(P) # Returns, not required. Used for sanity check and later in efficient risk analysis

            # 11. Unconstrained portfolio weights
            ef = EfficientFrontier(None, S, weight_bounds=WB)
            ef.min_volatility()
            weights = ef.clean_weights()

            # 12. Evaluate performance of Minimum Variance portfolio and save the result
            gmv = base_optimizer.portfolio_performance(weights, mu, S, verbose=False)
            gmv_vol = gmv[1]*100 # annual volatility of the unconstrained portfolio in pct

            # 13. Constrained portfolios: Loop over different budgets
            annual_vols_array[:,i], n_assets_array[:,i], rmse_array[:,i], risk_deviation_array[:,i] = evaluate_discrete_allocations(
            budget_array, weights, P, P_latest, S, SR, gmv_vol)

        # Format and save results so we dont have to rerun sims
        # risk_deviation_array
        df = pd.DataFrame(risk_deviation_array)
        df = df.set_axis(end_date, axis=1, inplace=False)  # set column names
        filepath = Path(f'{pf}/{method}/RDA_RESULTS_{countrycd}_TOP_{nstocks}.csv') # Set name
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath, index=False)

        # n_assets_array
        df1 = pd.DataFrame(n_assets_array)
        df1 = df1.set_axis(end_date, axis=1, inplace=False)
        filepath = Path(f'{pf}/{method}/NAA_RESULTS_{countrycd}_TOP_{nstocks}.csv') # Set name
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df1.to_csv(filepath, index=False)

        # annual_vols_array
        df2 = pd.DataFrame(annual_vols_array)
        df2 = df2.set_axis(end_date, axis=1, inplace=False)
        filepath = Path(f'{pf}/{method}/AVA_RESULTS_{countrycd}_TOP_{nstocks}.csv') # Set name
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df2.to_csv(filepath, index=False)

        # rmse_array
        df3 = pd.DataFrame(rmse_array)
        df3 = df3.set_axis(end_date, axis=1, inplace=False)
        filepath = Path(f'{pf}/{method}/RMSE_RESULTS_{countrycd}_TOP_{nstocks}.csv') # Set name
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df3.to_csv(filepath, index=False)

        df4 = pd.DataFrame(p_latest_avg[0],index=end_date, columns=['AVG'])
        filepath = Path(f'{pf}/{method}/TABLE/P_LATEST_{countrycd}_TOP_{nstocks}.csv') # Set name
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df4.to_csv(filepath, index=True)

        print(f'{method}, country:{countrycd}, nstocks:{nstocks} - Simulation done, results saved')


def mrb_index(df, threshold, countrycd):
    " Utility function, used for plotting the minimum required budget"
    min_reg_budget_index = []
    for col in df.columns:
        try:
            x = df[df[col] < threshold].index[0] # Alternative: a = np.min(np.where(df[col].lt(threshold))[0])
            min_reg_budget_index.append(x)
        except IndexError:
            #print(f'Indexerror, {countrycd}, {col}')
            x = 3
            min_reg_budget_index.append(x)
            continue
    return min_reg_budget_index
    