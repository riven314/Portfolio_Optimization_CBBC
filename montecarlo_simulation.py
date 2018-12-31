#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 22:00:37 2018

@author: Alex Lau
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abupy
from collections import defaultdict

FUND = 0.023
RATIO = 1/100
UNIT = 10000

# symbol_list = ['hk00700', 'hk00066', 'hk00016']
# mean_2017, cov_mat_2017 = Get_MeanCovariance(symbol_list, yr_n = 1, time_horizon = 'monthly', end_date = '2017-12-31')
def Get_MeanCovariance(symbol_list, yr_n, time_horizon, end_date):
    hist_data = get_stock_hist(symbol_list, yr_n, end_date = end_date)
    cbbc_info = get_CBBC_info(symbol_list, hist_data, time_horizon = time_horizon)
    sim_df = MonteCarlo_Sim(hist_data, cbbc_info, time_horizon = time_horizon)
    #mean = sim_df.mean()
    #cov_mat = sim_df.cov()
    return sim_df

def get_cost(spot_price, strike, position, fund = FUND, ratio = RATIO, unit = UNIT):
    if position == 'long':
        init_cost = ((spot_price - strike) * ratio + fund) * unit
        return init_cost
    elif position == 'short':
        init_cost = ((strike - spot_price) * ratio + fund) * unit
        return init_cost

def get_stock_hist(symbol_list, yr_n, end_date):
    hist_data = abupy.ABuSymbolPd.make_kl_df(symbol_list, n_folds= yr_n, end = end_date)
    return hist_data

def get_CBBC_info(symbol_list, hist_data, time_horizon):
    cbbc_info = defaultdict()
    
    # For weekly return, set strike to be +-3% from spot, call level to be +-0.8% from strike
    if time_horizon == 'weekly':
        strike_load = 0.03
        call_load = 0.008
    # For monthly return, set strike to be +-12% from spot, call level to be +-3.2% from strike
    elif time_horizon == 'monthly':
        strike_load = 0.12
        call_load = 0.032
    
    for symbol in symbol_list:
        sub_info = defaultdict()    
        # Create bull contracts spec
        # Use the latest close price as spot price
        spot_price = hist_data[symbol].iloc[-1][0]
        lng_strike = spot_price * (1 - strike_load)
        lng_call = lng_strike * (1 + call_load)
        lng_cost = get_cost(spot_price, lng_strike, 'long')
        lng_key = 'bull_' + time_horizon
        sub_info[lng_key] = {'spot': spot_price,  'strike': lng_strike, 'call': lng_call, 'init_cost': lng_cost}
        # Create bear contracts spec
        sht_strike = spot_price * (1 + strike_load)
        sht_call = sht_strike * (1 - call_load)
        sht_cost = get_cost(spot_price, sht_strike, 'short')
        sht_key = 'bear_' + time_horizon
        sub_info[sht_key] = {'spot': spot_price,  'strike': sht_strike, 'call': sht_call, 'init_cost': sht_cost}
        
        cbbc_info[symbol] = sub_info
    return cbbc_info

def Stocks_Sim(hist_data, time_horizon):
    if time_horizon == 'weekly':
        period = 5
    elif time_horizon == 'monthly':
        period = 20
    # Get cumulative return
    symbol_list = list(hist_data.items)
    # Set up a dataframe with rows = time series return, columns = stock index
    ret_df = pd.DataFrame({symbol:hist_data[symbol]['p_change'] for symbol in symbol_list})
    # Imput ret_df by its mean
    ret_df = ret_df.fillna(ret_df.mean())
    ret_df = 1 + (ret_df/ 100)
    # Bootstrap resampling 
    rand_ret_df = ret_df.sample(period)
    # Get time-series simulated cumulative return for all stocks
    rand_cum_ret_df = rand_ret_df.cumprod()
    rand_cum_price_df = rand_cum_ret_df.copy()
    # Convert to net return in %
    rand_cum_ret_df = (rand_cum_ret_df - 1) * 100
    for symbol in symbol_list:
        # Get time-series simulated stock price for all stocks
        # Use the latest close prices * simulated cumulative returns
        spot_price = hist_data[symbol].iloc[-1][0] 
        rand_cum_price_df[symbol] = spot_price * rand_cum_price_df[symbol]
        #print('{} spot price = {}'.format(symbol, spot_price))
    return rand_cum_ret_df, rand_cum_price_df

def CBBC_End_Return(rand_cum_df, cbbc_info):
    cbbc_keys = list(cbbc_info.keys())
    cbbc_ret_dict = dict()
    for symbol in cbbc_keys:
        for product in cbbc_info[symbol].keys():
            # Get simulated payoff for bull contracts at the end
            if 'bull' in product:
                # If call level is touched, set payoff = 0
                if np.sum(rand_cum_df[symbol] <= cbbc_info[symbol][product]['call']):
                    payoff = 0
                else:
                    ult_price = rand_cum_df[symbol][-1]
                    strike = cbbc_info[symbol][product]['strike']
                    mult = RATIO * UNIT
                    payoff = (ult_price - strike) * mult
            # Get simulated payoff for bear contracts at the end
            elif 'bear' in product:
                # if call level is touched, set payoff = 0
                if np.sum(rand_cum_df[symbol] >= cbbc_info[symbol][product]['call']):
                    payoff = 0
                else:
                    ult_price = rand_cum_df[symbol][-1]
                    strike = cbbc_info[symbol][product]['strike']
                    mult = RATIO * UNIT
                    payoff = (strike - ult_price) * mult
            # Get simulated return at the end
            ret = (payoff / cbbc_info[symbol][product]['init_cost'] - 1) * 100
            name = symbol + '_' + product
            cbbc_ret_dict[name] = ret
    return cbbc_ret_dict

# By default simulation number = 5000
# 1 row = a one-period ultimate return
def MonteCarlo_Sim(hist_data, cbbc_info, time_horizon, sim_n = 5000):
    rows = dict()
    for i in range(sim_n):
        rand_cum_ret_df, rand_cum_price_df = Stocks_Sim(hist_data, time_horizon = time_horizon)
        cbbc_ret_dict = CBBC_End_Return(rand_cum_price_df, cbbc_info)
        # Embed stock ultimate return on cbbc_ret_dict
        symbol_list = list(hist_data)
        for symbol in symbol_list:
            cbbc_ret_dict[symbol] = rand_cum_ret_df[symbol][-1]
        sim_key = 'sim_' + str(i)
        rows[sim_key] = cbbc_ret_dict
    sim_df = pd.DataFrame.from_dict(rows, orient='index')
    sim_df = sim_df.reindex_axis(sorted(sim_df), axis=1)
    return sim_df

