#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 21:57:49 2018

Reference: 
1. https://github.com/markharley/Markowitz/blob/master/markowitz.py
2. https://github.com/psthomas/efficient-frontier/blob/master/efficient_frontier_final.py

@author: Alex Lau
"""
import numpy as np
from numpy.linalg import solve, inv
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
# Abupy for data extraction of HK stock market
import abupy


def get_return_matrix(choice_symbols, n_year):
    kl = abupy.ABuSymbolPd.make_kl_df(choice_symbols, n_folds=2)
    return_mat = []
    for symbol in kl:
        pre_close = np.array(kl[symbol].pre_close, dtype = np.float64)
        return_vec = get_return(pre_close)
        return_mat.append(return_vec)
    return_mat = np.matrix(return_mat, dtype = np.float64)
    np.nan_to_num(return_mat, copy = False)
    return return_mat

# Reference: https://github.com/markharley/Markowitz/blob/master/markowitz.py
def efficient_frontier(returns, short_sell = False):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 200
    qs = [10**(-5.0 * t/N + 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))

    # np.mean computes vector of means along
    # as mean of each row
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    # -- documentation at http://cvxopt.org/userguide/coneprog.html#quadratic-programming
    if short_sell == True:
        G = -opt.matrix(0.0, (n, n)) # negative nxn id
    else:
        G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n, 1)) # n x 1 zero vector
    A = opt.matrix(1.0, (1, n)) # 1 x n 1-vector
    b = opt.matrix(1.0) # [1.0]

    ## Calculate efficient frontier weights using quadratic programming
    # q is a measure of risk tolerance
    portfolios = [solvers.qp(S, -q*pbar, G, h, A, b)['x'] for q in qs]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def convert_portfolios(portfolios):
    ''' Takes in a cvxopt matrix of portfolios, returns list '''
    port_list = []
    for portfolio in portfolios:
        temp = np.array(portfolio).T
        port_list.append(temp[0].tolist())
        
    return port_list

def covmean_portfolio(covariances, mean_returns):
    ''' returns an optimal portfolio given a covariance matrix and matrix of mean returns '''
    n = len(mean_returns)
    
    N = 200
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    S = opt.matrix(covariances)  # how to convert array to matrix?  

    pbar = opt.matrix(mean_returns)  # how to convert array to matrix?

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    port_list = convert_portfolios(portfolios)
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    frontier_returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios] 
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(frontier_returns, risks, 2)
    #print m1 # result: [ 159.38531535   -3.32476303    0.4910851 ]
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  

    return np.asarray(wt), frontier_returns, risks, port_list
