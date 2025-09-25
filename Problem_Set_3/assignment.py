import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

## Part 1: For the 49 industry portfolio spreadsheet

# Load the data up to column 'Other'
def load_data():
    industries = pd.read_excel('data/Problem_Set3.xlsx', sheet_name = '49_Industry_Portfolios', header=6)
    industries = industries.loc[:, :'Other']
    market_proxy = pd.read_excel('data/Problem_Set3.xlsx', sheet_name='Market_proxy', header=5)
    return industries, market_proxy

industries, market_proxy = load_data()

# Clean the data: 
def clean_data(df, replacement):
    df = df.replace([-99.99, -999], replacement)
    df['date'] = pd.to_datetime(df.iloc[:,0], format='%Y%m')
    df.set_index('date', inplace=True)
    df = df.iloc[:,1:]
    df = df.dropna(how='all')
    return df

industries = clean_data(industries, 0)
market_proxy = clean_data(market_proxy, 0)

print(industries.head(-5))
print(market_proxy.head(-5))

## Question A
'''Consider the cross-sectional regression,
(1) Ri = γ0 + γM βiM + ηi,
where γ0 and γM are regression parameters and βiM = cov(Ri, RM)/σ2(RM).
If the CAPM holds, then what should γ0 and γM equal (for both the Sharpe/Lintner and Black versions)?
'''

# Answer
''' We know that the difference between the Sharpe-Linter and Black versions of CAPM is that Sharpe-Linter
assumes that investors can borrow and lend at the risk-free rate, while Black assumes that they cannot.
That being said, if the CAPM holds then γ0 = rf (the risk free rate) and γM = (E[RM] - rf).'''

## Question B 
''' Estimate γ0 and γM using the approach pioneered by Fama and MacBeth.'''
'''1) This means we need to run a time series regression of each industry's excess return
2) Then we will use the estimated betas'''

# Lets start with the time series regression of each industry's excess return 
results = []
for portfolios in industries.columns: 
    Y = industries[portfolios] - market_proxy['RF']
    X = market_proxy['Mkt-RF']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    alphas = model.params.iloc[0]
    beta = model.params.iloc[1]
    results.append((portfolios, beta, alphas))

results_df = pd.DataFrame(results, columns = ['Industry', 'Beta', 'Alpha'])
print(results_df)

# Now lets run the cross-sectional regression of the average excess returns on the estimated betas 

xw
Y_cs = industries.mean() - market_proxy['RF'].mean()
X_cs = results_df['Beta']
X_cs = sm.add_constant(X_cs)
model_cs = sm.OLS(Y_cs, X_cs).fit()
