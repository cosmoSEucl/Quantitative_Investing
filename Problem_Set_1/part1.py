import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
### Part 1: 
## Question A 
# Load the data

stocks = pd.read_excel('/Users/comecosmolabautiere/Desktop/Yale/Fall Semester/Fall 1/Quant Investing/Week 1/Problem_Set1_2025.xlsx', sheet_name='stock_return', header=5)
stocks['date'] = pd.to_datetime(stocks['date'], format='%Y%m')
stocks.set_index('date', inplace=True)
print(stocks.head(5))

# Data cleanign
print(stocks.isnull().sum())

''' Using the file "Problem_Set1_2023.xls ", form equal-weight portfolios using the first 5, first 10, first 25, 
and all 50 stocks. Calculate the sample mean and standard deviation of returns for each of the four equal-weight portfolios. 
Plot estimated standard deviations as a function of the number of stocks in the equal-weight portfolio. Comment on the shape of the function. 
Are the results consistent with what you would expect theoretically? Eye-balling the graph, does it look like adding more and more stocks will diversify away 
all of the standard deviation? Why or why not?'''

def equal_weight_portfolio(stocks, n):
    # Select the first n stocks
    selected_stocks = stocks.iloc[:, :n]
    # Calculate the equal weight portfolio return
    mean_returns = selected_stocks.mean(axis=1)
    cov_matrix = selected_stocks.cov()
    weights = np.ones(n)/n
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    std_dev = np.sqrt(variance)
    return mean_returns.mean(), std_dev, variance,  cov_matrix, weights


portfolio_sizes = [5, 10, 25, 50]
results = []
for size in portfolio_sizes: 
    mean_return, std_dev, variance, cov_matrix, weights = equal_weight_portfolio(stocks, size)
    results.append((size, mean_return, std_dev))
    results_df = pd.DataFrame(results, columns = ['Number of Stocks', 'Mean Return', 'Standard Deviation'])
print(results_df)

# Plotting the results 
plt.figure(figsize=(10,6))
sns.lineplot(data=results_df, x='Number of Stocks', y='Standard Deviation')
plt.title('Standard Deviation of Equal-Weight Portfolios')
plt.xlabel('Number of Stocks')
plt.ylabel('Standard Deviation')
plt.grid()
plt.show()

''' Comments on the shape of the functions: The curve is downward slopping which is what you would expect theoretically as each stock added should reduce the overall risk of the portfolio.
However, the curve seems to be flattening as the number of stocks in the portfolio  increases, indicating that adding more stocks will not diversify away all of the standard deviation. 
This can be due to the fact that as the stocks are not perfectly uncorrelated, there is some systematic risk that cannot be diversified away'''

## Question B 
''' For all four equal-weight portfolios, decompose the estimated portfolio variance
into its two components (the contributions of variances and covariances). Hint:
you do not have to estimate the pairwise covariances in order to compute the
decomposition. Plot the percentage of the portfolio’s variance due to the variances
of individual security returns as a function of the number of stocks in the
portfolio. Comment on the shape of the function. Are the results consistent with
what you would expect theoretically?'''

decomposition_results = []
for size in portfolio_sizes:
    mean_return, std_dev, variance, cov_matrix, weights = equal_weight_portfolio(stocks, size)
    # Contribution of variances (individual stock variances)
    var_contribution = np.sum(np.diag(cov_matrix)) / size**2
    # Convert to percentage of total portfolio variance
    var_contribution_pct = (var_contribution / variance) * 100
    decomposition_results.append((size, var_contribution_pct))

decomposition_df = pd.DataFrame(decomposition_results, columns=['Number of Stocks', 'Stock Variance as % of Total Variance'])
print(decomposition_df)

plt.figure(figsize=(10,6))
sns.lineplot(data=decomposition_df, x='Number of Stocks', y='Stock Variance as % of Total Variance')
plt.title('Individual Stock Variance Contribution to Portfolio Variance')
plt.xlabel('Number of Stocks')
plt.ylabel('Stock Variance as % of Total Variance')
plt.grid()
plt.show()

## Question C 
'''Suppose instead of equal-weighted portfolios, we computed value-weighted 
(e.g., weighted by market capitalization) portfolios. Would you expect the value-weighted portfolios
 to exhibit more or less variance relative to equal-weighted portfolios? What might it depend on? 
 (Do not make any calculations here, just answer what it will depend on.)'''

''' It depends, if the larger market cap stocks are more volatile with high idiosyncratic risk, 
then the value weighted portfolio would exhibit more variance. On the other hand, if the larger market cap
stocks are more stable and less volatile, given their higher weights, the value weighted portfolio would show less variance.'''

## Question D 
''' Compute the test statistics for whether the mean return of each of the four equal- weight portfolios you calculated in part
 b) is different from zero. What statistical distribution do these test statistics follow? 
 Do you reject or fail to reject the null hypothesis that each of the mean returns on the portfolios is different from zero?'''
t_statistic, p_value = stats.ttest_1samp(results_df['Mean Return'], 0)
print(f"T-statistic: {t_statistic}, P-value: {p_value}")