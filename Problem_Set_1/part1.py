import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

### Part 1: 
## Question A 
# Load the data

stocks = pd.read_excel('data/Problem_Set1_2025.xlsx', sheet_name='stock_return', header=5)
stocks['date'] = pd.to_datetime(stocks['date'], format='%Y%m')
stocks.set_index('date', inplace=True)
stocks = stocks / 100

# Data cleaning
print(stocks.isnull().sum())
stocks = stocks.fillna(0)

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
The variance of the portfolio will always be less than the variances of each individual stock, so long as correlation < 1. The risk cannot go to zero. 
Diversification lowers risk, but only down to a positive limit.
This can be due to the fact that as the stocks are not perfectly negatively correlated, there is some systematic risk that cannot be diversified away'''

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
stocks are more stable and less volatile, given their higher weights, the value weighted portfolio would show less variance.
Additionally, the correlation between the stocks also plays a role. If the larger market cap stocks are highly correlated with each other, 
then the value-weighted portfolio may exhibit more variance. '''

## Question D 
''' Compute the test statistics for whether the mean return of each of the four equal- weight portfolios you calculated in part
 b) is different from zero. What statistical distribution do these test statistics follow? 
 Do you reject or fail to reject the null hypothesis that each of the mean returns on the portfolios is different from zero?'''

print("T-test results for each equal-weight portfolio:")
print("=" * 60)

for size in portfolio_sizes:
    # Calculate the portfolio returns time series for this portfolio size
    selected_stocks = stocks.iloc[:, :size]
    portfolio_returns = selected_stocks.mean(axis=1)  # Equal-weight portfolio returns
    
    # Perform one-sample t-test: H0: mean = 0 vs H1: mean ≠ 0
    t_statistic, p_value = stats.ttest_1samp(portfolio_returns, 0)
    
    # Calculate degrees of freedom
    n_observations = len(portfolio_returns)
    degrees_of_freedom = n_observations - 1
    
    # Decision at 5% significance level
    alpha = 0.05
    reject_null = p_value < alpha
    
    print(f"\nPortfolio with {size} stocks:")
    print(f"  Sample size (n): {n_observations}")
    print(f"  Sample mean: {portfolio_returns.mean():.6f}")
    print(f"  T-statistic: {t_statistic:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Degrees of freedom: {degrees_of_freedom}")
    print(f"  Reject H0 at α=0.05: {'Yes' if reject_null else 'No'}")

print(f"\n{'='*60}")

''' These tests follow a t-distribution with (n-1) degrees of freedom, in other words 179.
 Given that the p-values for all portfolios are less than 0.05, then we reject the null hypothesis such that the mean returns
  are different from zero. '''

## Question E 
'''Choose the first stock and test whether its returns follow a normal distribution (hint: compute the sample studentized range, skewness, 
and kurtosis and compare them to what you would expect under a normal distribution. A simple chi-square test can be performed using the sample skewness and kurtosis measures 
to see if the null of normally distributed returns is rejected). Repeat this test for the equal-weighted portfolio of all 50 stocks as well as the market portfolio index of all NYSE, AMEX,
and Nasdaq stocks (which is also included in the spreadsheet).'''

''' We know that, if a distribution is normal, then the skewness (which measures a distribution's asymmetry) should be around 0 and the kurtosis (which measures the tailedness) should be around 3.'''
first_stock = stocks['TXN']
print(first_stock.head(5))

density_plot = sns.kdeplot(first_stock, fill=True)
density_plot.set_title('Density Plot of TXN Returns')
plt.xlabel('Returns')
plt.ylabel('Density')
plt.show()

def check_normality(input_value):
    """
    Checks normality for either a single stock (by ticker name) or an equal-weight portfolio (by number of stocks).
    input_value: str (ticker name) or int (number of stocks in portfolio)
    """
    if isinstance(input_value, str):
        # Single stock by ticker name
        data = stocks[input_value]
        label = f"Stock: {input_value}"
    elif isinstance(input_value, int):
        # Equal-weight portfolio of first 'input_value' stocks
        data = stocks.iloc[:, :input_value].mean(axis=1)
        label = f"Equal-weight portfolio of first {input_value} stocks"
    else:
        raise ValueError("Input must be a ticker name (str) or number of stocks (int).")
    
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data, fisher=False)
    print(f"{label}\nSkewness: {skewness}\nKurtosis: {kurtosis}")

    n = len(data)
    chi_square_stat = (n/6) * skewness**2 + (n/24) * (kurtosis - 3)**2
    p_value = 1 - stats.chi2.cdf(chi_square_stat, df=2)
    print(f"Chi-square statistic: {chi_square_stat}")
    print(f"p-value: {p_value}")

check_normality('TXN'), check_normality(50), check_normality('Market (Value Weighted Index)')

'''For TXN Skewness: -0.182038016679627
Kurtosis: 4.029679276602602. That being said, we can see that the kurtosis is higher than 3, indicating that the distribution has heavier tails than a normal distribution. In addition, 
the skewness is slighty negative indicating a slight left skew. We found similar values of skewness and kurtosis for the equal-weighted portfolio of all 50 stocks and the market portfolio index of all NYSE, AMEX, and Nasdaq stocks.
Furthermore, the p-values for all three tests are less than 0.05, leading us to reject the null hypothesis that the returns are normally distributed.'''

