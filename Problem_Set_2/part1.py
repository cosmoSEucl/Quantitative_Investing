import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

### Part 1: 
'''1. Find the minimum variance and tangency portfolios of the industries. 
(hint:  you will need to compute the means (arithmetic average), standard deviations, 
variances, and covariance matrix of the industries.  The risk-free rate is given in the spreadsheet.)  
Comment on the different weights applied to each industry under the MVP and Tangent portfolios. '''

## Question A 
'''a)	Compute the means and standard deviations of the MVP and Tangent portfolios.  
Plot the efficient frontier of these 10 industries and plot the 10 industries as well on a mean-standard deviation diagram.  
Why does the efficient frontier exhibit the shape that it does (i.e., why is it a parabola)?'''
# Load the data
industry = pd.read_excel('data/Problem_Set2_2025.xlsx',sheet_name = 'Industry_returns', header=20)
print(industry.describe())
industry['date'] = pd.to_datetime(industry.iloc[:,0], format='%Y%m')
industry.set_index('date', inplace=True)
industry = industry.drop(industry.columns[0], axis=1)
industry = industry / 100

''' To compute the means and standard deviations of the MVP and Tangent portfolios,
we first need to calculate variance-covariance V matrix as it is the same all portfolios.
Only the portfolio weights differ. Then we will need the average returns vector as well as the average risk free-rate'''
industry_norf = industry.iloc[:,:-1]
cov_matrix = industry_norf.cov()
means_vector = industry_norf.mean()
risk_free_rate = industry.iloc[:, -1].mean()

# Now lets determine the weights of the MVP and Tangent portfolios

def min_var_portfolio(cov_matrix):
    ones = np.ones(len(cov_matrix))
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    weights_mvp = np.dot(inv_cov_matrix,ones) / np.dot(ones.T, np.dot(inv_cov_matrix, ones))
    return weights_mvp

def tangent_portfolio(cov_matrix, risk_free_rate, means_vector):
    excess_returns = means_vector - risk_free_rate
    ones = np.ones(len(cov_matrix))
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    weights_tan = np.dot(inv_cov_matrix, excess_returns) / np.dot(ones.T, np.dot(inv_cov_matrix, excess_returns))
    return weights_tan

weights_mvp = min_var_portfolio(cov_matrix)
weights_tan = tangent_portfolio(cov_matrix, risk_free_rate, means_vector)

# Lets now compute the means and standard deviations of the MVP and tangent portfolios 

def portfolio_performance(weights, means_vectors, cov_matrix):
    returns = np.dot(weights.T, means_vectors)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

mvp_performance = portfolio_performance(weights_mvp, means_vector, cov_matrix)
tan_performance = portfolio_performance(weights_tan, means_vector, cov_matrix)
print(f'MVP Mean Return: {mvp_performance[0]}, MVP Risk: {mvp_performance[1]}')
print(f'Tangent Mean Return: {tan_performance[0]}, Tangent Risk: {tan_performance[1]}')

# Plotting the efficient frontier 
''' I need to plot the efficient frontier of these 10 industries
and plot the 10 industries as well on a mean-standard deviation diagram. For a given return, 
the efficient frontier is the portfolio with the lowest possible risk (variance).'''
def plot_efficient_frontier(means_vector, cov_matrix, industry_norf, mvp_performance, tan_performance):
    mu = means_vector.values
    V = cov_matrix.values
    ones = np.ones(len(mu))

    # Calculate constants for the analytical frontier
    A = mu.T @ np.linalg.inv(V) @ mu
    B = mu.T @ np.linalg.inv(V) @ ones
    C = ones.T @ np.linalg.inv(V) @ ones
    D = A * C - B**2

    # Range of target returns (extend beyond min/max to show short selling)
    mu_p = np.linspace(mu.min() - (mu.max() - mu.min()) * 0.5, mu.max() + (mu.max() - mu.min()) * 0.5, 300)
    # Corresponding variances (analytical frontier)
    sigma_p = np.sqrt((C * mu_p**2 - 2 * B * mu_p + A) / D)

    plt.figure(figsize=(10, 7))

    # Plot analytical efficient frontier (parabola)
    plt.plot(sigma_p, mu_p, color='b', label='Efficient Frontier')

    # Plot individual industries
    for i, col in enumerate(industry_norf.columns):
        plt.scatter(industry_norf.std()[i], means_vector[i], marker='o', s=100, label=col)

    # Plot MVP and Tangency portfolios
    plt.scatter(mvp_performance[1], mvp_performance[0], marker='*', color='r', s=300, label='MVP')
    plt.scatter(tan_performance[1], tan_performance[0], marker='*', color='g', s=300, label='Tangency Portfolio')

    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Return')
    plt.title('Efficient Frontier and Industry Returns (Short Selling Allowed)')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

plot_efficient_frontier(means_vector, cov_matrix, industry_norf, mvp_performance, tan_performance)
