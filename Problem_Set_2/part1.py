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
the efficient frontier is the portfolio with the lowest possible risk (variance). The following graph
assumes that short selling is allowed.'''
def efficient_frontier_data(means_vector, cov_matrix, n_points=300, extend=0.5):
    mu = np.asarray(means_vector)
    V = np.asarray(cov_matrix)
    ones = np.ones(len(mu))

    invV = np.linalg.inv(V)
    A = mu.T @ invV @ mu
    B = mu.T @ invV @ ones
    C = ones.T @ invV @ ones
    D = A * C - B**2

    mu_min, mu_max = mu.min(), mu.max()
    mu_range = mu_max - mu_min
    mu_p = np.linspace(mu_min - extend * mu_range, mu_max + extend * mu_range, n_points)
    sigma_p = np.sqrt((C * mu_p**2 - 2 * B * mu_p + A) / D)
    return mu_p, sigma_p

# Get efficient frontier data (no plotting inside the function)
mu_p, sigma_p = efficient_frontier_data(means_vector, cov_matrix)

# Plot outside the function
plt.figure(figsize=(10, 7))
plt.plot(sigma_p, mu_p, color='b', label='Efficient Frontier')

# Plot individual industries
ind_stds = industry_norf.std()
for col in industry_norf.columns:
    plt.scatter(ind_stds[col], means_vector[col], s=100, label=col)

# Plot MVP and Tangency portfolios
plt.scatter(mvp_performance[1], mvp_performance[0], marker='*', color='r', s=300, label='MVP')
plt.scatter(tan_performance[1], tan_performance[0], marker='*', color='g', s=300, label='Tangency Portfolio')

plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('Efficient Frontier and Industry Returns (Short Selling Allowed)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

''' The efficient frontier corresponds to the set of optimal portfolios for each given level of risk.
The relationship between risk and return is quadratic.'''

# Question B
'''b)	Comment on the reliability of the mean return estimates for each industry.  
Then, artificially change the mean return estimates of each industry by a one standard error increase. 
How much does the Tangent portfolio change?  Does the efficient frontier change a lot or a little?'''
 
# Standard error of the mean 
std_errors = industry_norf.std() / np.sqrt(len(industry_norf))
# Increase each mean return estimate by one standard error 
adjusted_means_vector = means_vector + std_errors
# Recalculate the tangent portfolio weights with the adjusted means 
adjusted_weights_tan = tangent_portfolio(cov_matrix, risk_free_rate, adjusted_means_vector)
# Recalculate the tangent portfolio performance with the adjusted means
adjusted_tan_performance = portfolio_performance(adjusted_weights_tan, adjusted_means_vector, cov_matrix)
print(f'Adjusted Tangent Mean Return: {adjusted_tan_performance[0] - tan_performance[0]} , Adjusted Tangent Risk: {adjusted_tan_performance[1] - tan_performance[1]}')
print(f'Change in Tangent Weights: {adjusted_weights_tan - weights_tan}')

# Get new efficient frontier data with adjusted means
adjusted_mu_p, adjusted_sigma_p = efficient_frontier_data(adjusted_means_vector,cov_matrix=cov_matrix)
# Plot both efficient frontiers for comparison 
plt.figure(figsize=(10, 7))
plt.plot(sigma_p, mu_p, color='b', label='Original Efficient Frontier')
plt.plot(adjusted_sigma_p, adjusted_mu_p, color='r', label='Adjusted Efficient Frontier')
plt.scatter(tan_performance[1], tan_performance[0], marker='*', color='g', s=300, label='Original Tangency Portfolio')
plt.scatter(adjusted_tan_performance[1], adjusted_tan_performance[0], marker='*', color='r', s=300, label='Adjusted Tangency Portfolio')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('Efficient Frontier Comparison')
plt.legend()
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# Question C
'''c)	Comment on the reliability of the covariance matrix estimate.  
First, assume that all covariances are zero and recompute the efficient frontier 
using the diagonal matrix of variances as the covariance matrix.  
Then, assume very simply that the covariance matrix is just the identity matrix 
(i.e., a matrix of ones along the diagonal and zeros everywhere else).  
Does the mean-variance frontier change a lot or a little, relative to b)?  
How important are the covariance terms relative to the variance terms?'''

# Use diagonal matrix of variances for "zero covariance" case
diagonal_matrix = np.diag(industry_norf.var())
# Identity matrix for "identity covariance" case
cov_matrix_identity = np.eye(len(means_vector))

# Get efficient frontier data with different covariance matrices
mu_p_diag, sigma_p_diag = efficient_frontier_data(means_vector, cov_matrix=diagonal_matrix)
mu_p_identity, sigma_p_identity = efficient_frontier_data(means_vector, cov_matrix=cov_matrix_identity)

# Plot all efficient frontiers for comparison
plt.figure(figsize=(10, 7))
plt.plot(sigma_p, mu_p, color='b', label='Original Efficient Frontier')
plt.plot(sigma_p_diag, mu_p_diag, color='c', label='Diagonal Covariance Efficient Frontier')
plt.plot(sigma_p_identity, mu_p_identity, color='m', label='Identity Covariance Efficient Frontier')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('Efficient Frontier Comparison')
plt.legend()
plt.show()

# Question D 
'''d)	Run some simulations similar to what Jorion did in his study.  
Using the mean and covariance matrix you calculated in sample from the historical returns, 
use these parameters to simulate data under a multivariate normal distribution.  
•	Draw a random sample of 10 (N) returns from this distribution T times (T = the number of months). 
 This gives you one simulation.
•	Calculate the tangency and minimum variance portfolio weights from these simulated data. 
 Then, apply these weights to the actual (NOT SIMULATED) returns on the industries 
 (e.g., the weights come from the simulated returns, but they are applied to true/actual returns on the industries).
•	Then repeat 1,000 times and save the mean and standard deviation of each MVP and 
Tangency portfolio you calculated under each simulation of data used to get the weights and applied to actual returns.
•	On two separate plots of mean-standard deviation space, plot the simulated MVP and Tangency portfolios relative 
to the ones calculated using weights estimated from the real data. 
 (One plot for MVP and one for Tangency portfolios, each plot will contain 1001 data points).
•	These plots indicate the estimation error (under a normal distribution) of the Tangency and MVP weights.  Which portfolio (MVP or Tangency) is estimated with less error?  Why?
'''

num_simulations = 1000
num_assets = len(means_vector)
num_periods = len(industry_norf)
simulated_mvp = []
simulated_tan = []
for _ in range(num_simulations):
    simulated_returns = np.random.multivariate_normal(means_vector, cov_matrix, size=num_periods)
    simulated_df = pd.DataFrame(simulated_returns, columns=industry_norf.columns)
    sim_cov_matrix = simulated_df.cov()
    sim_means_vector = simulated_df.mean()
    
    sim_weights_mvp = min_var_portfolio(sim_cov_matrix)
    sim_weights_tan = tangent_portfolio(sim_cov_matrix, risk_free_rate, sim_means_vector)
    
    sim_mvp_performance = portfolio_performance(sim_weights_mvp, means_vector, cov_matrix)
    sim_tan_performance = portfolio_performance(sim_weights_tan, means_vector, cov_matrix)
    
    simulated_mvp.append(sim_mvp_performance)
    simulated_tan.append(sim_tan_performance)

simulated_mvp = np.array(simulated_mvp)
print(simulated_mvp)
simulated_tan = np.array(simulated_tan)
print(simulated_tan)

# Plot MVP simulations
plt.figure(figsize=(10, 7))
plt.scatter(simulated_mvp[:,1], simulated_mvp[:,0], alpha=0.5, label='Simulated MVPs') # simulated_mvp[:,1] = all of the standard deviations, simulated_mvp[:,0] = all of the returns
plt.scatter(mvp_performance[1], mvp_performance[0], color='r', s=200, label='Actual MVP')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('MVP Simulations vs Actual')
plt.legend()
plt.grid(True)
# Set axis limits based on data range with a small margin
x_margin = (simulated_mvp[:,1].max() - simulated_mvp[:,1].min()) * 0.1
y_margin = (simulated_mvp[:,0].max() - simulated_mvp[:,0].min()) * 0.1
plt.xlim(simulated_mvp[:,1].min() - x_margin, simulated_mvp[:,1].max() + x_margin)
plt.ylim(simulated_mvp[:,0].min() - y_margin, simulated_mvp[:,0].max() + y_margin)
plt.show()

# Plot Tangency simulations
plt.figure(figsize=(10, 7))
plt.scatter(simulated_tan[:,1], simulated_tan[:,0], alpha=0.5, label='Simulated Tangency Portfolios')
plt.scatter(tan_performance[1], tan_performance[0], color='r', s=200, label='Actual Tangency Portfolio')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('Tangency Portfolio Simulations vs Actual')
plt.legend()
plt.grid(True)
# Set axis limits based on data range with a small margin
x_margin = (simulated_tan[:,1].max() - simulated_tan[:,1].min()) * 0.1
y_margin = (simulated_tan[:,0].max() - simulated_tan[:,0].min()) * 0.1
plt.xlim(simulated_tan[:,1].min() - x_margin, simulated_tan[:,1].max() + x_margin)
plt.ylim(simulated_tan[:,0].min() - y_margin, simulated_tan[:,0].max() + y_margin)
plt.show()