import numpy as np
import matplotlib.pyplot as plt

# Dati osservati
data = np.array([5.2, 4.8, 6.1, 5.9, 5.0])
n = len(data)
sigma2 = 0.25  # varianza nota

# Parametri prior
mu_prior_mean = 0
mu_prior_var = 100

# Definiamo la funzione di log-posterior (senza costanti)
def log_posterior(mu, data, sigma2, mu_prior_mean, mu_prior_var):
    likelihood = -0.5 * np.sum((data - mu)**2) / sigma2
    prior = -0.5 * (mu - mu_prior_mean)**2 / mu_prior_var
    return likelihood + prior

# Metropolis-Hastings MCMC
def metropolis_hastings(log_posterior, initial_mu, iterations, proposal_std, data, sigma2, mu_prior_mean, mu_prior_var):
    mu_current = initial_mu
    samples = []
    
    for i in range(iterations):
        mu_proposal = np.random.normal(mu_current, proposal_std)
        
        log_posterior_current = log_posterior(mu_current, data, sigma2, mu_prior_mean, mu_prior_var)
        log_posterior_proposal = log_posterior(mu_proposal, data, sigma2, mu_prior_mean, mu_prior_var)
        
        acceptance_ratio = np.exp(log_posterior_proposal - log_posterior_current)
        
        if np.random.rand() < acceptance_ratio:
            mu_current = mu_proposal  # accetta la proposta
        
        samples.append(mu_current)
    
    return np.array(samples)

# Parametri MCMC
initial_mu = 0
iterations = 10000
proposal_std = 0.5  # standard deviation della proposta

# Eseguiamo l'algoritmo
samples = metropolis_hastings(
    log_posterior,
    initial_mu,
    iterations,
    proposal_std,
    data,
    sigma2,
    mu_prior_mean,
    mu_prior_var
)

# Plot dei risultati
plt.figure(figsize=(12,6))
plt.plot(samples)
plt.title('Traccia delle iterazioni MCMC')
plt.xlabel('Iterazione')
plt.ylabel('Valore di μ')
plt.grid()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(samples, bins=50, density=True)
plt.title('Distribuzione approssimata a posteriori di μ')
plt.xlabel('μ')
plt.ylabel('Densità')
plt.grid()
plt.show()

# Stima finale
print(f"Stima di μ (media campioni MCMC): {np.mean(samples):.3f}")
print(f"Intervallo credibile 95%: {np.percentile(samples, 2.5):.3f} - {np.percentile(samples, 97.5):.3f}")
