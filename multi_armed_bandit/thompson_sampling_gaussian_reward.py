import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Bandit:
    def __init__(self, p, t=1, m=0.5, l=1):
        # Win rate is gaussian, p=mean, t=actual precision
        self.p = p
        self.t = t

        # Estimated mean and precision parameters for gaussian distribution
        self.m = m
        self.l = l

        # Keep count of the reward for this bandit
        self.total_sum = 0
        
        # Samples collected
        self.N = 0

    def pull(self):
        # Reward value is continuous normally distributed value
        # Computing standard deviation from precision
        sigma = np.sqrt(1/self.t)
        return np.random.normal(self.p, sigma)

    def sample(self):
        # Return a sample from the gaussian estimated distribution of the bandit
        # Computing standard deviation from precision
        sigma = np.sqrt(1/self.l)
        return np.random.normal(self.m, sigma)

    def update(self, result):
        self.N += 1

        # Sum the obtained result
        self.total_sum += result

        # Update our posterior (m and lambda)
        # m = 1/(num_samples*actual_precision + lambda_start)*(start_lambda*start_m + actual_precision*sum(results))
        self.m = (1/(self.l+self.t)) * (self.l*self.m + self.t*result)

        # lambda = num_samples*actual_precision + lambda_start
        self.l += self.t

        

# EXPERIMENT HYPERPARAMETERS
NUM_PULLS = 10000
BANDIT_PROBABILITIES = [5, 10, 20]
#BANDIT_PRECISION = [1, 1.2, 1.5] # If commented all have 1 as precision
OPTIMAL_BANDIT = np.argmax(BANDIT_PROBABILITIES)

# Creating all bandits with a=b=1 for no prior information is known
bandits = [Bandit(p) for p in BANDIT_PROBABILITIES] 
rewards = np.zeros(NUM_PULLS)
optimal_plays = 0

# Check on distributions on samples
check_on_samples = [5, 10, 100, 200, 500, 1000, 5000, 10000]
#check_on_samples = []

# Initialization: the bayesian method allows user to input prior knowledge,
# if the initial estimated mean is too far away from the true mean, our estimated
# distribution will not cover the true distribution values, and therefore any suboptimal
# choice that's picked first will be exploited.
for i in range(10): # Pure exploration for 10 iterations
    bandit = np.random.choice(bandits)
    result = bandit.pull()
    bandit.update(result)

# Running the actual experiment
for i in range(NUM_PULLS):
    # Choosing the bandit with the best beta distribution sample
    j = np.argmax([b.sample() for b in bandits])
    bandit = bandits[j]

    # Checking if optimal bandit was played
    if j == OPTIMAL_BANDIT: optimal_plays += 1

    # Playing that bandit and updating its alpha and beta
    result = bandit.pull()
    bandit.update(result)

    # Updating result log 
    rewards[i] = result

    # Plotting the distributions of all bandits
    if (i+1) in check_on_samples:
        for bandit in bandits:
            x_vals = np.linspace(-30,30,500)
            y_vals = norm.pdf(x_vals, bandit.m, np.sqrt(1/bandit.l))
            plt.plot(x_vals, y_vals, label=f"miu={bandit.p}; sigma={np.sqrt(1/bandit.t)}")
        plt.title(f"{i+1} Pulls")
        plt.grid()
        plt.legend()
        plt.show()

# Plotting useful info
print(f"Optimal plays: {optimal_plays}")

cumulative_reward = np.cumsum(rewards)
win_rate = cumulative_reward/(np.arange(NUM_PULLS) + 1)

# Log plot of reward over time
plt.plot(win_rate)
plt.plot(np.ones(NUM_PULLS) * np.max(BANDIT_PROBABILITIES))
plt.grid()
plt.xscale("log")
plt.plot()
plt.show()

# Linear plot of reward over time
plt.plot(win_rate)
plt.plot(np.ones(NUM_PULLS) * np.max(BANDIT_PROBABILITIES))
plt.grid()
plt.plot()
plt.show()