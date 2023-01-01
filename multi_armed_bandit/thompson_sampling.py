import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class Bandit:
    def __init__(self, p, a=1, b=1):
        # Actual win rate
        self.p = p

        # Alpha and Beta parameters for beta distrbution
        self.a = a
        self.b = b
        
        # Samples collected
        self.N = 0

    def pull(self):
        # Value less than p means a win
        return np.random.random() < self.p

    def sample(self):
        # Return a sample from the beta distribution of the bandit
        return np.random.beta(self.a, self.b)

    def update(self, result):
        self.N += 1

        # Update our posterior (alpha and beta)
        # alpha = alpha + sum(results)
        self.a += result

        # beta = beta + num_results - sum(results)
        self.b += (1 - result)

# EXPERIMENT HYPERPARAMETERS
NUM_PULLS = 10000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.7]
OPTIMAL_BANDIT = np.argmax(BANDIT_PROBABILITIES)

# Creating all bandits with a=b=1 for no prior information is known
bandits = [Bandit(p) for p in BANDIT_PROBABILITIES] 
rewards = np.zeros(NUM_PULLS)
optimal_plays = 0

# Check on distributions on samples
check_on_samples = [5, 10, 100, 200, 500, 1000, 5000, 10000]

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
            x_vals = np.linspace(0,1,200)
            y_vals = beta.pdf(x_vals, bandit.a, bandit.b)
            plt.plot(x_vals, y_vals, label=f"true mean = {bandit.p}; win rate={bandit.a-1}/{bandit.N}")
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

