import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, p):
        # Actual win rate
        self.p = p

        # Estimated win rate
        self.p_estimate = 0
        
        # Samples collected
        self.N = 0

    def pull(self):
        # Value less than p means a win
        return np.random.random() < self.p

    def update(self, result):
        self.N += 1

        # Update win rate estimate
        self.p_estimate = ((self.N-1)*self.p_estimate + result)/self.N

# EXPERIMENT HYPERPARAMETERS
NUM_PULLS = 10000
EPS = 0.1
BANDIT_WIN_RATES = [0.2, 0.5, 0.7]

# Creating the bandits
bandits = [Bandit(p) for p in BANDIT_WIN_RATES]
optimal_bandit = bandits[np.argmax([BANDIT_WIN_RATES])]
rewards = np.zeros(NUM_PULLS)
num_explored = 0
num_exploited = 0
num_optimal = 0

# Doing the actual experiment
for i in range(NUM_PULLS):
    # Explore
    if np.random.random() < EPS:
        num_explored += 1
        # Choosing a random bandit
        bandit = np.random.choice(bandits)

    # Exploit
    else:
        num_exploited += 1
        # Grab the bandit with Maximum likelihood estimate
        max_p_arg = np.argmax([b.p_estimate for b in bandits])
        bandit  = bandits[max_p_arg]

    # Checking if optimal bandit was played
    if bandit == optimal_bandit: 
        num_optimal += 1

    # Using selected bandit and updating estimate win rate
    result = bandit.pull()
    bandit.update(result)

    # Saving data to plot
    rewards[i] = result

# Plotting and printing some useful data
for i, b in enumerate(bandits):
    print(f"mean estimate for bandit {i+1} = {b.p_estimate}")

print(f"Explored times: {num_explored}")
print(f"Exploited times: {num_exploited}")
print(f"Optimal plays: {num_optimal}")

cumulative_rewards = np.cumsum(rewards)
win_rates = cumulative_rewards / (np.arange(NUM_PULLS) + 1)
plt.plot(win_rates)
plt.plot(np.ones(NUM_PULLS)*optimal_bandit.p)
plt.grid()
plt.xscale("log")
plt.show()