import matplotlib.pyplot as plt
import numpy as np

class Bandit:
    def __init__(self, p):
        # Actual win rate
        self.p = p

        # Estimated win rate
        self.p_estimate = 0
        
        # Samples collected
        self.N = 0

    def pull(self):
        # Using a normal distribution std_dev * np.random.randn + mean
        return np.random.random() < self.p

    def update(self, result):
        self.N += 1

        # Update win rate estimate
        self.p_estimate = ((self.N-1)*self.p_estimate + result)/self.N


# EXPERIMENT HYPERPARAMENTERS
NUM_PULLS = 10000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.7]
OPTIMAL_BANDIT = np.argmax(BANDIT_PROBABILITIES)

# Creating all bandits
bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
rewards = np.zeros(NUM_PULLS)
optimal_plays = 0

# Playing each bandit once
for i, bandit in enumerate(bandits):
    result = bandit.pull()
    bandit.update(result)
    rewards[i] = result

# Running the actual experiment
for i in range(len(bandits), NUM_PULLS):
    # Choosing bandit with UCB1
    j = np.argmax([(b.p_estimate + np.sqrt(2*np.log(i)/b.N)) for b in bandits])
    bandit = bandits[j]

    # If playing optimal bandit
    if j == OPTIMAL_BANDIT: optimal_plays += 1

    # Playing the selected bandit
    result = bandit.pull()
    bandit.update(result)

    # Updating rewards log
    rewards[i] = result

# Plotting useful info
print(f"Optimal plays: {optimal_plays}")

cumulative_reward = np.cumsum(rewards)
win_rate = cumulative_reward / (np.arange(NUM_PULLS) + 1)

# Log plot
plt.plot(win_rate)
plt.plot(np.ones(NUM_PULLS) * np.max(BANDIT_PROBABILITIES))
plt.grid()
plt.xscale("log")
plt.show()

# Linear plot
plt.plot(win_rate)
plt.plot(np.ones(NUM_PULLS) * np.max(BANDIT_PROBABILITIES))
plt.grid()
plt.show()