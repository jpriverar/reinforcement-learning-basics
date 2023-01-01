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
        # Using a normal distribution std_dev * np.random.randn + mean
        return np.random.randn() + self.p

    def update(self, result):
        self.N += 1

        # Update win rate estimate
        self.p_estimate = ((self.N-1)*self.p_estimate + result)/self.N

def run_experiment(probabilities, eps, num_trials):
    # Creating the bandits
    bandits = [Bandit(p) for p in probabilities]
    optimal_bandit = bandits[np.argmax([probabilities])]
    rewards = np.zeros(num_trials)
    num_explored = 0
    num_exploited = 0
    num_optimal = 0

    # Doing the actual experiment
    for i in range(num_trials):
        # Explore
        if np.random.random() < eps:
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
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(num_trials)*optimal_bandit.p)
    plt.grid()
    plt.xscale("log")
    plt.show()

    return win_rates

if __name__ == "__main__":
    wr1 = run_experiment([0.2, 0.5, 0.7], 0.1, 100000)
    wr2 = run_experiment([0.2, 0.5, 0.7], 0.05, 100000)
    wr3 = run_experiment([0.2, 0.5, 0.7], 0.01, 100000)

    # Log plot
    plt.plot(wr1, label="eps = 0.1")
    plt.plot(wr2, label="eps = 0.05")
    plt.plot(wr3, label="eps = 0.01")
    plt.grid()
    plt.xscale("log")
    plt.legend()
    plt.show()

    # Linear plot
    plt.plot(wr1, label="eps = 0.1")
    plt.plot(wr2, label="eps = 0.05")
    plt.plot(wr3, label="eps = 0.01")
    plt.grid()
    plt.legend()
    plt.show()