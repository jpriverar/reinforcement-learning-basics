import sys
sys.path.append("..")
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from gridworld.gridworld import standard_grid

class IterativeEvaluationPolicy(threading.Thread):
    def __init__(self, environment, daemon=False):
        # Initializing the thread instance
        super().__init__(target=self.run_algorithm, daemon=daemon)
        
        # To store the value function
        self.value_s = dict()

        # To store the steps required before convergence
        self.steps = 0

        # Store the environment to interact and perform actions
        self.g = environment

    def find_value_function(self, discount_factor=0.9, delta_threshold=0.001):
        # Getting all playable states
        all_states = self.g.get_all_states()

        # Initializing to zero all value function values
        value_s = {state:[0] for state in all_states}

        # Number of steps required to converge
        steps = 0

        # Infinite loop until all value functions converge
        while True:
            # Increase the number of steps
            steps += 1

            # Play the game normally, as long as it's not over
            # No need to play the game in this case, we are "cheating"
            # Initialize delta to zero
            delta = 0

            for state in all_states:
                # Only update the value if it's not a terminal state, cause otherwise it's just zero
                if state in self.g.terminal: continue

                # If not terminal state then we perform the upate
                value_old = value_s[state][-1]
                value_s[state].append(self.__bellman(state, discount_factor, value_s))

                # Update the delta value
                delta = max(delta, abs(value_s[state][-1] - value_old))

            # Check if all deltas have crossed the threshold
            if delta < delta_threshold: 
                print("Value function has converged")
                break

        return steps, value_s

    def __bellman(self, state, gamma, value_s):
        # Setting the agent on the state to calculate
        self.g.set_state(state)

        # Initializing the value to zero
        value = 0
        for action in self.g.get_actions(state):
            for next_state in self.g.get_all_states():
                # Reward is deterministic so no need to sum over it
                # value(s) = policy(a|s) * probs(s'|a,s) * [r + gamma*value(s')]
                # Get the probability of falling in such next state
                prob_next_state = g.probabilities[state, action].get(next_state,0)
                value += (action == self.policy(state)) * (prob_next_state) * (self.g.rewards.get(next_state,0) + gamma*value_s[next_state][-1])

        # Return the total sum of future rewards
        return value

    # Define the policy for the agent
    def policy(self, state):
        return "R"

    # Main function runinng the algorithm
    def run_algorithm(self):
        # Wait a moment for the gridworld instance to start
        time.sleep(3)

        # Defining a delta threshold and a discount factor
        discount_factor = 0.9
        delta_threshold = 0.001

        # Calculating the value function
        steps, value_s = self.find_value_function(discount_factor=discount_factor, delta_threshold=delta_threshold)

        # Show the computed final values on the gridboard
        self.g.show_values_on_board(value_s)

        # Assigning the results to the instance variables
        self.steps = steps
        self.value_s = value_s

# Creating a standard grid object
g = standard_grid(windy=True, probabilities="Auto")

# Starting algorithm program in a separate thread
evaluate_policy = IterativeEvaluationPolicy(g, daemon=True)
evaluate_policy.start()

# Starting the game mainloop
g.mainloop()

# After game is over, print and plot useful information
print(f"Required steps before convergence: {evaluate_policy.steps}")

for state, values_in_time in evaluate_policy.value_s.items():
    try:
        plt.plot(np.arange(evaluate_policy.steps+1), values_in_time, label=str(state))
    except Exception as e:
        print("Not plotting terminal state, for its value does not change")

plt.title("V(s) variation in time")
plt.xlabel("Step")
plt.ylabel("V(s)")
plt.grid()
plt.legend()
plt.show()