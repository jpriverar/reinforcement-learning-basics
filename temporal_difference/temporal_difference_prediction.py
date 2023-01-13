import sys
sys.path.append("..")
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from gridworld.gridworld import standard_grid

class TemporalDifferencePrediction(threading.Thread):
    def __init__(self, environment, daemon=False):
        # Initializing the thread instance
        super().__init__(target=self.run_algorithm, daemon=daemon)

        # Store the environment to interact and perform actions
        self.g = environment

        # Initializing to zero all value function values
        # Getting all playable states
        all_states = self.g.get_all_states()
        
        # To store the value function
        self.value_s = {state:0 for state in all_states}

        # To store the steps required before convergence
        self.steps = 0
        
        # Creating a policy selecting available actions at random
        self.policy = dict()
        for state in all_states:
            # Only assigning actions to non-terminal state
            if self.g.is_terminal(state): continue

            actions = self.g.get_actions(state)
            self.policy[state] = np.random.choice(actions)

    def estimate_value_function(self, gamma, delta_threshold, alpha, max_steps, min_episodes):
        # Initializing number of episodes played
        episodes = 0

        # Main loop until convergence
        while True:
            # Incresing the number of episodes played
            episodes += 1
        
            # Initializing the number of steps performed
            steps = 0

            # Initializing biggest change
            delta = 0

            # Caching the starting state 
            curr_state = self.g.current_state()

            # Play the game until it's over or until we reach maximum number of steps
            while not self.g.game_over() and steps <= max_steps:
                # Getting the action to play according to the agent's policy
                action = self.policy[curr_state]

                # Playing such action and getting info
                reward = self.g.move(action)
                next_state = self.g.current_state()

                # Updating V(s) for the current state
                # V(s) = V(s) + alpha(r + gamma*V(s') - V(s))
                old_v = self.value_s[curr_state]
                new_v = old_v + alpha*(reward + gamma*self.value_s[next_state] - old_v)
                self.value_s[curr_state] = new_v

                # Add one step taken to the counter and shifting states
                curr_state = next_state
                steps += 1

            # After game is over, if we've played enough
            if episodes >= min_episodes:
                # If we have, check for convergence
                if delta < delta_threshold: 
                    print(f"Convergence reached after {episodes} episodes")
                    break

            # If we didn't, just keep playing
            self.g.reset()

    # Main function runinng the algorithm -> To improve the policy
    def run_algorithm(self):
        # Wait a moment for the gridworld instance to start
        time.sleep(3)

        # Defining a delta threshold, discount factor for returns and learning rate
        discount_factor = 0.9
        convergence_min = 0.001
        learning_rate = 0.9

        # Minimum of episodes to play before checking for convergence and max steps per 
        # episode to avoid infinite loops
        max_steps = 20
        min_episodes = 10

        # Compute optimal value function until convergence
        # Calculating the value function for the current policy
        self.estimate_value_function(gamma=discount_factor, delta_threshold=convergence_min, alpha=learning_rate, min_episodes=min_episodes, max_steps=max_steps)
            
        # Show the initial computed final values on the gridboard
        self.g.show_values_on_board(self.value_s, self.policy)


# Creating a standard grid object
g = standard_grid()

# Starting
#  algorithm program in a separate thread
temp_diff_pred = TemporalDifferencePrediction(g, daemon=True)
temp_diff_pred.start()

# Starting the game mainloop
g.mainloop()