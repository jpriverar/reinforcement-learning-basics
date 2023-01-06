import sys
sys.path.append("..")
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from gridworld.gridworld import standard_grid

class MonteCarloEvaluation(threading.Thread):
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

    def play_game(self, max_steps):
        # Getting all playable states
        all_states = [state for state in self.g.get_all_states() if not self.g.is_terminal(state)]

        # Initializing the number of steps performed
        steps = 0

        # To save the states we have visited so far (first visit Monte Carlo), and their respective reward
        visited_states = []
        rewards = []

        # Play the game until it's over or until we reach maximum number of steps
        # Set the agent in a random starting position
        random_state = all_states[np.random.choice(len(all_states))]
        self.g.set_state(random_state)

        # Saving the starting state 
        curr_state = self.g.current_state()
        visited_states.append(curr_state)

        # Saving a zero reward for this state
        rewards.append(0)

        while not self.g.game_over() and steps <= max_steps:
            # Getting the action to play according to the agent's policy
            action = self.policy[curr_state]

            # Playing such action
            reward = self.g.move(action)

            # Saving the state where we landed
            curr_state = self.g.current_state()
            visited_states.append(curr_state)

            # Saving the obtained reward sample
            rewards.append(reward)

            # Add one step taken to the counter
            steps += 1

        return visited_states, rewards

    def estimate_value_function(self, discount_factor=0.9, delta_threshold=0.001, max_steps=20):
        # Getting all non terminal states
        all_states = [state for state in self.g.get_all_states() if not self.g.is_terminal(state)]

        # To save the return samples from each state
        returns = {state:[] for state in all_states}

        # Countin the episodes played before convergence
        episodes = 0

        # Loop for 100 episodes(until convergence)
        while episodes < 20:
            # Counting the starting episode
            episodes += 1

            visited_states, rewards = self.play_game(max_steps=max_steps)

            # Once the game is over, compute the value function with the sample mean
            # To store the biggest change in value for any state
            delta = 0

            # Starting value estimation, since last estate must be terminal state
            G = 0
            # Going backwards through the visited states except the last state
            for time_step, state in reversed(list(enumerate(visited_states[:-1]))):
                G = rewards[time_step+1] + discount_factor*G

                # For first visit Monte Carlo, check if we have been in this state before
                if state not in visited_states[:time_step]:
                    returns[state].append(G)

                    # Check the change in the value for convergence
                    old_v = self.value_s[state]
                    new_v = np.mean(returns[state])

                    # Saving the new value
                    self.value_s[state] = new_v

                    # Update the delta value
                    delta = max(delta, np.abs(self.value_s[state] - old_v))

            # Check if convergence was achived
            #if delta < delta_threshold:
            #    print(f"Value function has converged after {episodes} episodes")
            #    break

            # If it was't achived, then restart the game to keep playing
            self.g.reset()  


    # Main function runinng the algorithm -> To improve the policy
    def run_algorithm(self):
        # Wait a moment for the gridworld instance to start
        time.sleep(3)

        # Defining a delta threshold and a discount factor
        discount_factor = 0.9
        delta_threshold = 0.001

        # Compute optimal value function until convergence
        # Calculating the value function for the current policy
        self.estimate_value_function(discount_factor=discount_factor, delta_threshold=delta_threshold)
            
        # Show the initial computed final values on the gridboard
        self.g.show_values_on_board(self.value_s, self.policy)


# Creating a standard grid object
g = standard_grid()

# Starting
#  algorithm program in a separate thread
policy_evaluation = MonteCarloEvaluation(g, daemon=True)
policy_evaluation.start()

# Starting the game mainloop
g.mainloop()