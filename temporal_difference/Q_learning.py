import sys
sys.path.append("..")
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from gridworld.gridworld import standard_grid

class TemporalDifferenceControl(threading.Thread):
    def __init__(self, environment, daemon=False):
        # Initializing the thread instance
        super().__init__(target=self.run_algorithm, daemon=daemon)

        # Store the environment to interact and perform actions
        self.g = environment

        # Initializing to zero all value function values
        # Getting all playable states
        all_states = self.g.get_all_states()
        
        # To store the action value function
        self.value_q = {(state, action):0 for state in all_states for action in self.g.get_actions(state)}
        
        # Creating a policy selecting available actions at random
        self.policy = dict()
        for state in all_states:
            # Only assigning actions to non-terminal state
            if self.g.is_terminal(state): continue

            actions = self.g.get_actions(state)
            self.policy[state] = np.random.choice(actions)

    def estimate_value_function(self, gamma, delta_threshold, alpha, max_steps, min_episodes, max_episodes, epsilon):
        # Initializing number of episodes played
        episodes = 0

        # Main loop until convergence
        while episodes < max_episodes:
            # Incresing the number of episodes played
            episodes += 1
            print(f"Starting episode {episodes}")
        
            # Initializing the number of steps performed
            steps = 0

            # Initializing biggest change
            delta = 0

            # Caching the starting state
            curr_state = self.g.current_state()

            # Play the game until it's over or until we reach maximum number of steps
            while not self.g.game_over() and steps < max_steps:
                # Getting the epsilon-greedy action to play
                curr_action = self.get_epsilong_greedy_action(curr_state, epsilon)
                
                # Playing such action and increasing counter
                reward = self.g.move(curr_action)
                steps += 1

                # Otherwise get the next state info
                next_state = self.g.current_state()

                # Updating Q(s,a) for the current state
                # Q(s,a) = Q(s,a) + alpha(r + gamma*Q(s', best_action) - Q(s,a))
                best_next_action = self.policy.get(next_state, None)
                old_v = self.value_q[curr_state, curr_action]
                new_v = old_v + alpha*(reward + gamma*self.value_q.get((next_state, best_next_action), 0) - old_v)
                self.value_q[curr_state, curr_action] = new_v

                # Updating the biggest change
                delta = max(delta, np.abs(new_v - old_v))

                # Getting the best action known so far by the samples
                all_actions = self.g.get_actions(curr_state)
                action_values = [self.value_q[curr_state, a] for a in all_actions]
                self.policy[curr_state] = all_actions[np.argmax(action_values)]

                # Shifting states and actions
                curr_state = next_state

            # After game is over, if we've played enough and convergence achieved, then break
            #if episodes >= min_episodes and delta < delta_threshold:
            #    print(f"Convergence reached after {episodes} episodes")
            #    break

            # If we didn't, just keep playing
            self.g.reset()

    def get_epsilong_greedy_action(self, state, epsilon):
        # Check if not terminal state
        if self.g.is_terminal(state): return None

        # Possible actions for the state
        actions = self.g.get_actions(state)

        # Exploration
        if np.random.random() < epsilon:
            action = np.random.choice(actions)
        # Exploitation
        else:
            action = self.policy[state] 

        return action

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
        max_episodes = 1000

        # Percentage of exploration vs. exploitation
        epsilon = 0.9

        # Compute optimal value function until convergence
        # Calculating the value function for the current policy
        self.estimate_value_function(gamma=discount_factor, delta_threshold=convergence_min, alpha=learning_rate, min_episodes=min_episodes, max_episodes=max_episodes, max_steps=max_steps, epsilon=epsilon)
            
        # Show the initial computed final values on the gridboard
        # Getting the V(s) values from the computed Q(s,a)
        all_states = self.g.get_all_states()

        value_s = {state:0 for state in all_states}

        for state in all_states:
            if not self.g.is_terminal(state):
                all_actions = self.g.get_actions(state)
                for action in all_actions:
                    if action == self.policy[state]: value_s[state] = self.value_q[state, action]

        self.g.show_values_on_board(value_s, self.policy)

# Creating a standard grid object
g = standard_grid(rows=5, cols=6, step_cost=-0.1)

# Starting
#  algorithm program in a separate thread
temp_diff_control = TemporalDifferenceControl(g, daemon=True)
temp_diff_control.start()

# Starting the game mainloop
g.mainloop()