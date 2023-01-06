import sys
sys.path.append("..")
import time
import threading
import numpy as np
from gridworld.gridworld import standard_grid

class MonteCarloExploringStarts(threading.Thread):
    def __init__(self, environment, daemon=False):
        # Initializing the thread instance
        super().__init__(target=self.run_algorithm, daemon=daemon)

        # Store the environment to interact and perform actions
        self.g = environment

        # Initializing to zero all value function values
        # Getting all playable states
        all_states = self.g.get_all_states()
        
        # To store the value function
        # key = state, action
        # value = sample mean
        self.value_q = {(state, action):0 for state in all_states for action in self.g.get_actions(state)}

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

        # Play the game until it's over or until we reach maximum number of steps
        # Set the agent in a random starting position
        random_state = all_states[np.random.choice(len(all_states))]
        self.g.set_state(random_state)
        curr_state = self.g.current_state()

        # Starting with a random action for the starting state
        action = np.random.choice(self.g.get_actions(curr_state))

        # To save the states we have visited so far (first visit Monte Carlo), and their respective action and reward
        visited_states = [curr_state]
        performed_actions = [action]
        rewards = [0]

        while not self.g.game_over() and steps <= max_steps:
            # Playing already determined action
            reward = self.g.move(action)

            # Saving the state where we landed
            curr_state = self.g.current_state()
            visited_states.append(curr_state)

            # Saving the obtained sample reward 
            rewards.append(reward)

            # Getting the action to play according to the agent's policy and saving it
            action = self.policy.get(curr_state, None)
            performed_actions.append(action)

            # Add one step taken to the counter
            steps += 1

        return visited_states, performed_actions, rewards

    def estimate_action_values(self, discount_factor=0.9, delta_threshold=0.001, max_steps=20, max_episodes=100):
        # Getting all non terminal states
        all_states = [state for state in self.g.get_all_states() if not self.g.is_terminal(state)]

        # To save the number of return samples from each action in each state
        returns = {(state, action):0 for state in all_states for action in self.g.get_actions(state)}

        # Countin the episodes played before convergence
        episodes = 0

        # Loop for 100 episodes(until convergence)
        while episodes < max_episodes:
            # Counting the starting episode
            episodes += 1

            # Playing the game with the current policy
            print(f"Starting episode {episodes}")
            visited_states, performed_actions, rewards = self.play_game(max_steps=max_steps)
            print(f"Episode {episodes} finished")
            print(visited_states)
            print(performed_actions)
            print(rewards)
            # Once the game is over, compute the action value function with the sample mean
            # Starting action value estimation, since last estate must be terminal state
            G = 0

            # Going backwards through the visited states except the last state
            for time_step, (state, action) in reversed(list(enumerate(zip(visited_states[:-1], performed_actions[:-1])))):
                G = rewards[time_step+1] + discount_factor*G

                # For first visit Monte Carlo, check if we have been in this state and performed this action before
                if (state, action) not in zip(visited_states[:time_step], performed_actions[:time_step]):
                    print(f"Updating value for {time_step}: ({state},{action})")
                    # Counting the newly collected sample
                    returns[state, action] += 1

                    # Computing the sample mean from the previous sample mean
                    sample_mean = self.value_q[state, action]
                    num_samples = returns[state, action]
                    # mean = 1/samples * (samples-1 * prev_mean + new_sample)
                    new_mean = ((num_samples-1)*sample_mean + G)/num_samples

                    # Saving the new mean action value
                    self.value_q[state, action] = new_mean

                    # Updating the policy for the action with the best sample mean so far
                    all_actions = self.g.get_actions(state)
                    action_values = [self.value_q[state, a] for a in all_actions]
                    print(all_actions)
                    print(action_values)
                    self.policy[state] = all_actions[np.argmax(action_values)]

            # Restart the game to keep playing and collecting samples
            self.g.reset()  


    # Main function runinng the algorithm -> To improve the policy
    def run_algorithm(self):
        # Wait a moment for the gridworld instance to start
        time.sleep(3)

        # Defining a delta threshold and a discount factor
        discount_factor = 0.9
        delta_threshold = 0.001
        episodes = 1000

        # Compute optimal value function until convergence
        # Calculating the value function for the current policy
        self.estimate_action_values(discount_factor=discount_factor, delta_threshold=delta_threshold, max_episodes=episodes)
            
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
g = standard_grid(windy=True, probabilities="Auto", wind_strength=1, step_cost=-0.1)

# Starting algorithm program in a separate thread
policy_improvement = MonteCarloExploringStarts(g, daemon=True)
policy_improvement.start()

# Starting the game mainloop
g.mainloop()