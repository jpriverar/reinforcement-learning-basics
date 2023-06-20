import sys
sys.path.append("..")
import time
import threading
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from gridworld.gridworld import standard_grid

class Model:
    def __init__(self, samples):
        # Defining and training our feature expansion model
        self.featurizer = RBFSampler() 
        self.featurizer.fit(samples)

        # Getting the dimensions form such model
        dims = self.featurizer.random_offset_.shape[0]

        # Then initialize our model weights
        self.w = np.zeros(dims)

    def predict(self, state):
        x = self.featurizer.transform([state])[0]
        return np.matmul(x, self.w)

    def grad(self, state):
        x = self.featurizer.transform([state])[0]
        return x

class TemporalDifferenceControl(threading.Thread):
    def __init__(self, environment, daemon=False):
        # Initializing the thread instance
        super().__init__(target=self.run_algorithm, daemon=daemon)

        # Store the environment to interact and perform actions
        self.g = environment

        # Action encoding variable
        self.action_ref = {"U":(1,0,0,0),
                           "D":(0,1,0,0),
                           "L":(0,0,1,0),
                           "R":(0,0,0,1),
                           "N":(0,0,0,0)}
        
        # Creating a policy selecting available actions at random
        all_states = self.g.get_all_states()
        self.policy = dict()
        for state in all_states:
            # Only assigning actions to non-terminal state
            if self.g.is_terminal(state): continue

            actions = self.g.get_actions(state)
            self.policy[state] = np.random.choice(actions)

    def encode(self, state, action):
        encoded_action = self.action_ref[action]
        return state + encoded_action

    def get_best_action_value(self, state):
        # If we are in terminal state, just return the value
        if self.g.is_terminal(state):
            return "N", self.model.predict(self.encode(state, "N"))

        # Getting the action value for every possible action
        all_actions = self.g.get_actions(state)
        action_values = [self.model.predict(self.encode(state, action)) for action in all_actions]

        return all_actions[np.argmax(action_values)], max(action_values)

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

                # Transforming both states
                Qsa = self.model.predict(self.encode(curr_state, curr_action))
                Qsa2 = self.get_best_action_value(next_state)[1]

                # Getting the return of the current state
                if self.g.is_terminal(next_state):
                    y = reward
                else:
                    y = reward + gamma*Qsa2

                # Updating the weights of the approximation model (gradient ascent)
                # w = w + alpha * (y - gamma*w*similarity(s)) * similarity(s)
                self.model.w += alpha*(y - Qsa)*self.model.grad(self.encode(curr_state, curr_action))

                # Updating the policy to the best known action so far
                self.policy[curr_state] = self.get_best_action_value(curr_state)[0]

                # Shifting states and actions
                curr_state = next_state

            # If we didn't, just keep playing
            self.g.reset()

    def get_epsilong_greedy_action(self, state, epsilon):
        # Check if not terminal state
        if self.g.is_terminal(state): return "N"

        # Possible actions for the state
        actions = self.g.get_actions(state)

        # Exploration
        if np.random.random() < epsilon:
            action = np.random.choice(actions)
        # Exploitation
        else:
            action = self.policy[state] 

        return action

    def gather_samples(self, num_episodes):
        print("Collecting samples from the environment...")
        # To save all the state samples
        samples = []

        # Collecting samples during the defined number of episodes
        for i in range(num_episodes):
            while not self.g.game_over():
                # We get the state and save it
                curr_state = self.g.current_state()

                # Choose a random action to do
                action = np.random.choice(self.g.get_actions(curr_state))

                # Encode the action and concatenate it to the state
                encoded_state_action = self.encode(curr_state, action)
                samples.append(encoded_state_action)

                # Performed the selected action and get the reward
                reward = self.g.move(action)
            
            # If game is over then reset to keep playing
            encoded_state_action = self.encode(self.g.current_state(), "N")
            samples.append(encoded_state_action)
            self.g.reset()
        
        return samples

    # Main function runinng the algorithm -> To improve the policy
    def run_algorithm(self):
        # Wait a moment for the gridworld instance to start
        time.sleep(1)

        # Collecting samples from the environment to initialize our model's RBF kernel
        samples = self.gather_samples(250)
        
        # Initialize our linear model
        print("Samples collected, training RBF Sampler...")
        self.model = Model(samples)

        # Defining a delta threshold, discount factor for returns and learning rate
        discount_factor = 0.9
        convergence_min = 0.001
        learning_rate = 0.01

        # Minimum of episodes to play before checking for convergence and max steps per 
        # episode to avoid infinite loops
        max_steps = 20
        min_episodes = 10
        max_episodes = 1000

        # Percentage of exploration vs. exploitation
        epsilon = 0.9

        # Compute optimal value function until convergence
        # Calculating the value function for the current policy
        print("Starting actual reinforcement learning...")
        self.estimate_value_function(gamma=discount_factor, delta_threshold=convergence_min, alpha=learning_rate, min_episodes=min_episodes, max_episodes=max_episodes, max_steps=max_steps, epsilon=epsilon)
            
        # Show the initial computed final values on the gridboard
        # Getting the V(s) values from the model weights
        value_s = dict()
        for state in self.g.get_all_states():
            if not self.g.is_terminal(state):
                all_actions = self.g.get_actions(state)
                for action in all_actions:
                    if action == self.policy[state]: value_s[state] = self.model.predict(self.encode(state, action))

        self.g.show_values_on_board(value_s, self.policy)

# Creating a standard grid object
g = standard_grid(step_cost=-0.1)

# Starting
#  algorithm program in a separate thread
temp_diff_pred = TemporalDifferenceControl(g, daemon=True)
temp_diff_pred.start()

# Starting the game mainloop
g.mainloop()