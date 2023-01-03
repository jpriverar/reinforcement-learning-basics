import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from gridworld.gridworld import standard_grid

class ValueIterator(threading.Thread):
    def __init__(self, environment, daemon=False):
        # Initializing the thread instance
        super().__init__(target=self.run_algorithm, daemon=daemon)

        # Store the environment to interact and perform actions
        self.g = environment

        # Initializing to zero all value function values
        # Getting all playable states
        all_states = self.g.get_all_states()
        
        # To store the value function
        self.value_s = {state:[0] for state in all_states}

        # To store the steps required before convergence
        self.steps = 0

        # Storing the agent policy we are gonna improve
        # key = state, value = action
        self.policy = dict()

        # Creating a policy selecting available actions at random
        for state in all_states:
            # Only assigning actions to non-terminal state
            if self.g.is_terminal(state): continue

            actions = self.g.get_actions(state)
            self.policy[state] = np.random.choice(actions)

    def find_value_function(self, discount_factor=0.9, delta_threshold=0.001):
        # Getting all playable states
        all_states = self.g.get_all_states()

        # Number of steps required to converge
        self.steps = 0

        # Infinite loop until all value functions converge
        while True:
            # Increase the number of steps
            self.steps += 1

            # Play the game normally, as long as it's not over
            # No need to play the game in this case, we are "cheating"
            # Initialize delta to zero
            delta = 0

            for state in all_states:
                # Only update the value if it's not a terminal state, cause otherwise it's just zero
                if self.g.is_terminal(state): continue

                # If not terminal state then we perform the upate
                value_old = self.value_s[state][-1]
                self.value_s[state].append(max(self.__bellman(state, discount_factor)))

                # Update the delta value
                delta = max(delta, abs(self.value_s[state][-1] - value_old))

            # Check if all deltas have crossed the threshold
            if delta < delta_threshold: 
                print(f"Value function has converged after {self.steps} steps")
                break

    def __bellman(self, state, gamma):
        # Setting the agent on the state to calculate
        self.g.set_state(state)

        # List possible actions in the current state
        actions = self.g.get_actions(state)

        # Action-value for all the possible actions
        action_values = []
        
        # Initializing the value to zero
        for action in actions:
            action_val = 0
            for next_state in self.g.get_all_states():
                # Reward is deterministic so no need to sum over it
                # value(s) = policy(a|s) * probs(s'|a,s) * [r + gamma*value(s')]
                # If probabilities for state transition were declared
                if g.probabilities: 
                    prob_next_state = g.probabilities[state, action].get(next_state,0)

                # Else just get the next state (deterministic)
                else:
                    prob_next_state = 1 if next_state == g.get_next_state(state, action) else 0 

                action_val += (prob_next_state) * (self.g.rewards.get(next_state,0) + gamma*self.value_s[next_state][-1])

            # Saving the action_val for the current action only
            action_values.append(action_val)

        # Return the action values of all possible actions in the current state
        return action_values
        
    def improve_policy(self):
        print("Improving policy...")

        # Getting all playable states
        all_states = self.g.get_all_states()

        for state in all_states:
            # Only trying different actions on non-terminal states
            if self.g.is_terminal(state): continue

            # To know to which action the value corresponds
            actions = self.g.get_actions(state)

            # Getting the best possible action based on the current value function
            new_action = actions[np.argmax(self.__bellman(state, 0.9))]

            # Updating the policy
            self.policy[state] = new_action

        print("Policy improved!")

    # Main function runinng the algorithm -> To improve the policy
    def run_algorithm(self):
        # Wait a moment for the gridworld instance to start
        time.sleep(3)

        # Defining a delta threshold and a discount factor
        discount_factor = 0.9
        delta_threshold = 0.001

        # Compute optimal value function until convergence
        # Calculating the value function for the current policy
        self.find_value_function(discount_factor=discount_factor, delta_threshold=delta_threshold)

        # Once the optimal value function is obtained, compute the optimal policy
        self.improve_policy()
            
        # Show the initial computed final values on the gridboard
        self.g.show_values_on_board(self.value_s, self.policy)


# Creating a standard grid object
g = standard_grid(windy=True, probabilities="Auto", wind_strength=1)

# Starting algorithm program in a separate thread
value_iteration = ValueIterator(g, daemon=True)
value_iteration.start()

# Starting the game mainloop
g.mainloop()