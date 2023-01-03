import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from gridworld.gridworld import standard_grid

class PolicyImprovement(threading.Thread):
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
                self.value_s[state].append(self.__bellman(state, discount_factor))

                # Update the delta value
                delta = max(delta, abs(self.value_s[state][-1] - value_old))

            # Check if all deltas have crossed the threshold
            if delta < delta_threshold: 
                print(f"Value function has converged after {self.steps} steps")
                break

    def __bellman(self, state, gamma, best_action=False):
        # Setting the agent on the state to calculate
        self.g.set_state(state)

        # List possible actions in the current state
        actions = self.g.get_actions(state)

        # Action-value for all the possible actions
        action_values = []
        
        # Initializing the value to zero
        value = 0
        for action in actions:
            action_val = 0
            for next_state in self.g.get_all_states():
                # Reward is deterministic so no need to sum over it
                # value(s) = policy(a|s) * probs(s'|a,s) * [r + gamma*value(s')]
                action_val += (next_state == self.g.get_next_state(state, action)) * (self.g.rewards.get(next_state,0) + gamma*self.value_s[next_state][-1])

            # Saving the action_val for the current action only
            action_values.append(action_val)
            value += (action == self.policy[state]) * action_val

        #print(f"{state}: {actions}, {action_values}")

        # If best action wanted return argmax
        if best_action:
            return actions[np.argmax(action_values)]
        
        # Else return the total sum of future rewards
        else:
            return value

    def improve_policy(self, gamma):
        print("Improving policy...")

        # To know if the policy has actually changed
        diff_actions = 0

        # Getting all playable states
        all_states = self.g.get_all_states()

        for state in all_states:
            # Only trying different actions on non-terminal states
            if self.g.is_terminal(state): continue

            # Getting the best possible action based on the current value function
            old_action = self.policy[state]
            new_action = self.__bellman(state, 0.9, best_action=True)

            # Updating the policy
            self.policy[state] = new_action

            # Updating the stability flag
            if old_action != new_action: diff_actions += 1

        print(f"Improvement attemp finished, {diff_actions} actions changed")

        # If the policy is stable then there must be zero changes in the policy
        return (diff_actions == 0)

    # Main function runinng the algorithm -> To improve the policy
    def run_algorithm(self):
        # Wait a moment for the gridworld instance to start
        time.sleep(3)

        # Defining a delta threshold and a discount factor
        discount_factor = 0.9
        delta_threshold = 0.001

        # Main loop to improve policy until getting optimal
        while True:
            # Calculating the value function for the current policy
            self.find_value_function(discount_factor=discount_factor, delta_threshold=delta_threshold)
            
            # Show the initial computed final values on the gridboard
            self.g.show_values_on_board(self.value_s, self.policy)

            # Improving the policy once
            stable = self.improve_policy(discount_factor)

            # Giving the user some seconds to visualize results before displaying new results
            time.sleep(2)
            self.g.reset()
            self.g.show_values_on_board(self.value_s, self.policy)

            # If the policy is stable then break out
            if stable: 
                print("Optimal value function achieved...")
                break

        

# Creating a standard grid object
g = standard_grid()

# Starting algorithm program in a separate thread
policy_improver = PolicyImprovement(g, daemon=True)
policy_improver.start()

# Starting the game mainloop
g.mainloop()

'''
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
'''