import tkinter as tk
from .main_frame import MainFrame
import os
import numpy as np

class Gridworld(tk.Tk):
    def __init__(self, rows, columns, *args, **kwargs):
        #Initializing the tkinter root window
        super().__init__(*args, **kwargs)

        # Changing the window title
        self.winfo_toplevel().title("Gridworld")

        # Saving the rows and columns
        self.rows = rows
        self.cols = columns

        # Log of performed actions so far
        self.last_move = []

        # Creating the main frame with canvas and controls
        self.main_frame = MainFrame(self, rows, columns)
        self.main_frame.pack()

    def setup(self, start_position, walls, rewards, probabilities, terminals, step_cost=0):
        # Setup all game properties like rewards, obstacles and environment dynamics
        # Check the start_position is inside the game area
        if (0 <= start_position[1] < self.cols) and (0 <= start_position[0] < self.rows):
            self.i = start_position[0]
            self.j = start_position[1]

        # Check that all walls are inside the game area
        for (y,x) in walls: # List of tuples with the wall coordinate
            if not (0 <= x < self.cols) or not (0 <= y < self.rows) or (y,x) == start_position: 
                raise Exception("Declared wall outside of game area or in starting position")

        # Check that all the rewards are inside the game area
        for (y,x), _ in rewards.items(): # Dictionary with key tuple of coordinates and value equal to the reward
            if not (0 <= x < self.cols) or not (0 <= y < self.rows): 
                raise Exception("Declared reward outside of game area")

        # Check all terminal states are inside the game area
        for (y,x) in terminals: # List of tuples with the wall coordinate
            if not (0 <= x < self.cols) or not (0 <= y < self.rows): 
                raise Exception("Declared terminal state outside of game area or in starting position")

        # Check that all probabilities are inside the game area
        # key = (s, a), value = {s': p(s'|s,a)}
        for (state, action), next_state_probs in probabilities.items():
            # Check first if the starting state is inside the game area
            y, x = state
            if not (0 <= x < self.cols) or not (0 <= y < self.rows):
                raise Exception(f"Declared probability for a state {state} outside of game area")

            # Then check if the action is allowed for the starting state
            if action not in self.get_actions(state):
                raise Exception(f"Declared probability for an invalid action '{action}' in the state {state}")

            # Finally check that all the next states are inside the game area
            for next_state, next_prob in next_state_probs.items():
                y, x = next_state
                if not (0 <= x < self.cols) or not (0 <= y < self.rows):
                    raise Exception(f"Declared probability to move to a state {next_state} outside of game area")

        # Saving setup info
        self.start = start_position
        self.walls = walls
        self.rewards = rewards
        self.probabilities = probabilities
        self.terminal = terminals
        self.step_cost = step_cost

        # Setting up the step cost as rewards for all the non declared reward states
        for state in self.get_all_states():
            # If state has not a declared reward, assign the step cost to it
            if state not in self.rewards:
                self.rewards[state] = self.step_cost

        true_rewards = {key:val for key,val in self.rewards.items() if key in self.terminal}
        # Updating all widget information
        self.update()
        self.main_frame.controls.update_available_actions(self.get_actions(self.current_state()))
        self.main_frame.board.init(start_position, walls, true_rewards)

    def update_frame(self):
        # Update the available moves in the controls
        self.main_frame.controls.update_available_actions(self.get_actions(self.current_state()))

        # Move the agent in the canvas
        self.main_frame.board.move_agent(self.current_state())

    def get_actions(self, state):
        # Returns the available actions given a certain state
        actions = []

        # If we are in terminal state, game is over, no actions available
        if self.is_terminal(state): return actions

        # Checking to the left
        if state[1] > 0 and (state[0], state[1]-1) not in self.walls:
            actions.append("L")

        # Checking to the right
        if state[1] < self.cols-1 and (state[0], state[1]+1) not in self.walls:
            actions.append("R")

        # Checking upwards
        if state[0] > 0 and (state[0]-1, state[1]) not in self.walls:
            actions.append("U")

        # Checking downwards
        if state[0] < self.rows-1 and (state[0]+1, state[1]) not in self.walls:
            actions.append("D")

        return actions

    def current_state(self):
        # Return the state where the agent is currently in
        return (self.i, self.j)

    def set_state(self, state):
        self.i = state[0]
        self.j = state[1]

    def move(self, action):
        # Save the current state to be able to go back if needed
        self.last_move.append(self.current_state())

        # Performs a move of the agent and updates it in canvas
        # Checking first of all it's a valid action
        if action in self.get_actions((self.i, self.j)):

            # If probabilities were deterministic, simply change state based on action
            if not self.probabilities:
                if action == "L": self.j -= 1
                elif action == "R": self.j += 1
                elif action == "U": self.i -= 1
                elif action == "D": self.i += 1

            # However, if probabilities were defined, then move based on them
            else:
                # Then getting the next state based on the defined probabilities
                # p(s'|s,a)
                next_state_probs = self.probabilities[self.current_state(), action]
                next_states = list(next_state_probs.keys())
                next_probs = list(next_state_probs.values())

                # Choosing the actual next state
                next_state = next_states[np.random.choice(len(next_states), p=next_probs)]

                # Setting the agent state to the chosen state
                self.set_state(next_state)

            # Updating the frame elements
            self.update_frame()

        # Get the correspoding reward from the move
        # If reward in that state is declared, return it, otherwise return 0 reward
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self):
        if len(self.last_move) == 0:
            print("No moves to undo!")
            return

        # Removing the last saved moved, for it has been undone
        last_state = self.last_move.pop()
        print(last_state)

        # Moving the agent to the previous state
        self.set_state(last_state)

        # Updating the frame elements
        self.update_frame()

    def get_next_state(self, state, action):
        # Returns what the next state would be given the current state and a certain action
        # Only useful when gridworld is deterministic
        i, j = state
        if action in self.get_actions(state):
            if action == "L": j -= 1
            elif action == "R": j += 1
            elif action == "U": i -= 1
            elif action == "D": i += 1

        # Return hypothetic state
        return i,j 

    def is_terminal(self, state):
        # Returns whether the current state is a terminal state
        return state in self.terminal

    def game_over(self):
        # Returns whether the game is over by reaching a terminal state
        return self.current_state() in self.terminal

    def reset(self):
        self.set_state(self.start)
        self.update_frame()
        self.last_move = []
        self.main_frame.board.delete("all")
        true_rewards = {key:val for key,val in self.rewards.items() if key in self.terminal}
        self.main_frame.board.init(self.start, self.walls, true_rewards)

    def get_all_states(self):
        states = []
        for i in range(self.rows):
            for j in range(self.cols):
                state = (i,j)
                # Return all states that are not a wall and are not terminal states
                if state not in self.walls: states.append(state)

        return states

    def show_values_on_board(self, value_s, policy=None):
        values = {key:val for key, val in value_s.items() if key not in self.terminal}
        self.main_frame.board.fill_space_value(values)

        # Redrawing basic elements on board that were deleted by the value colors
        # We don't want to draw rewards related to step cost
        true_rewards = {key:val for key,val in self.rewards.items() if key in self.terminal}
        self.main_frame.board.draw_basic_elements(self.walls, true_rewards)

        # Drawing the policy as arrows in the board
        if policy is not None:
            self.main_frame.board.draw_actions(policy)

def standard_grid(rows=3, cols=4, windy=False, probabilities=None, step_cost=0, wind_dir="R", wind_strength=0.1):
    # Creating gridworld instance
    g = Gridworld(rows,cols)

    # Defining game assets
    start_position = (2,0)
    walls = [(1,1)]
    rewards = {(0,3):1, 
                (1,3):-1}
    terminals = [(0,3), (1,3)]

    # If the game is deterministic (not windy), then there's no need to define
    # environment dynamics as probabilities, otherwise, we need to pass in such
    # probabilities
    reset_probs = False

    if not windy:
        probabilities = dict()
    if windy:
        if probabilities is None:
            raise Exception("Probabilities are required for a windy gridworld")
        elif probabilities == "Auto":
            probabilities = dict()
            reset_probs = True

    # Initializing the game characteristics
    g.setup(start_position, walls, rewards, probabilities, terminals, step_cost)

    # Reassigning probabilities
    if reset_probs:
        random_probs = generate_windy_probabilities(g, wind_dir=wind_dir, wind_strenght=wind_strength)
        g.setup(start_position, walls, rewards, random_probs, terminals, step_cost)

    # Setting fixed size for the board
    g.resizable(False, False)

    # Return the gridworld object
    return g

def generate_windy_probabilities(gridworld, wind_dir, wind_strenght):
    # Probabilities dictionary structure
    # {(state, action):{next_state:probability}}
    # key = (s,a)
    # value = (s':p(s'|s,a))
    probs = dict()

    # Defining probabilities for all states
    all_states = gridworld.get_all_states()

    for state in all_states:
        # Defining probabilities for all actions in the current state
        all_actions = gridworld.get_actions(state)

        for action in all_actions:
            probs[state, action] = dict()

            # If we are going up, then take the risk to move the wind direction, only if it's an allowed move
            if action == "U" and wind_dir in all_actions:
                # Wind strength cannot be greater than 1
                wind_strenght = min(wind_strenght, 1)

                # Probability to go in the wind direction will be at most 0.75
                wind_prob = wind_strenght*0.75
                action_prob = 1-wind_prob

                # Assigning the computed probabilities
                probs[state, action][gridworld.get_next_state(state, action)] = action_prob
                probs[state, action][gridworld.get_next_state(state, wind_dir)] = wind_prob

            else:
                probs[state, action][gridworld.get_next_state(state, action)] = 1
                
    return probs

if __name__ == "__main__":
    # Making sure we are in the right path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Creating gridworld instance
    gridworld = Gridworld(3,4)

    # Defining game assets
    start_position = (2,0)
    walls = [(1,1)]
    rewards = {(0,3):1, 
               (1,3):-1}
    terminals = [(0,3), (1,3)]
    probabilities = dict()

    # Initializing the game assets
    gridworld.setup(start_position, walls, rewards, probabilities, terminals)

    # Setting fixed size for the board
    gridworld.resizable(False, False)

    # Starting the game in a non-blocking thread
    gridworld.mainloop()