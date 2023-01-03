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

    def setup(self, start_position, walls, rewards, probabilities, terminals):
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
        
        # Updating all widget information
        self.update()
        self.main_frame.controls.update_available_actions(self.get_actions(self.current_state()))
        self.main_frame.board.init(start_position, walls, rewards)

    def update_frame(self):
        # Update the available moves in the controls
        self.main_frame.controls.update_available_actions(self.get_actions(self.current_state()))

        # Move the agent in the canvas
        self.main_frame.board.move_agent(self.current_state())

    def get_actions(self, state):
        # Returns the available actions given a certain state
        actions = []

        # If we are in terminal state, game is over, no actions available
        if self.game_over(): return actions

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
                next_state_probs = self.probabilities[self.current_state, action]
                next_states = next_state_probs.keys()
                next_probs = next_state_probs.values()

                # Choosing the actual next state
                next_state = np.random.choice(next_states, p=next_probs)

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
        self.last_move = []
        self.main_frame.board.delete("all")
        self.main_frame.board.init(self.start, self.walls, self.rewards)

    def get_all_states(self):
        states = []
        for i in range(self.rows):
            for j in range(self.cols):
                state = (i,j)
                # Return all states that are not a wall and are not terminal states
                if state not in self.walls: states.append(state)

        return states

    def show_values_on_board(self, value_s, policy=None):
        values = {key:val[-1] for key, val in value_s.items() if key not in self.terminal}
        self.main_frame.board.fill_space_value(values)

        # Redrawing basic elements on board that were deleted by the value colors
        self.main_frame.board.draw_basic_elements(self.walls, self.rewards)

        # Drawing the policy as arrows in the board
        if policy is not None:
            self.main_frame.board.draw_actions(policy)

def standard_grid(windy=False, probabilities=None):
    # Creating gridworld instance
    g = Gridworld(3,4)

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
    g.setup(start_position, walls, rewards, probabilities, terminals)

    # Reassigning probabilities
    if reset_probs:
        random_probs = generate_random_probabilities(g)
        g.setup(start_position, walls, rewards, random_probs, terminals)

    # Setting fixed size for the board
    g.resizable(False, False)

    # Return the gridworld object
    return g

def generate_random_probabilities(gridworld):
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

        # Defining probabilities for all posible states to land given the current state and the current action
        next_possible_states = [gridworld.get_next_state(state,action) for action in all_actions]

        for action in all_actions:
            probs[state, action] = dict()

            # Accumulating probabilities for normalization later on
            total_probability = 0
            for next_state in next_possible_states:
                # Defining the actual probability
                raw_probability = np.random.randint(0,11)
                probs[state,action][next_state] = raw_probability

                # Adding probability to the total sum
                total_probability += raw_probability

            # Afterwards, divide every probability over the total probablity
            # This way probability for all next states will add up to 1
            for next_state in next_possible_states:
                probs[state,action][next_state] /= total_probability

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