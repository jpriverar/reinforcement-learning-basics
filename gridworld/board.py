import tkinter as tk

class Board(tk.Canvas):
    def __init__(self, parent, rows, cols,*args, **kwargs):
        # Size in pixels of each square
        self.square_size = 100

        # Initializing the canvas
        super().__init__(parent, width=self.square_size*cols, height=self.square_size*rows, highlightthickness=2, highlightbackground="black", *args, **kwargs)

        # Saving object parent
        self.parent = parent

        # Saving the size of the board
        self.rows = rows
        self.cols = cols

    def init(self, start_position, walls, rewards):
        # Coloring all the square and lines elements
        self.draw_basic_elements(walls, rewards)

        # Drawing the game agent
        y, x = start_position
        self.agent = self.create_bitmap(x*self.square_size+self.square_size//2, y*self.square_size+self.square_size//2, bitmap="@D:/Udemy/Reinforement Learning/gridworld/assets/agent.xbm")

    def draw_basic_elements(self, walls, rewards):
        # Coloring the board walls
        for (y,x) in walls:
            point1 = (x*self.square_size, y*self.square_size)
            point2 = ((x+1)*self.square_size, (y+1)*self.square_size)
            self.create_rectangle(point1, point2, fill="gray")

        # Putting reward values on each space and coloring it accordingly
        self.fill_space_value(rewards)

        # Putting the state (coordinates) on each square
        for i in range(self.rows):
            for j in range(self.cols):
                # Label position
                x = j*self.square_size + 15 # Add a small offset to visulize entirely
                y = i*self.square_size + 10

                # Defining the actual label
                label = f"({i},{j})"
                self.create_text(x, y, text=label)

        # Creating the board grid lines
        # Vertical lines
        for i in range(1, self.cols):
            x_coord = i*self.square_size
            self.create_line(x_coord, 0, x_coord, self.winfo_height(), width=2)

        #Horizontal lines
        for i in range(1, self.rows):
            y_coord = i*self.square_size
            self.create_line(0, y_coord, self.winfo_width(), y_coord, width=2)

    def draw_actions(self, actions):
        for (y,x), action in actions.items():
            # Calculating the offset for the direction of the action arrow
            if action == "U":
                offset1 = (self.square_size//2, self.square_size//4)
                offset2 = (self.square_size//2, -self.square_size//4)

            elif action == "D":
                offset1 = (self.square_size//2, 3*self.square_size//4)
                offset2 = (self.square_size//2, 5*self.square_size//4)

            elif action == "R":
                offset1 = (3*self.square_size//4, self.square_size//2)
                offset2 = (5*self.square_size//4, self.square_size//2)

            elif action == "L":
                offset1 = (self.square_size//4, self.square_size//2)
                offset2 = (-self.square_size//4, self.square_size//2)

            # Getting starting and final points for the arrow
            base = (x*self.square_size, y*self.square_size)
            point1 = (base[0]+offset1[0], base[1]+offset1[1])
            point2 = (base[0]+offset2[0], base[1]+offset2[1])

            # Drawing the arrow itself
            self.create_line(point1, point2, arrow=tk.LAST)
    
    def move_agent(self, state):
        y, x = state
        self.coords(self.agent, x*self.square_size+self.square_size//2, y*self.square_size+self.square_size//2)

    def fill_space_value(self, values):
        for (i,j), value in values.items():
            point1 = (j*self.square_size, i*self.square_size)
            point2 = ((j+1)*self.square_size, (i+1)*self.square_size)

            # Creating the color in hexadecimal
            # If the value is equal to zero then add a bit to every square so it's not completely black
            if value == 0:
                hex_color = "#202020"
            else:
                r = hex(int(value*255)) if (value < 0) else hex(0)
                g = hex(int(value*255)) if (value >= 0) else hex(0)
                b = hex(0)
                hex_color = format_color_space(r,g,b)

            self.create_rectangle(point1, point2, fill=hex_color)

            # Putting the reward in text inside the square
            self.create_text((point1[0]+point2[0])/2, (point1[1]+point2[1])/2, text=f'{value:.4f}')

def format_color_space(r,g,b):
    r = r.split("x")[1] + "0"
    g = g.split("x")[1] + "0"
    b = b.split("x")[1] + "0"
    return f"#{r[0:2]}{g[0:2]}{b[0:2]}"