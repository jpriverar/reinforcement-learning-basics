import tkinter as tk

class Controls(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, bg="red", *args, **kwargs)

        # Saving the object parent
        self.parent = parent

        # Creating the control buttons to play
        # Movement buttons
        self.left_button = tk.Button(self, text="<-", cursor="hand2",command=lambda:self.parent.parent.move("L"), bitmap="@D:/Udemy/Reinforement Learning/dynamic_programming/gridworld/assets/left_arrow.xbm")
        self.right_button = tk.Button(self, text="->", cursor="hand2", command=lambda:self.parent.parent.move("R"), bitmap="@D:/Udemy/Reinforement Learning/dynamic_programming/gridworld/assets/right_arrow.xbm")
        self.up_button = tk.Button(self, text="UP", cursor="hand2", command=lambda:self.parent.parent.move("U"), bitmap="@D:/Udemy/Reinforement Learning/dynamic_programming/gridworld/assets/up_arrow.xbm")
        self.down_button = tk.Button(self, text="DOWN", cursor="hand2", command=lambda:self.parent.parent.move("D"), bitmap="@D:/Udemy/Reinforement Learning/dynamic_programming/gridworld/assets/down_arrow.xbm")

        # Undo and reset buttons
        self.undo_button = tk.Button(self, text="UNDO", cursor="hand2", command=lambda:self.parent.parent.undo_move(), bitmap="@D:/Udemy/Reinforement Learning/dynamic_programming/gridworld/assets/undo_arrow.xbm")
        self.reset_button = tk.Button(self, text="RESET", cursor="hand2", command=lambda:self.parent.parent.reset())

        # Packing the buttons into the frame
        self.left_button.grid(row=0, column=0, padx=2, pady=4)
        self.up_button.grid(row=0, column=1, padx=2, pady=4)
        self.down_button.grid(row=0, column=2, padx=2, pady=4)
        self.right_button.grid(row=0, column=3, padx=2, pady=4)
        self.undo_button.grid(row=0, column=5, padx=2, pady=4)
        self.reset_button.grid(row=0, column=6, padx=2, pady=4)

    def update_available_actions(self, actions):
        # Disable all action buttons
        self.left_button["state"] = "disabled"
        self.right_button["state"] = "disabled"
        self.up_button["state"] = "disabled"
        self.down_button["state"] = "disabled"

        # Re-enable buttons for available actions
        for action in actions:
            if action == "L": self.left_button["state"] = "normal"
            elif action == "R": self.right_button["state"] = "normal"
            elif action == "U": self.up_button["state"] = "normal"
            elif action == "D": self.down_button["state"] = "normal"

