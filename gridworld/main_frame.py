import tkinter as tk
from .board import Board
from .controls import Controls

class MainFrame(tk.Frame):
    def __init__(self, parent, rows, columns, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Saving object parent
        self.parent = parent

        # Creating the board canvas to draw
        self.board = Board(self, rows, columns)
        self.board.grid(row=0, column=0, padx=2, pady=2, sticky="NSEW")

        # Creating the controls frame
        self.controls = Controls(self)
        self.controls.grid(row=1, column=0, padx=2, pady=2, sticky="NSEW")
