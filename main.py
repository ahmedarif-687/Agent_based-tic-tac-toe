import tkinter as tk
from tkinter import messagebox
import random

class TicTacToe:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe - You vs Computer")
        self.window.geometry("400x500")
        self.window.configure(bg='lightblue')
        
        # Game state
        self.board = ['' for _ in range(9)]
        self.player = 'X'  # Player is X
        self.computer = 'O'  # Computer is O
        self.current_player = self.player
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        # Title
        title = tk.Label(self.window, text="Tic Tac Toe", 
                        font=('Arial', 24, 'bold'), 
                        bg='lightblue', fg='darkblue')
        title.pack(pady=20)
        
        # Info label
        self.info_label = tk.Label(self.window, text="Your turn! (X)", 
                                  font=('Arial', 14), 
                                  bg='lightblue', fg='green')
        self.info_label.pack(pady=10)
        
        # Game board frame
        board_frame = tk.Frame(self.window, bg='lightblue')
        board_frame.pack(pady=20)
        
        # Create 9 buttons for the board
        self.buttons = []
        for i in range(3):
            for j in range(3):
                btn = tk.Button(board_frame, text='', 
                               font=('Arial', 20, 'bold'),
                               width=3, height=1,
                               bg='white', fg='black',
                               command=lambda row=i, col=j: self.player_move(row*3 + col))
                btn.grid(row=i, column=j, padx=2, pady=2)
                self.buttons.append(btn)
        
        # Reset button
        reset_btn = tk.Button(self.window, text="New Game", 
                             font=('Arial', 14, 'bold'),
                             bg='orange', fg='white',
                             command=self.reset_game)
        reset_btn.pack(pady=20)
        
    def player_move(self, position):
        # Only allow move if it's player's turn and position is empty
        if self.current_player == self.player and self.board[position] == '':
            self.board[position] = self.player
            self.buttons[position].config(text=self.player, fg='blue')
            
            # Check if player won
            if self.check_winner(self.player):
                self.info_label.config(text="🎉 You Won! 🎉", fg='green')
                self.disable_buttons()
                messagebox.showinfo("Game Over", "Congratulations! You won!")
                return
            
            # Check if it's a tie
            if self.is_board_full():
                self.info_label.config(text="It's a Tie!", fg='orange')
                messagebox.showinfo("Game Over", "It's a tie!")
                return
            
            # Switch to computer's turn
            self.current_player = self.computer
            self.info_label.config(text="Computer's turn...", fg='red')
            
            # Computer makes move after a short delay
            self.window.after(500, self.computer_move)
    
    def computer_move(self):
        # Simple computer AI - random move
        available_positions = [i for i in range(9) if self.board[i] == '']
        
        if available_positions:
            position = random.choice(available_positions)
            self.board[position] = self.computer
            self.buttons[position].config(text=self.computer, fg='red')
            
            # Check if computer won
            if self.check_winner(self.computer):
                self.info_label.config(text="💻 Computer Won! 💻", fg='red')
                self.disable_buttons()
                messagebox.showinfo("Game Over", "Computer won! Try again!")
                return
            
            # Check if it's a tie
            if self.is_board_full():
                self.info_label.config(text="It's a Tie!", fg='orange')
                messagebox.showinfo("Game Over", "It's a tie!")
                return
            
            # Switch back to player's turn
            self.current_player = self.player
            self.info_label.config(text="Your turn! (X)", fg='green')
    
    def check_winner(self, player):
        # Check all winning combinations
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in winning_combinations:
            if all(self.board[i] == player for i in combo):
                # Highlight winning combination
                for i in combo:
                    self.buttons[i].config(bg='lightgreen')
                return True
        return False
    
    def is_board_full(self):
        return '' not in self.board
    
    def disable_buttons(self):
        for btn in self.buttons:
            btn.config(state='disabled')
    
    def reset_game(self):
        # Reset game state
        self.board = ['' for _ in range(9)]
        self.current_player = self.player
        self.info_label.config(text="Your turn! (X)", fg='green')
        
        # Reset buttons
        for btn in self.buttons:
            btn.config(text='', bg='white', state='normal')
    
    def run(self):
        self.window.mainloop()

# Run the game
if __name__ == "__main__":
    game = TicTacToe()
    game.run()