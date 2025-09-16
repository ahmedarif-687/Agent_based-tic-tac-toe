import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import random
import pickle
import os

class TicTacToeRL:
    def __init__(self):
        # Q-learning parameters
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # exploration rate for training
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Game state
        self.reset_game()
        
    def reset_game(self):
        self.board = [0] * 9  # 0: empty, 1: player, -1: agent
        self.game_over = False
        self.winner = None
        
    def get_state(self):
        # Convert board to string for Q-table key
        return ''.join(map(str, self.board))
    
    def get_available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]
    
    def make_move(self, position, player):
        if self.board[position] == 0:
            self.board[position] = player
            return True
        return False
    
    def check_winner(self):
        # Check all winning combinations
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 0:
                return self.board[combo[0]]
        
        if 0 not in self.board:
            return 0  # Tie
        
        return None  # Game continues
    
    def get_reward(self, player):
        winner = self.check_winner()
        if winner == player:
            return 1  # Win
        elif winner == -player:
            return -1  # Loss
        elif winner == 0:
            return 0  # Tie
        else:
            return 0  # Game continues
    
    def get_q_value(self, state, action):
        return self.q_table.get(f"{state}_{action}", 0)
    
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        
        if next_state:
            next_actions = self.get_available_actions()
            if next_actions:
                max_next_q = max([self.get_q_value(next_state, a) for a in next_actions])
            else:
                max_next_q = 0
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[f"{state}_{action}"] = new_q
    
    def choose_action(self, training=True):
        available_actions = self.get_available_actions()
        if not available_actions:
            return None
        
        if training and random.random() < self.epsilon:
            # Exploration
            return random.choice(available_actions)
        else:
            # Exploitation
            state = self.get_state()
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def train(self, episodes=50000):
        print("Training RL agent...")
        wins = 0
        losses = 0
        ties = 0
        
        for episode in range(episodes):
            self.reset_game()
            
            while not self.game_over:
                # Agent's turn (-1)
                state = self.get_state()
                action = self.choose_action(training=True)
                
                if action is not None:
                    self.make_move(action, -1)
                    winner = self.check_winner()
                    
                    if winner is not None:
                        self.game_over = True
                        reward = self.get_reward(-1)
                        self.update_q_value(state, action, reward, None)
                        
                        if winner == -1:
                            wins += 1
                        elif winner == 1:
                            losses += 1
                        else:
                            ties += 1
                        continue
                
                # Random opponent's turn (1) - simulating random player
                available = self.get_available_actions()
                if available:
                    opponent_move = random.choice(available)
                    self.make_move(opponent_move, 1)
                    
                    next_state = self.get_state()
                    winner = self.check_winner()
                    
                    if winner is not None:
                        self.game_over = True
                        reward = self.get_reward(-1)
                        if winner == -1:
                            wins += 1
                        elif winner == 1:
                            losses += 1
                        else:
                            ties += 1
                    else:
                        reward = 0
                    
                    self.update_q_value(state, action, reward, next_state if not self.game_over else None)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if (episode + 1) % 10000 == 0:
                print(f"Episode {episode + 1}: Wins: {wins}, Losses: {losses}, Ties: {ties}")
                wins = losses = ties = 0
        
        print("Training completed!")
        self.epsilon = 0  # No exploration during actual play
    
    def save_model(self, filename="rl_tictactoe_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="rl_tictactoe_model.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {filename}")
            return True
        return False

class TicTacToeGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe - You vs RL Agent")
        self.window.geometry("500x600")
        self.window.configure(bg='lightblue')
        
        # Initialize RL agent
        self.rl_agent = TicTacToeRL()
        
        # Game state
        self.board = [0] * 9  # 0: empty, 1: player (X), -1: agent (O)
        self.player_turn = True
        self.game_over = False
        
        # Stats
        self.player_wins = 0
        self.agent_wins = 0
        self.ties = 0
        
        self.create_ui()
        self.setup_agent()
        
    def setup_agent(self):
        # Try to load existing model
        if not self.rl_agent.load_model():
            # Train new model
            response = messagebox.askyesno("Training", 
                                         "No trained model found. Do you want to train a new RL agent?\n"
                                         "(This may take a minute)")
            if response:
                self.train_agent()
            else:
                messagebox.showinfo("Info", "Using untrained agent. Performance may be poor.")
    
    def train_agent(self):
        # Show training progress
        progress_window = tk.Toplevel(self.window)
        progress_window.title("Training...")
        progress_window.geometry("300x100")
        progress_window.configure(bg='lightblue')
        
        tk.Label(progress_window, text="Training RL Agent...", 
                font=('Arial', 14), bg='lightblue').pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, length=250, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start()
        
        def train_thread():
            self.rl_agent.train(episodes=300000)  # Reduced for faster training
            self.rl_agent.save_model()
            progress_window.destroy()
            messagebox.showinfo("Training Complete", "RL Agent trained successfully!")
        
        # Run training in background
        self.window.after(100, train_thread)
        
    def create_ui(self):
        # Title
        title = tk.Label(self.window, text="Tic Tac Toe vs RL Agent", 
                        font=('Arial', 20, 'bold'), 
                        bg='lightblue', fg='darkblue')
        title.pack(pady=10)
        
        # Stats
        stats_frame = tk.Frame(self.window, bg='lightblue')
        stats_frame.pack(pady=10)
        
        self.stats_label = tk.Label(stats_frame, 
                                   text="You: 0 | Agent: 0 | Ties: 0", 
                                   font=('Arial', 12), 
                                   bg='lightblue', fg='black')
        self.stats_label.pack()
        
        # Info label
        self.info_label = tk.Label(self.window, text="Your turn! (X)", 
                                  font=('Arial', 14), 
                                  bg='lightblue', fg='green')
        self.info_label.pack(pady=10)
        
        # Game board
        board_frame = tk.Frame(self.window, bg='lightblue')
        board_frame.pack(pady=20)
        
        self.buttons = []
        for i in range(3):
            for j in range(3):
                btn = tk.Button(board_frame, text='', 
                               font=('Arial', 20, 'bold'),
                               width=3, height=1,
                               bg='white', fg='black',
                               command=lambda pos=i*3 + j: self.player_move(pos))
                btn.grid(row=i, column=j, padx=2, pady=2)
                self.buttons.append(btn)
        
        # Control buttons
        control_frame = tk.Frame(self.window, bg='lightblue')
        control_frame.pack(pady=20)
        
        new_game_btn = tk.Button(control_frame, text="New Game", 
                                font=('Arial', 12, 'bold'),
                                bg='orange', fg='white',
                                command=self.reset_game)
        new_game_btn.pack(side=tk.LEFT, padx=10)
        
        retrain_btn = tk.Button(control_frame, text="Retrain Agent", 
                               font=('Arial', 12, 'bold'),
                               bg='purple', fg='white',
                               command=self.train_agent)
        retrain_btn.pack(side=tk.LEFT, padx=10)
        
    def player_move(self, position):
        if not self.player_turn or self.game_over or self.board[position] != 0:
            return
        
        # Make player move
        self.board[position] = 1
        self.rl_agent.board = self.board.copy()
        self.buttons[position].config(text='X', fg='blue', bg='lightcyan')
        
        # Check if player won
        winner = self.rl_agent.check_winner()
        if winner == 1:
            self.end_game("🎉 You Won! 🎉", 'green')
            self.player_wins += 1
            self.update_stats()
            return
        elif winner == 0:
            self.end_game("It's a Tie!", 'orange')
            self.ties += 1
            self.update_stats()
            return
        
        # Switch to agent's turn
        self.player_turn = False
        self.info_label.config(text="Agent thinking...", fg='red')
        
        # Agent move with delay
        self.window.after(500, self.agent_move)
    
    def agent_move(self):
        if self.game_over:
            return
        
        # Agent makes move
        self.rl_agent.board = self.board.copy()
        action = self.rl_agent.choose_action(training=False)
        
        if action is not None:
            self.board[action] = -1
            self.rl_agent.board = self.board.copy()
            self.buttons[action].config(text='O', fg='red', bg='mistyrose')
            
            # Check if agent won
            winner = self.rl_agent.check_winner()
            if winner == -1:
                self.end_game("🤖 Agent Won! 🤖", 'red')
                self.agent_wins += 1
                self.update_stats()
                return
            elif winner == 0:
                self.end_game("It's a Tie!", 'orange')
                self.ties += 1
                self.update_stats()
                return
        
        # Switch back to player's turn
        self.player_turn = True
        self.info_label.config(text="Your turn! (X)", fg='green')
    
    def end_game(self, message, color):
        self.game_over = True
        self.info_label.config(text=message, fg=color)
        
        # Disable all buttons
        for btn in self.buttons:
            btn.config(state='disabled')
    
    def reset_game(self):
        self.board = [0] * 9
        self.rl_agent.reset_game()
        self.player_turn = True
        self.game_over = False
        
        self.info_label.config(text="Your turn! (X)", fg='green')
        
        # Reset buttons
        for btn in self.buttons:
            btn.config(text='', bg='white', fg='black', state='normal')
    
    def update_stats(self):
        self.stats_label.config(text=f"You: {self.player_wins} | Agent: {self.agent_wins} | Ties: {self.ties}")
    
    def run(self):
        self.window.mainloop()

# Run the game
if __name__ == "__main__":
    game = TicTacToeGUI()
    game.run()