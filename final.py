import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import random
import pickle
import os
import threading
import time
import copy

class UnbeatableAI:
    def __init__(self):
        # Enhanced learning parameters for rapid improvement
        self.q_table = {}
        self.learning_rate = 0.5  # High for fast adaptation
        self.discount_factor = 0.99  # Look further ahead
        self.epsilon = 0.2  # Start with exploration
        self.epsilon_min = 0.005  # Very low minimum for exploitation
        self.epsilon_decay = 0.99  # Adjusted for early exploration
        
        # Advanced memory systems
        self.game_memory = []  # Store complete games
        self.player_patterns = {}  # Track player's favorite moves
        self.opening_book = {}  # Store optimal opening moves
        self.endgame_patterns = {}  # Store endgame scenarios
        
        # Performance tracking
        self.win_rate_history = []
        self.games_analyzed = 0
        self.strategic_depth = 1  # How many moves ahead to think
        
        # Adaptive parameters
        self.frustration_level = 0.0
        self.confidence_level = 0.5
        self.learning_intensity = 1.0
        self.pattern_recognition_strength = 1.0
        
        # Meta-learning
        self.successful_strategies = {}
        self.failed_strategies = {}
        
        # Track repeated player wins and specific patterns
        self.player_win_streak = 0
        self.repeated_pattern_penalties = {}
        
        self.reset_game()
        
    def reset_game(self):
        self.board = [0] * 9
        self.game_over = False
        self.move_history = []  # Track all moves this game
        self.state_action_pairs = []
        
    def get_state(self):
        """Enhanced state representation including game phase"""
        base_state = ''.join(map(str, self.board))
        move_count = sum(1 for x in self.board if x != 0)
        
        if move_count <= 2:
            phase = "opening"
        elif move_count <= 6:
            phase = "middle"
        else:
            phase = "endgame"
            
        return f"{base_state}_{phase}"
    
    def get_available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]
    
    def make_move(self, position, player):
        if self.board[position] == 0:
            self.board[position] = player
            self.move_history.append((position, player))
            return True
        return False
    
    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 0:
                return self.board[combo[0]]
        
        if 0 not in self.board:
            return 0
        return None
    
    def evaluate_position(self, board, player):
        """Advanced position evaluation with diagonal emphasis"""
        score = 0
        
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for combo in winning_combinations:
            line = [board[i] for i in combo]
            
            if line.count(player) == 3:
                return 1000
            elif line.count(-player) == 3:
                return -1000
            elif line.count(player) == 2 and line.count(0) == 1:
                score += 150 if combo in [[0, 4, 8], [2, 4, 6]] else 100  # Higher for diagonals
            elif line.count(player) == 1 and line.count(0) == 2:
                score += 15 if combo in [[0, 4, 8], [2, 4, 6]] else 10
            elif line.count(-player) == 2 and line.count(0) == 1:
                score -= 120 if combo in [[0, 4, 8], [2, 4, 6]] else 80  # Higher penalty for diagonal threats
        
        # Strategic position values
        center_bonus = 50 if board[4] == player else 0  # Increased center value
        corner_bonus = sum(20 for pos in [0, 2, 6, 8] if board[pos] == player)
        
        return score + center_bonus + corner_bonus
    
    def minimax_with_learning(self, board, depth, maximizing, alpha=-float('inf'), beta=float('inf')):
        """Enhanced minimax with alpha-beta pruning"""
        winner = self.check_winner_for_board(board)
        
        if winner == -1:  # AI wins
            return 1000 - depth
        elif winner == 1:  # Player wins
            return -1000 + depth
        elif winner == 0 or depth >= self.strategic_depth + 2:
            return self.evaluate_position(board, -1)
        
        available = [i for i in range(9) if board[i] == 0]
        
        if maximizing:
            max_eval = -float('inf')
            for move in available:
                new_board = board.copy()
                new_board[move] = -1
                eval_score = self.minimax_with_learning(new_board, depth + 1, False, alpha, beta)
                
                q_bonus = self.q_table.get(f"{''.join(map(str, new_board))}_{move}", 0) * 0.1
                eval_score += q_bonus
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in available:
                new_board = board.copy()
                new_board[move] = 1
                eval_score = self.minimax_with_learning(new_board, depth + 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def check_winner_for_board(self, board):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for combo in winning_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] != 0:
                return board[combo[0]]
        
        if 0 not in board:
            return 0
        return None
    
    def analyze_player_patterns(self):
        """Analyze player's patterns, prioritize diagonals 0-4-8 and 2-4-6"""
        if len(self.game_memory) < 3:
            return {}
        
        patterns = {}
        for game in self.game_memory[-20:]:
            player_moves = [move for move, player in game if player == 1]
            
            # Opening preferences
            if player_moves:
                first_move = player_moves[0]
                patterns[f"opening_{first_move}"] = patterns.get(f"opening_{first_move}", 0) + 1
            
            # Move sequences (focus on 0-4-8 and 2-4-6)
            for i in range(len(player_moves) - 2):
                sequence = f"{player_moves[i]}_{player_moves[i+1]}_{player_moves[i+2]}"
                patterns[sequence] = patterns.get(sequence, 0) + 1
                if sequence in ["0_4_8", "2_4_6"]:
                    patterns[sequence] += 10  # Extra weight for diagonals
            
            # Track winning patterns
            if self.determine_game_winner(game) == 1:
                final_state = [0] * 9
                for move, player in game:
                    final_state[move] = player
                state_key = ''.join(map(str, final_state))
                patterns[f"win_{state_key}"] = patterns.get(f"win_{state_key}", 0) + 1
                # Extra weight for diagonal wins
                if final_state[0] == 1 and final_state[4] == 1 and final_state[8] == 1:
                    patterns[f"win_{state_key}"] += 10
                if final_state[2] == 1 and final_state[4] == 1 and final_state[6] == 1:
                    patterns[f"win_{state_key}"] += 10
        
        # Log detected patterns
        print("🔍 Detected player patterns:", {k: v for k, v in patterns.items() if v > 2 or k.startswith("win_") or k in ["0_4_8", "2_4_6"]})
        return patterns
    
    def get_counter_strategy_bonus(self, action):
        """Bonus for moves that counter player patterns, especially diagonals"""
        patterns = self.analyze_player_patterns()
        bonus = 0
        
        # Counter common openings
        for pattern, frequency in patterns.items():
            if pattern.startswith("opening_") and frequency > 2:
                common_opening = int(pattern.split("_")[1])
                if self.counters_opening(action, common_opening):
                    bonus += frequency * 20
            
            # Counter diagonal winning patterns
            if pattern.startswith("win_") and frequency > 1:
                win_state = pattern.split("_")[1]
                if self.board[int(action)] == 0 and self.would_block_win(action, win_state):
                    bonus += frequency * 60  # Increased for diagonals
            
            # Counter specific diagonal sequences
            if pattern in ["0_4_8", "2_4_6"] and frequency > 1:
                if action in [4, 8] and pattern == "0_4_8":
                    bonus += frequency * 50
                if action in [4, 6] and pattern == "2_4_6":
                    bonus += frequency * 50
        
        # Additional bonus for player win streak
        if self.player_win_streak > 1:
            bonus += self.player_win_streak * 50
        
        # Bonus for blocking diagonal threats
        if self.board[0] == 1 and self.board[4] == 1 and action == 8:
            bonus += 100
        if self.board[2] == 1 and self.board[4] == 1 and action == 6:
            bonus += 100
        
        print(f"🎯 Counter-strategy bonus for action {action}: {bonus}")
        return bonus
    
    def would_block_win(self, action, win_state):
        """Check if action blocks a winning state"""
        temp_board = self.board.copy()
        temp_board[action] = -1
        return ''.join(map(str, temp_board)) != win_state
    
    def counters_opening(self, action, opening_move):
        """Check if action counters a common opening, prioritize center"""
        counters = {
            0: [4, 8],  # Counter 0-4-8 diagonal
            2: [4, 6],  # Counter 2-4-6 diagonal
            4: [0, 1, 2, 3, 5, 6, 7, 8],
            6: [4, 2],
            8: [4, 0],
            1: [4, 7],
            3: [4, 5],
            5: [4, 3],
            7: [4, 1]
        }
        return action in counters.get(opening_move, [])
    
    def choose_action(self, training=True):
        available_actions = self.get_available_actions()
        if not available_actions:
            return None
        
        state = self.get_state()
        
        # Immediate win check
        for action in available_actions:
            temp_board = self.board.copy()
            temp_board[action] = -1
            if self.check_winner_for_board(temp_board) == -1:
                print(f"🏆 AI found winning move: {action}")
                return action
        
        # Immediate block check
        for action in available_actions:
            temp_board = self.board.copy()
            temp_board[action] = 1
            if self.check_winner_for_board(temp_board) == 1:
                print(f"🛑 AI blocking player's win with move: {action}")
                return action
        
        # Specific counter to diagonal setups
        if self.board[0] == 1 and 4 in available_actions:
            print(f"🎯 AI countering diagonal 0-4-8 by taking center: 4")
            return 4
        if self.board[2] == 1 and 4 in available_actions:
            print(f"🎯 AI countering diagonal 2-4-6 by taking center: 4")
            return 4
        if self.board[0] == 1 and self.board[4] == 1 and 8 in available_actions:
            print(f"🛑 AI blocking diagonal 0-4-8 with move: 8")
            return 8
        if self.board[2] == 1 and self.board[4] == 1 and 6 in available_actions:
            print(f"🛑 AI blocking diagonal 2-4-6 with move: 6")
            return 6
        
        # Use minimax earlier
        if self.confidence_level > 0.3 and len(available_actions) <= 8:  # Lowered threshold
            best_move = None
            best_score = -float('inf')
            
            for action in available_actions:
                temp_board = self.board.copy()
                temp_board[action] = -1
                score = self.minimax_with_learning(temp_board, 0, False)
                
                q_bonus = self.get_q_value(state, action) * 0.2
                pattern_bonus = self.get_counter_strategy_bonus(action) * self.pattern_recognition_strength
                total_score = score + q_bonus + pattern_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_move = action
            
            if best_move is not None:
                print(f"♟️ AI chose minimax move: {best_move} (score: {best_score})")
                return best_move
        
        # Epsilon-greedy with pattern recognition
        adaptive_epsilon = self.epsilon * (1 - self.confidence_level)
        
        if training and random.random() < adaptive_epsilon:
            strategic_moves = []
            for action in available_actions:
                if action == 4:
                    strategic_moves.extend([action] * 4)  # Strong preference for center
                elif action in [0, 2, 6, 8]:
                    strategic_moves.extend([action] * 2)
                else:
                    strategic_moves.append(action)
            move = random.choice(strategic_moves)
            print(f"🔄 AI exploring with move: {move}")
            return move
        else:
            best_actions = []
            best_score = -float('inf')
            
            for action in available_actions:
                q_val = self.get_q_value(state, action)
                strategic_bonus = 0
                if action == 4:
                    strategic_bonus = 20  # Increased for center
                elif action in [0, 2, 6, 8]:
                    strategic_bonus = 10
                pattern_bonus = self.get_counter_strategy_bonus(action)
                aggression_bonus = self.frustration_level * 5
                
                total_score = q_val + strategic_bonus + pattern_bonus + aggression_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_actions = [action]
                elif abs(total_score - best_score) < 0.01:
                    best_actions.append(action)
            
            move = random.choice(best_actions)
            print(f"🎯 AI exploiting with move: {move} (score: {best_score})")
            return move
    
    def get_q_value(self, state, action):
        return self.q_table.get(f"{state}_{action}", 0)
    
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        
        if next_state and not self.game_over:
            next_actions = self.get_available_actions()
            if next_actions:
                max_next_q = max([self.get_q_value(next_state, a) for a in next_actions])
            else:
                max_next_q = 0
        else:
            max_next_q = 0
        
        adaptive_lr = self.learning_rate * self.learning_intensity * (1 + self.frustration_level * 0.5)
        new_q = current_q + adaptive_lr * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[f"{state}_{action}"] = new_q
    
    def intensive_learning_session(self, game_result):
        """Extra learning for non-wins, target diagonal losses"""
        if game_result != -1:
            self.strategic_depth = min(4, self.strategic_depth + 1)
            replay_count = min(5, max(2, int(self.frustration_level * 3)))
            
            for _ in range(replay_count):
                for i, (state, action) in enumerate(self.state_action_pairs):
                    temporal_penalty = (len(self.state_action_pairs) - i) * 3
                    base_penalty = -100 if game_result == 1 else -30  # Harsher for losses
                    total_reward = base_penalty - temporal_penalty
                    
                    # Target diagonal wins
                    if game_result == 1 and state in self.repeated_pattern_penalties:
                        total_reward += self.repeated_pattern_penalties[state]
                    
                    self.update_q_value(state, action, total_reward, None)
            
            # Learn from successful games
            winning_games = [game for game in self.game_memory if self.determine_game_winner(game) == -1]
            for game in winning_games[-5:]:
                self.replay_successful_game(game)
    
    def determine_game_winner(self, game_moves):
        board = [0] * 9
        for move, player in game_moves:
            board[move] = player
        return self.check_winner_for_board(board)
    
    def replay_successful_game(self, game_moves):
        board = [0] * 9
        ai_moves = []
        
        for move, player in game_moves:
            if player == -1:
                state = ''.join(map(str, board))
                ai_moves.append((state, move))
            board[move] = player
        
        for i, (state, action) in enumerate(ai_moves):
            reward = 15 + (i + 1) * 3
            self.update_q_value(state, action, reward, None)
    
    def learn_from_game(self, game_result):
        print(f"🧠 AI analyzing game result: {game_result}")
        
        self.game_memory.append(self.move_history.copy())
        if len(self.game_memory) > 50:
            self.game_memory.pop(0)
        
        self.games_analyzed += 1
        
        # Track diagonal wins
        if game_result == 1:
            final_state = [0] * 9
            for move, player in self.move_history:
                final_state[move] = player
            state_key = ''.join(map(str, final_state))
            if (final_state[0] == 1 and final_state[4] == 1 and final_state[8] == 1) or \
               (final_state[2] == 1 and final_state[4] == 1 and final_state[6] == 1):
                self.repeated_pattern_penalties[state_key] = self.repeated_pattern_penalties.get(state_key, 0) - 150 * self.player_win_streak
        
        if game_result == -1:
            self.win_rate_history.append(1)
            self.frustration_level = max(0, self.frustration_level - 0.3)
            self.confidence_level = min(1.0, self.confidence_level + 0.1)
            self.player_win_streak = 0
            print("🏆 Victory! Confidence increased.")
        elif game_result == 1:
            self.win_rate_history.append(-1)
            self.frustration_level = min(3.0, self.frustration_level + 0.8)
            self.confidence_level = max(0.1, self.confidence_level - 0.2)
            self.player_win_streak += 1
            print(f"😡 Defeat! Frustration increased. Player win streak: {self.player_win_streak}")
        else:
            self.win_rate_history.append(0)
            self.frustration_level = min(3.0, self.frustration_level + 0.3)
            self.player_win_streak = max(0, self.player_win_streak - 1)
            print("😐 Tie. Need to improve strategy.")
        
        if len(self.win_rate_history) > 20:
            self.win_rate_history.pop(0)
        
        recent_wins = sum(1 for x in self.win_rate_history if x == 1)
        recent_games = len(self.win_rate_history)
        win_rate = recent_wins / recent_games if recent_games > 0 else 0
        
        if win_rate < 0.6 or self.player_win_streak > 1:
            self.learning_intensity = min(3.0, self.learning_intensity + 0.5)
            self.pattern_recognition_strength = min(2.0, self.pattern_recognition_strength + 0.3)
            self.strategic_depth = min(4, self.strategic_depth + 2)  # Faster depth increase
            print("📈 Increasing learning intensity due to low win rate or player streak!")
        elif win_rate > 0.8:
            self.learning_intensity = max(0.8, self.learning_intensity - 0.1)
            print("🎯 Maintaining strategic advantage.")
        
        for i, (state, action) in enumerate(self.state_action_pairs):
            base_reward = self.calculate_move_reward(action, game_result, i)
            if game_result == 1 and state in self.repeated_pattern_penalties:
                base_reward += self.repeated_pattern_penalties[state]
            next_state = None
            if i < len(self.state_action_pairs) - 1:
                next_state = self.state_action_pairs[i + 1][0]
            
            self.update_q_value(state, action, base_reward, next_state)
        
        if game_result != -1:
            self.intensive_learning_session(game_result)
        
        if self.frustration_level > 2.0 or self.player_win_streak > 1:
            self.epsilon *= 0.85
        elif win_rate > 0.7:
            self.epsilon *= 0.95
        else:
            self.epsilon *= self.epsilon_decay
        
        self.epsilon = max(self.epsilon, self.epsilon_min)
        self.learning_rate = max(0.1, self.learning_rate * 0.99)
        
        print(f"📊 Stats - Frustration: {self.frustration_level:.2f}, Confidence: {self.confidence_level:.2f}")
        print(f"🎯 Learning intensity: {self.learning_intensity:.2f}, Strategic depth: {self.strategic_depth}")
        print(f"🧠 Knowledge base: {len(self.q_table)} states learned")
    
    def calculate_move_reward(self, action, game_result, move_index):
        base_reward = 0
        
        if game_result == -1:
            base_reward = 20 + move_index * 2
        elif game_result == 1:
            base_reward = -30 - (len(self.state_action_pairs) - move_index) * 3
        else:
            base_reward = 5
        
        if action == 4:
            base_reward += 10  # Increased for center
        elif action in [0, 2, 6, 8]:
            base_reward += 5
        
        return base_reward * self.learning_intensity
    
    def record_move(self, state, action):
        self.state_action_pairs.append((state, action))
    
    def save_model(self, filename="unbeatable_ai.pkl"):
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'frustration_level': self.frustration_level,
            'confidence_level': self.confidence_level,
            'learning_intensity': self.learning_intensity,
            'strategic_depth': self.strategic_depth,
            'game_memory': self.game_memory,
            'win_rate_history': self.win_rate_history,
            'games_analyzed': self.games_analyzed,
            'pattern_recognition_strength': self.pattern_recognition_strength,
            'player_win_streak': self.player_win_streak,
            'repeated_pattern_penalties': self.repeated_pattern_penalties
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"🧠 Unbeatable AI saved! Games analyzed: {self.games_analyzed}")
    
    def load_model(self, filename="unbeatable_ai.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data.get('q_table', {})
                    self.epsilon = data.get('epsilon', 0.1)
                    self.frustration_level = data.get('frustration_level', 0)
                    self.confidence_level = data.get('confidence_level', 0.5)
                    self.learning_intensity = data.get('learning_intensity', 1.0)
                    self.strategic_depth = data.get('strategic_depth', 1)
                    self.game_memory = data.get('game_memory', [])
                    self.win_rate_history = data.get('win_rate_history', [])
                    self.games_analyzed = data.get('games_analyzed', 0)
                    self.pattern_recognition_strength = data.get('pattern_recognition_strength', 1.0)
                    self.player_win_streak = data.get('player_win_streak', 0)
                    self.repeated_pattern_penalties = data.get('repeated_pattern_penalties', {})
                print(f"🤖 Loaded unbeatable AI! {self.games_analyzed} games analyzed")
                return True
            except:
                print("⚠️ Failed to load AI model, starting fresh")
                return False
        return False

class UnbeatableGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🧠 UNBEATABLE AI - Evolves to Perfection!")
        self.window.geometry("700x900")
        self.window.configure(bg='#0a0a0a')
        
        self.ai = UnbeatableAI()
        self.board = [0] * 9
        self.player_turn = True
        self.game_over = False
        self.player_wins = 0
        self.ai_wins = 0
        self.ties = 0
        self.games_played = 0
        
        self.create_ui()
        self.ai.load_model()
        
    def create_ui(self):
        title_frame = tk.Frame(self.window, bg='#0a0a0a')
        title_frame.pack(pady=15)
        
        title = tk.Label(title_frame, text="🧠 UNBEATABLE AI", 
                        font=('Arial', 30, 'bold'), bg='#0a0a0a', fg='#ff073a')
        title.pack()
        
        subtitle = tk.Label(title_frame, text="⚡ Self-Learning • Pattern Recognition • Strategic Evolution", 
                           font=('Arial', 12, 'italic'), bg='#0a0a0a', fg='#ffa502')
        subtitle.pack()
        
        status_frame = tk.LabelFrame(self.window, text="🤖 AI EVOLUTION STATUS", 
                                    font=('Arial', 13, 'bold'), 
                                    bg='#0a0a0a', fg='#00d2d3', 
                                    relief='ridge', bd=3)
        status_frame.pack(pady=15, padx=20, fill='x')
        
        self.intelligence_status = tk.Label(status_frame, text="🧠 Intelligence: Beginner", 
                                          font=('Arial', 11, 'bold'), bg='#0a0a0a', fg='#2ed573')
        self.intelligence_status.pack(pady=2)
        
        self.strategy_status = tk.Label(status_frame, text="♟️ Strategic Depth: Level 1", 
                                       font=('Arial', 10), bg='#0a0a0a', fg='#70a1ff')
        self.strategy_status.pack()
        
        self.learning_status = tk.Label(status_frame, text="📚 Learning Intensity: Normal", 
                                       font=('Arial', 10), bg='#0a0a0a', fg='#ffa502')
        self.learning_status.pack()
        
        self.pattern_status = tk.Label(status_frame, text="🎯 Pattern Recognition: Basic", 
                                      font=('Arial', 10), bg='#0a0a0a', fg='#ff6b6b')
        self.pattern_status.pack()
        
        self.confidence_status = tk.Label(status_frame, text="💪 Confidence: Uncertain", 
                                         font=('Arial', 10), bg='#0a0a0a', fg='#ffd32a')
        self.confidence_status.pack()
        
        self.frustration_status = tk.Label(status_frame, text="😐 Frustration: Calm", 
                                          font=('Arial', 10), bg='#0a0a0a', fg='#70a1ff')
        self.frustration_status.pack()
        
        self.streak_status = tk.Label(status_frame, text="🏆 Player Win Streak: 0", 
                                     font=('Arial', 10), bg='#0a0a0a', fg='#ff6b6b')
        self.streak_status.pack()
        
        stats_frame = tk.LabelFrame(self.window, text="📊 BATTLE STATISTICS", 
                                   font=('Arial', 12, 'bold'), 
                                   bg='#0a0a0a', fg='#ffd32a',
                                   relief='ridge', bd=2)
        stats_frame.pack(pady=10, padx=20, fill='x')
        
        self.stats_label = tk.Label(stats_frame, 
                                   text="🏆 Human: 0 | 🤖 AI: 0 | 🤝 Ties: 0", 
                                   font=('Arial', 12, 'bold'), 
                                   bg='#0a0a0a', fg='#ffffff')
        self.stats_label.pack(pady=5)
        
        self.winrate_label = tk.Label(stats_frame, text="📈 AI Win Rate: 0%", 
                                     font=('Arial', 11), bg='#0a0a0a', fg='#ff6b6b')
        self.winrate_label.pack()
        
        self.info_label = tk.Label(self.window, text="🎯 Your move! AI is studying your patterns...", 
                                  font=('Arial', 14, 'bold'), 
                                  bg='#0a0a0a', fg='#2ed573')
        self.info_label.pack(pady=15)
        
        board_frame = tk.Frame(self.window, bg='#0a0a0a')
        board_frame.pack(pady=20)
        
        self.buttons = []
        for i in range(3):
            for j in range(3):
                btn = tk.Button(board_frame, text='', 
                               font=('Arial', 32, 'bold'),
                               width=3, height=1,
                               bg='#1a1a1a', fg='#ffffff',
                               activebackground='#2a2a2a',
                               relief='raised', bd=4,
                               command=lambda pos=i*3 + j: self.player_move(pos))
                btn.grid(row=i, column=j, padx=5, pady=5)
                self.buttons.append(btn)
        
        control_frame = tk.Frame(self.window, bg='#0a0a0a')
        control_frame.pack(pady=25)
        
        new_game_btn = tk.Button(control_frame, text="🔄 New Battle", 
                                font=('Arial', 12, 'bold'),
                                bg='#3742fa', fg='white', width=12,
                                command=self.reset_game)
        new_game_btn.pack(side=tk.LEFT, padx=8)
        
        train_btn = tk.Button(control_frame, text="🚀 Intensive Training", 
                             font=('Arial', 12, 'bold'),
                             bg='#ff6348', fg='white', width=15,
                             command=self.run_training_session)
        train_btn.pack(side=tk.LEFT, padx=8)
        
        reset_ai_btn = tk.Button(control_frame, text="🧠 Reset AI", 
                                font=('Arial', 12, 'bold'),
                                bg='#ff3838', fg='white', width=10,
                                command=self.reset_ai)
        reset_ai_btn.pack(side=tk.LEFT, padx=8)
        
    def run_training_session(self):
        """Run intensive training against strong opponent"""
        def train():
            original_text = self.info_label.cget('text')
            self.info_label.config(text="🚀 AI INTENSIVE TRAINING IN PROGRESS...", fg='#ff073a')
            
            for _ in range(100):
                training_ai = UnbeatableAI()
                training_ai.q_table = self.ai.q_table.copy()
                
                board = [0] * 9
                game_moves = []
                current_player = 1
                
                while True:
                    available = [i for i in range(9) if board[i] == 0]
                    if not available:
                        break
                    
                    if current_player == 1:
                        training_ai.board = board.copy()
                        move = training_ai.choose_action(training=False)
                        if move is None:
                            move = random.choice(available)
                    else:
                        training_ai.board = board.copy()
                        move = training_ai.choose_action(training=True)
                        if move is None:
                            break
                    
                    board[move] = current_player
                    game_moves.append((move, current_player))
                    
                    winner = training_ai.check_winner_for_board(board)
                    if winner is not None:
                        break
                    
                    current_player *= -1
                
                training_ai.move_history = game_moves
                ai_moves = [(i, move, player) for i, (move, player) in enumerate(game_moves) if player == -1]
                
                for i, move, player in ai_moves:
                    state = ''.join(map(str, board))
                    training_ai.state_action_pairs.append((state, move))
                
                training_ai.learn_from_game(winner if winner is not None else 0)
                
                for key, value in training_ai.q_table.items():
                    if key in self.ai.q_table:
                        self.ai.q_table[key] = (self.ai.q_table[key] + value) / 2
                    else:
                        self.ai.q_table[key] = value
            
            self.ai.strategic_depth = min(4, self.ai.strategic_depth + 1)
            self.ai.confidence_level = min(1.0, self.ai.confidence_level + 0.2)
            self.ai.pattern_recognition_strength = min(2.0, self.ai.pattern_recognition_strength + 0.3)
            
            self.update_ai_status()
            self.info_label.config(text="🚀 TRAINING COMPLETE! AI evolved significantly!", fg='#2ed573')
            self.ai.save_model()
            
            self.window.after(3000, lambda: self.info_label.config(text=original_text, fg='#2ed573'))
        
        threading.Thread(target=train, daemon=True).start()
    
    def update_ai_status(self):
        q_size = len(self.ai.q_table)
        games = self.ai.games_analyzed
        
        if q_size < 50 or games < 5:
            intel_text = "🧠 Intelligence: Novice"
            color = '#70a1ff'
        elif q_size < 200 or games < 15:
            intel_text = "🧠 Intelligence: Learning"
            color = '#ffa502'
        elif q_size < 500 or games < 30:
            intel_text = "🧠 Intelligence: Competent"
            color = '#ff6348'
        elif q_size < 1000 or games < 50:
            intel_text = "🧠 Intelligence: Advanced"
            color = '#ff073a'
        elif q_size < 2000 or games < 100:
            intel_text = "🧠 Intelligence: Expert"
            color = '#8b00ff'
        else:
            intel_text = "🧠 Intelligence: UNBEATABLE"
            color = '#00ff00'
        
        self.intelligence_status.config(text=intel_text, fg=color)
        
        depth = self.ai.strategic_depth
        if depth >= 3:
            strategy_text = f"♟️ Strategic Depth: MASTER (Level {depth})"
            color = '#ff073a'
        elif depth >= 2:
            strategy_text = f"♟️ Strategic Depth: Advanced (Level {depth})"
            color = '#ff6348'
        else:
            strategy_text = f"♟️ Strategic Depth: Basic (Level {depth})"
            color = '#70a1ff'
        
        self.strategy_status.config(text=strategy_text, fg=color)
        
        intensity = self.ai.learning_intensity
        if intensity > 2.5:
            learning_text = "📚 Learning: OVERDRIVE"
            color = '#ff073a'
        elif intensity > 2.0:
            learning_text = "📚 Learning: INTENSE"
            color = '#ff6348'
        elif intensity > 1.5:
            learning_text = "📚 Learning: HIGH"
            color = '#ffa502'
        else:
            learning_text = "📚 Learning: STANDARD"
            color = '#2ed573'
        
        self.learning_status.config(text=learning_text, fg=color)
        
        pattern_strength = self.ai.pattern_recognition_strength
        if pattern_strength > 1.7:
            pattern_text = "🎯 Pattern Recognition: PSYCHIC"
            color = '#8b00ff'
        elif pattern_strength > 1.4:
            pattern_text = "🎯 Pattern Recognition: ADVANCED"
            color = '#ff073a'
        elif pattern_strength > 1.1:
            pattern_text = "🎯 Pattern Recognition: GOOD"
            color = '#ffa502'
        else:
            pattern_text = "🎯 Pattern Recognition: BASIC"
            color = '#70a1ff'
        
        self.pattern_status.config(text=pattern_text, fg=color)
        
        confidence = self.ai.confidence_level
        if confidence > 0.8:
            conf_text = "💪 Confidence: DOMINANT"
            color = '#2ed573'
        elif confidence > 0.6:
            conf_text = "💪 Confidence: HIGH"
            color = '#ffa502'
        elif confidence > 0.4:
            conf_text = "💪 Confidence: MODERATE"
            color = '#70a1ff'
        else:
            conf_text = "💪 Confidence: UNCERTAIN"
            color = '#ff6b6b'
        
        self.confidence_status.config(text=conf_text, fg=color)
        
        frustration = self.ai.frustration_level
        if frustration > 2.5:
            frust_text = "😡 Frustration: ENRAGED!"
            color = '#ff073a'
        elif frustration > 2.0:
            frust_text = "😠 Frustration: FURIOUS"
            color = '#ff6348'
        elif frustration > 1.5:
            frust_text = "😤 Frustration: ANGRY"
            color = '#ffa502'
        elif frustration > 0.8:
            frust_text = "🙂 Frustration: MILD"
            color = '#70a1ff'
        else:
            frust_text = "😌 Frustration: CALM"
            color = '#2ed573'
        
        self.frustration_status.config(text=frust_text, fg=color)
        
        streak = self.ai.player_win_streak
        if streak >= 3:
            streak_text = f"🏆 Player Win Streak: {streak} - AI is furious!"
            color = '#ff073a'
        elif streak >= 2:
            streak_text = f"🏆 Player Win Streak: {streak} - AI is adapting!"
            color = '#ff6348'
        elif streak >= 1:
            streak_text = f"🏆 Player Win Streak: {streak} - AI is alert!"
            color = '#ffa502'
        else:
            streak_text = f"🏆 Player Win Streak: {streak} - AI is calm"
            color = '#2ed573'
        
        self.streak_status.config(text=streak_text, fg=color)
    
    def player_move(self, position):
        if not self.player_turn or self.game_over or self.board[position] != 0:
            return
        
        self.board[position] = 1
        self.ai.board = self.board.copy()
        self.buttons[position].config(text='X', fg='#2ed573', bg='#0a3d0a')
        
        winner = self.ai.check_winner()
        if winner == 1:
            self.end_game("🎉 HUMAN VICTORY! AI analyzing defeat...", '#2ed573')
            self.player_wins += 1
            self.update_stats()
            threading.Thread(target=lambda: self.ai.learn_from_game(1), daemon=True).start()
            return
        elif winner == 0:
            self.end_game("🤝 TIE! AI studying for improvements...", '#ffd32a')
            self.ties += 1
            self.update_stats()
            threading.Thread(target=lambda: self.ai.learn_from_game(0), daemon=True).start()
            return
        
        self.player_turn = False
        self.info_label.config(text="🤖 AI computing optimal strategy...", fg='#ff6b6b')
        self.window.after(1500, self.ai_move)
    
    def ai_move(self):
        if self.game_over:
            return
        
        state = self.ai.get_state()
        self.ai.board = self.board.copy()
        action = self.ai.choose_action(training=True)
        
        if action is not None:
            self.ai.record_move(state, action)
            self.board[action] = -1
            self.ai.board = self.board.copy()
            self.buttons[action].config(text='O', fg='#ff6b6b', bg='#3d0a0a')
            
            winner = self.ai.check_winner()
            if winner == -1:
                self.end_game("🤖 AI VICTORY! Evolution successful!", '#ff6b6b')
                self.ai_wins += 1
                self.update_stats()
                threading.Thread(target=lambda: self.ai.learn_from_game(-1), daemon=True).start()
                return
            elif winner == 0:
                self.end_game("🤝 TIE! AI will do better next time...", '#ffd32a')
                self.ties += 1
                self.update_stats()
                threading.Thread(target=lambda: self.ai.learn_from_game(0), daemon=True).start()
                return
        
        self.player_turn = True
        self.info_label.config(text=f"🎯 Your move! AI is countering your diagonal strategy... (Streak: {self.ai.player_win_streak})", fg='#2ed573')
    
    def end_game(self, message, color):
        self.game_over = True
        self.games_played += 1
        self.info_label.config(text=message, fg=color)
        
        for btn in self.buttons:
            btn.config(state='disabled')
        
        self.window.after(1500, self.update_ai_status)
        self.window.after(2500, self.ai.save_model)
    
    def reset_game(self):
        self.board = [0] * 9
        self.ai.reset_game()
        self.player_turn = True
        self.game_over = False
        
        if self.ai.confidence_level > 0.7:
            message = "🎯 New battle! AI is confident and ready..."
        elif self.ai.frustration_level > 1.5 or self.ai.player_win_streak > 1:
            message = f"😡 New game! AI is frustrated and dangerous... (Player streak: {self.ai.player_win_streak})"
        else:
            message = "🔄 Fresh start! AI has learned from previous games..."
        
        self.info_label.config(text=message, fg='#2ed573')
        
        for btn in self.buttons:
            btn.config(text='', bg='#1a1a1a', fg='#ffffff', state='normal')
    
    def reset_ai(self):
        response = messagebox.askyesno("🧠 Reset Unbeatable AI", 
                                     "⚠️ This will erase ALL AI evolution!\n\n"
                                     "The AI will lose all its knowledge,\n"
                                     "strategic depth, and pattern recognition.\n\n"
                                     "Are you absolutely sure?")
        if response:
            self.ai = UnbeatableAI()
            self.update_ai_status()
            messagebox.showinfo("🧠 AI Reset", "AI completely reset! Back to square one.")
    
    def update_stats(self):
        self.stats_label.config(
            text=f"🏆 Human: {self.player_wins} | 🤖 AI: {self.ai_wins} | 🤝 Ties: {self.ties}"
        )
        
        if self.games_played > 0:
            ai_win_rate = (self.ai_wins / self.games_played) * 100
            if ai_win_rate >= 80:
                rate_text = f"📈 AI Win Rate: {ai_win_rate:.1f}% - DOMINATING!"
                color = '#ff073a'
            elif ai_win_rate >= 60:
                rate_text = f"📈 AI Win Rate: {ai_win_rate:.1f}% - Strong"
                color = '#ff6348'
            elif ai_win_rate >= 40:
                rate_text = f"📈 AI Win Rate: {ai_win_rate:.1f}% - Balanced"
                color = '#ffa502'
            else:
                rate_text = f"📈 AI Win Rate: {ai_win_rate:.1f}% - Learning"
                color = '#70a1ff'
        else:
            rate_text = "📈 AI Win Rate: No games yet"
            color = '#ffffff'
        
        self.winrate_label.config(text=rate_text, fg=color)
        self.update_ai_status()
    
    def run(self):
        welcome_msg = """🧠 WELCOME TO UNBEATABLE AI!

🔥 FEATURES:
• Advanced Q-Learning with Minimax Strategy
• Enhanced Diagonal Pattern Recognition
• Adaptive Learning Intensity
• Multi-Level Strategic Depth
• Frustration-Based Evolution
• Game Memory & Replay Learning

⚡ This AI:
✓ Never makes the same mistake twice
✓ Rapidly learns your diagonal patterns
✓ Gets stronger after every game
✓ Uses advanced strategic algorithms
✓ Counters 0-4-8 and 2-4-6 diagonals

🏆 CHALLENGE: Can you beat it before it becomes truly unbeatable?

The evolution begins NOW! ⏰"""
        
        messagebox.showinfo("🧠 Unbeatable AI", welcome_msg)
        self.window.mainloop()

if __name__ == "__main__":
    print("🧠 Starting Unbeatable Tic-Tac-Toe AI...")
    print("🔥 Advanced learning algorithms initializing...")
    print("⚡ Evolution mode: ACTIVE")
    game = UnbeatableGUI()
    game.run()