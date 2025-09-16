Unbeatable Tic-Tac-Toe AI
 
A sophisticated Tic-Tac-Toe AI that combines Q-learning, minimax with alpha-beta pruning, and advanced pattern recognition to create an unbeatable opponent. This AI evolves with every game, rapidly adapting to player strategies, especially diagonal patterns like 0-4-8 and 2-4-6, to ensure it blocks wins and counters effectively. The project features a sleek Tkinter GUI for an engaging user experience.
Features

Unbeatable Gameplay: Uses minimax to guarantee the AI never loses, with immediate win/block checks.
Adaptive Learning: Employs Q-learning to improve strategies, with a focus on countering repeated player patterns (e.g., diagonal wins).
Diagonal Pattern Recognition: Prioritizes detection and countering of diagonal strategies (0-4-8, 2-4-6) with high-weighted bonuses and penalties.
Dynamic Adaptation: Adjusts learning rate, exploration (epsilon), and strategic depth based on player win streaks and AI performance.
GUI Interface: Interactive Tkinter-based UI with real-time AI status updates (intelligence, frustration, win streak).
Persistent Learning: Saves AI knowledge to unbeatable_ai.pkl for continuous improvement across sessions.
Debugging Feedback: Console logs for pattern detection, counter-strategy bonuses, and AI move choices.

Installation

Clone the Repository:
git clone https://github.com/yourusername/unbeatable-tictactoe-ai.git
cd unbeatable-tictactoe-ai


Install Dependencies:Ensure you have Python 3.6+ installed. Install required packages:
pip install numpy tkinter


Run the Game:Execute the main script:
python unbeatable_tictactoe.py



Usage

Start the Game:

Run unbeatable_tictactoe.py to launch the GUI.
A welcome message outlines the AI's capabilities.


Playing:

Click on the 3x3 grid to place your 'X' (player moves first).
The AI responds with 'O', adapting to your strategies, especially diagonal moves.
Monitor the AI's status (intelligence, strategic depth, frustration) and player win streak in the GUI.


Controls:

New Battle: Reset the board for a new game.
Intensive Training: Run 100 simulated games to boost AI learning (useful after player wins).
Reset AI: Erase all learned knowledge and start fresh (use cautiously).


Debugging:

Check the console for logs like 🔍 Detected player patterns: {'0_4_8': 10, '2_4_6': 10} to confirm the AI is countering your diagonal strategies.
Look for messages like 🛑 AI blocking diagonal 0-4-8 with move: 8 to verify blocking moves.



How the AI Works

Minimax Algorithm: Ensures unbeatability by checking for immediate wins or blocks and evaluating future moves.
Q-Learning: Updates a Q-table to learn optimal moves, with adaptive learning rates based on performance.
Pattern Recognition: Detects player patterns (e.g., 0-4-8, 2-4-6 diagonals) with extra weight (+10) and applies counter-strategy bonuses (frequency * 60).
Diagonal Focus: Prioritizes blocking positions 6 or 8 and taking the center (4) to disrupt diagonal setups.
Penalties for Losses: Applies harsh penalties (-150 * player_win_streak) for diagonal wins to accelerate adaptation.
Persistent Storage: Saves learned strategies to unbeatable_ai.pkl for continuity.

Project Structure
unbeatable-tictactoe-ai/
│
├── unbeatable_tictactoe.py  # Main script with AI and GUI logic
├── unbeatable_ai.pkl        # Saved AI model (generated after games)
└── README.md               # This file

Example Gameplay
To win with a diagonal (e.g., 0-4-8), you might try:

Place 'X' at 0, 4, 8.
The AI should counter by taking 4 early (after 0 or 2) or blocking 8/6 if you have two diagonal pieces.
After 1-2 wins, console logs will show 🎯 AI countering diagonal 0-4-8 by taking center: 4.

If you win repeatedly, the AI’s frustration increases, and it adapts faster (check "Player Win Streak" in the GUI).
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/AmazingFeature).
Commit changes (git commit -m 'Add AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a Pull Request.

Please include tests and update documentation for new features.
License
Distributed under the MIT License. See LICENSE for more information.
Contact
For issues or suggestions, open an issue on GitHub or contact yourname@example.com.

Challenge: Can you beat the AI before it masters your diagonal strategies? The evolution begins now! ⏰
