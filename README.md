# MicroGo

MicroGo is a simplified version of the traditional Go game, implemented on a 5x5 board with novel mechanics such as stone expiration and limited stone counts. The project features an AI powered by a Convolutional Neural Network (CNN) and Monte Carlo Tree Search (MCTS), trained using supervised learning and reinforcement learning (RL) through self-play. The AI competes effectively against random agents and supports interactive human-AI gameplay via an in-notebook interface.

This project was developed as part of an Artificial Intelligence course by Soem (22k-4629), Danish Raza (22k-4183), and Muhammad Danish (22k-4381) under the supervision of Sir Abdullah Yaqoob.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training and Evaluation](#training-and-evaluation)
- [Contributing](#contributing)
- [References](#references)

## Features
- **Modified Go Rules**: 5x5 board, stones expire after three turns, each player has 30 stones.
- **AI Implementation**: CNN-based neural network (`GoNNet`) with MCTS for strategic move selection.
- **Training Pipeline**: Supervised pretraining on 10,000 random games, followed by RL via self-play.
- **Interactive Interface**: In-notebook HTML/JavaScript interface for human-AI gameplay in Google Colab.
- **Performance**: RL-trained AI achieves a 96% win rate against random agents and 73.3% against the pretrained model.

## Installation
MicroGo is implemented in Python and requires a GPU-enabled environment (e.g., Google Colab with CUDA) for optimal performance. Follow these steps to set up the project:

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or a local environment with GPU support
- Git (for cloning the repository)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/microgo.git
   cd microgo
   ```

2. **Install Dependencies**:
   Run the following command in your Python environment or Colab notebook:
   ```bash
   !pip install numpy torch coloredlogs tqdm matplotlib
   ```

3. **Verify GPU Support**:
   Ensure PyTorch is using CUDA (if available). The code automatically detects GPU availability:
   ```python
   import torch
   print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
   ```

4. **Download Pretrained Models** (optional):
   If available, download pretrained checkpoints (`pretrained_policy.pth`, `rl_iter_X.pth`) and place them in the `checkpoints/` directory.

## Usage
The project is structured as a Jupyter notebook (`microgo.ipynb`) for easy execution in Google Colab. Key components include game logic, neural network, MCTS, training, and an interactive interface.

### Running the Notebook
1. Open `microgo.ipynb` in Google Colab.
2. Execute cells sequentially to:
   - Install dependencies.
   - Define game logic (`Board`, `GoGame`).
   - Initialize and train the neural network (`GoNNet`).
   - Run MCTS-guided self-play and evaluation.
   - Launch the interactive game interface.

3. To play against the AI:
   - Run the cell under "In-notebook MicroGo visual interface".
   - Click on the 5x5 grid to place stones (White). The AI (Black) responds automatically.
   - The game ends with a winner declaration based on the territory score.

### Example Commands
- **Generate Random Games**:
  ```python
  game = GoGame(n=5)
  random_data = generate_random_games(game, num_games=10000)
  ```

- **Train the Model**:
  Pretraining and RL training are handled in the notebook cells. Adjust hyperparameters in the training cell:
  ```python
  TRAIN_ITERS = 8
  GAMES_PER_ITER = 60
  args_mcts = {'numMCTSSims': 40, 'cpuct': 1.5}
  ```

- **Evaluate AI**:
  Run the evaluation cell to compare AI performance:
  ```python
  w, l, d = quick_series(rl_agent, random_agent, games=50)
  print(f"RL latest vs Random → wins:{w} loses:{l} draws:{d}")
  ```

## Project Structure
```
microgo/
├── microgo.ipynb           # Main notebook with game logic, AI, and interface
├── checkpoints/            # Directory for model checkpoints
│   ├── pretrained_policy.pth
│   └── rl_iter_X.pth
├── random_games.pkl        # Cached random game data
└── README.md              # Project documentation
```

### Key Modules
1. **Board and Game Logic**: `Board` and `GoGame` classes manage the 5x5 board, stone expiration, and scoring.
2. **Neural Network**: `GoNNet` with residual blocks predicts move probabilities and game outcomes.
3. **MCTS**: Monte Carlo Tree Search for move selection, guided by the neural network.
4. **Data Augmentation**: Board rotations and flips to enhance training data.
5. **Self-Play and Training**: Supervised pretraining and RL-based self-play for AI improvement.
6. **Interactive Interface**: HTML/JavaScript-based UI for human-AI gameplay.

## Training and Evaluation
- **Pretraining**: Uses 120,000 samples from 600,000 random game states, achieving 8% policy accuracy after 12 epochs.
- **RL Training**: 8 iterations with 60 self-play games per iteration, optimizing policy and value losses.
- **Evaluation**:
  - Pretrained AI: 90% win rate vs. random agent (50 games).
  - RL-trained AI: 96% win rate vs. random agent, 73.3% vs. pretrained AI (30 games).
  - Decision time: ~0.1 seconds per move on GPU.

To retrain or fine-tune:
1. Modify `TRAIN_ITERS`, `GAMES_PER_ITER`, or `args_mcts` in the training cell.
2. Run the training cells to generate new checkpoints.
3. Evaluate using the provided `quick_series` function.


---

**Authors**: Soem (22k-4629), Danish Raza (22k-4183), Muhammad Danish (22k-4381)  
**Course**: Artificial Intelligence  
**Instructor**: Sir Abdullah Yaqoob
