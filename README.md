# Frozen Lake Deep Q-Learning (DQN & DDQN)

This repository contains an implementation of Deep Q-Learning (DQN) and Double Deep Q-Learning (DDQN) to solve the Frozen Lake environment from OpenAI Gym. The implementation includes training and testing scripts, visualization of Q-values, and comparison metrics between DQN and DDQN algorithms.

## Project Description
The Frozen Lake environment is a gridworld where the agent must navigate from a starting point to a goal while avoiding holes. The implementation leverages PyTorch to train neural networks for decision-making and uses replay memory for experience replay. This project demonstrates how reinforcement learning techniques can be applied to solve simple toy problems.

## Features
- Train and test DQN and DDQN algorithms on the Frozen Lake environment.
- Visualize Q-values before and after training.
- Compare performance metrics between DQN and DDQN.
- Adjustable hyperparameters and customizable environment settings (e.g., slippery or non-slippery).

## Prerequisites
Ensure you have the following installed on your Windows system:
- Python (3.12 or higher)
- pip (Python package manager)

---

## Setup Instructions
Follow these steps to set up the project on a Windows machine:

### Step 1: Install Miniconda (Recommended) or Anaconda
Download and install Miniconda (recommended) or Anaconda from the link below:

[Miniconda/Anaconda Download](https://www.anaconda.com/download/success)

---

### Step 2: Create a Virtual Environment
Create a virtual environment to isolate dependencies.

1. Open the **Anaconda Prompt** (Miniconda/Anaconda).
2. Navigate to a folder or disk of your choice.
3. Run the following commands:
   ```bash
   conda create -n gymenv
   conda activate gymenv
   conda install python=3.12
   ```

---

### Step 3: Install Dependencies
Install the Frozen Lake environment and other dependencies:

1. Install the Frozen Lake environment:
   ```bash
   pip install gymnasium[toy-text]
   ```

2. Install additional dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 4: Visual Studio Code Setup
Set up **Visual Studio Code** for the project:

1. Install Visual Studio Code if it is not already installed.
2. Open Visual Studio Code.
3. Press `Ctrl + Shift + P` and search for **Python: Select Interpreter**.
4. Select the Python interpreter associated with the `gymenv` environment.
5. Open the terminal in Visual Studio Code and run:
   ```bash
   conda activate gymenv
   ```

6. Clone the project repository:
   ```bash
   git clone https://github.com/DavidFH1999/FrozenLake-DQN-DDQN.git
   ```

---

### Step 5: Run the Training Script
To train the agent with DQN and DDQN:

1. Navigate to the project directory:
   ```bash
   cd FrozenLake-DQN-DDQN
   ```

2. Run the training script:
   ```bash
   python frozen_lake_DQN_DDQN.py
   ```

---

### Notes:
1. Ensure Python 3.12 is compatible with all the libraries in `requirements.txt`. Some libraries may not yet support Python 3.12, so consider using Python 3.10 or 3.11 if issues arise.
2. If `pip install gymnasium[toy-text]` fails, make sure you have `pip` updated:
   ```bash
   python -m pip install --upgrade pip
   ```

### Requirements File
Ensure your `requirements.txt` contains the following:
```
gymnasium
numpy
matplotlib
seaborn
torch
imageio
```

## File Structure
- `main.py`: Main script to run training and testing.
- `frozen_lake_dql.py`: Contains the implementation of DQN, DDQN, and supporting functions.
- `requirements.txt`: List of required dependencies.
- `README.md`: Documentation for the repository.
- `frozen_lake_map.json`: Saved environment map for reproducibility.
- `*.png`: Visualization outputs.
- `*.pt`: Trained model weights.

## How to Customize
- Modify hyperparameters like learning rate, epsilon decay, and memory size in the `FrozenLakeDQL` class.
- Change the Frozen Lake map by editing `frozen_lake_map.json` or enabling random map generation.

## Visualizations
1. **Q-Value Heatmaps:** Compare state visit counts for DQN and DDQN.
2. **Performance Metrics:** Plot cumulative rewards, smoothed rewards, and epsilon decay.
3. **Steps to Goal:** Observe how efficiently the agent reaches the goal over time.
