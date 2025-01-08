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
- Python (3.8 or higher)
- WSL (Windows Subsystem for Linux) with Ubuntu or a similar Linux distribution installed.
- pip (Python package manager)

## Setup Instructions
Follow these steps to set up the project on a Windows machine with WSL:

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/frozen-lake-dql.git
cd frozen-lake-dql
```

### Step 2: Create a Virtual Environment
Create a virtual environment to isolate dependencies.
```bash
python3 -m venv venv
source venv/bin/activate  # Activate the virtual environment
```

### Step 3: Install Dependencies
Install the required Python packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Additional Tools (Optional)
For advanced visualization:
```bash
pip install seaborn matplotlib
```

### Step 5: Run the Training Script
To train the agent with DQN and DDQN:
```bash
python main.py
```

### Requirements File
Ensure your `requirements.txt` contains the following:
```
gymnasium
numpy
matplotlib
seaborn
json
torch
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
