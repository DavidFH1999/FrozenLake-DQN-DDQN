import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import seaborn as sns
import json
import torch
import imageio
from torch import nn
import torch.nn.functional as F
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class FrozenLakeDQL():
    def __init__(self, randomMap):
        # For a 4x4 FrozenLake, we have 16 states
        self.state_visits = np.zeros(16, dtype=int)
        # init loss history
        self.loss_history = []
        # Use the map provided by MapManager
        self.random_map = generate_random_map(size=4)
        # 'dqn' or 'ddqn'
        self.algo = "dqn"
        # change filename according to algo
        self.file_pt = "frozen_lake_dqn.pt"
        self.file_png = "frozen_lake_dqn.png"
        # Save the map to a file
        if randomMap:
            with open("frozen_lake_map.json", "w") as file:
                json.dump(self.random_map, file)
        with open("frozen_lake_map.json", "r") as file:
            self.loaded_map = json.load(file)

    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    def setAlgo(self, algo):
        self.algo = algo
        # change filename according to algo
        self.file_pt = "frozen_lake_dqn.pt" if algo == 'dqn' else "frozen_lake_ddqn.pt"
        self.file_png = "frozen_lake_dqn.png" if algo == 'dqn' else "frozen_lake_ddqn.png"


    # Train the FrozeLake environment
    def train(self, episodes, algo, render=False, is_slippery=False):
        # set DQL Model
        self.setAlgo(algo)

        # Reset self.loss_history at the beginning
        self.loss_history = []
        self.state_visits = np.zeros(16)  # Assuming FrozenLake 4x4 grid
        success_counts = 0  # Counter for successful Episodes
        steps_per_episode = []  # List for steps per Episode

        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', desc=self.loaded_map, is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        if self.algo == "dqn":
            self.plot_all_qvalues(
                dqn=policy_dqn, 
                actions=frozen_lake.ACTIONS, 
                output_file="pre_all_q_values_dqn.png",
                preTraining=True
            )
        elif self.algo == "ddqn":
            self.plot_all_qvalues(
                dqn=policy_dqn, 
                actions=frozen_lake.ACTIONS, 
                output_file="pre_all_q_values_ddqn.png",
                preTraining=True
            )

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            self.state_visits[state] += 1  # Count that we've visited this state
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions   
            steps = 0 

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                steps += 1

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Count new_state visit
                self.state_visits[new_state] += 1

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            steps_per_episode.append(steps)  # Schritte speichern
            if reward == 1:
                success_counts += 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), self.file_pt)

        # Generate metrics for later
        # 1) Sum of rewards (last 100 episodes
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        
        # 2) Cumulative Rewards
        cumulative_rewards = np.cumsum(rewards_per_episode)

        # 3) Smoothed Rewards (moving average)
        def moving_average(data, window_size=100):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        smoothed_rewards = moving_average(rewards_per_episode)
        success_rate = moving_average(rewards_per_episode) * 100  # Prozentwerte
        avg_steps_to_goal = moving_average(steps_per_episode) 

        # save inside dictionary
        metrics = {
            "algo": self.algo,
            "rewards_per_episode": rewards_per_episode,
            "sum_rewards": sum_rewards,
            "cumulative_rewards": cumulative_rewards,
            "epsilon_history": epsilon_history,
            "smoothed_rewards": smoothed_rewards,
            "loss_history": self.loss_history[:],  # Kopie
            "state_visits": self.state_visits.copy(),
            "success_rate": success_rate,  # Erfolgsquote (Moving Average)
            "steps_to_goal": avg_steps_to_goal,

        }

        return metrics

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    if self.algo == 'dqn':
                        target = torch.FloatTensor(
                            reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                        )
                    elif self.algo == 'ddqn':
                        # Use policy network to select the next action
                        next_action = policy_dqn(self.state_to_dqn_input(new_state, num_states)).argmax().item()
                        
                        # Use target network to evaluate the selected action
                        target = torch.FloatTensor(
                            [reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states))[next_action].item()]
                        )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Store loss into loss_history
        self.loss_history.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, algo, is_slippery=False, gif_filename="frozen_lake_test.gif"):
        # set DQL Model
        self.setAlgo(algo)

        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', desc=self.loaded_map, is_slippery=is_slippery, render_mode='rgb_array')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load(self.file_pt, weights_only=True))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        if self.algo == "dqn":
            self.plot_all_qvalues(
                dqn=policy_dqn, 
                actions=frozen_lake.ACTIONS, 
                output_file="post_all_q_values_dqn.png",
                preTraining=False
            )
        elif self.algo == "ddqn":
            self.plot_all_qvalues(
                dqn=policy_dqn, 
                actions=frozen_lake.ACTIONS, 
                output_file="post_all_q_values_ddqn.png",
                preTraining=False
            )

        # List to store frames for GIF creation
        frames = []

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

                # Capture the frame
                frames.append(env.render())

        env.close()
        # Save frames as GIF
        print(f"Saving GIF to {self.algo.upper() + "_" + gif_filename}...")
        imageio.mimsave(self.algo.upper() + "_" + gif_filename, frames, fps=5)  # Adjust fps for playback speed

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states

    def plot_all_qvalues(self, dqn, actions, output_file="all_q_values.png", preTraining=False):
        """
        Creates a 4×4 grid, where each subplot is a bar chart 
        displaying the Q-values for a specific state.
        """
        num_states = dqn.fc1.in_features  # = 16 in 4x4-FrozenLake
        grid_size = int(np.sqrt(num_states))  # 4
        
        # Figure window with 4x4 subplots
        fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(10, 10))
        axes = axes.flatten()

        for s in range(num_states):
            # q-values via dqn and state_to_dqn_input
            with torch.no_grad():
                qvals_tensor = dqn(self.state_to_dqn_input(s, num_states))
            qvals = qvals_tensor.numpy()
            
            # Bar diagramm with 4 Bars
            ax = axes[s]
            bars = ax.bar(actions, qvals, color=['lightcoral' if q < 0 else 'lightblue' for q in qvals], edgecolor='black')
            
            # Highest Bar is orange
            best_action_idx = np.argmax(qvals)
            bars[best_action_idx].set_color('orange')

            # ax Title
            ax.set_title(f"State {s}", fontsize=10)
            
            # Values over/under Graph
            for i, b in enumerate(bars):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + (0.01 if b.get_height() >= 0 else -0.05),
                    f"{qvals[i]:.2f}",
                    ha='center',
                    va='bottom' if b.get_height() >= 0 else 'top', 
                    fontsize=8,
                )
            
            # Scale for negative Values
            ax.set_ylim([min(-0.1, min(qvals) * 1.1), max(0.1, max(qvals) * 1.1)])

        # Title
        if preTraining:
            fig.suptitle(self.algo.upper() + " - random, before training", fontsize=16, weight='bold')
        elif preTraining == False:
            fig.suptitle(self.algo.upper() + " - trained", fontsize=16, weight='bold')

        # Layout Optimization
        fig.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.savefig(output_file)
        plt.close()


def plot_comparison(dqn_metrics, ddqn_metrics, output_filename="comparison.png"):
    """
    Compares DQN vs. DDQN in a single plot (3 rows x 3 columns):
    1) Average Rewards (last 100 episodes)
    2) Epsilon Progression
    3) Cumulative Rewards
    4) Smoothed Rewards
    5) Loss Progression
    6) Heatmaps (Side by Side)
    7) Success Rate (Moving Average)
    8) Steps to Goal

    """

    plt.figure(figsize=(16, 12))
    
    # 1) Average Rewards (Last 100 Episodes)
    plt.subplot(3, 3, 1)
    plt.plot(dqn_metrics["sum_rewards"], label="DQN")
    plt.plot(ddqn_metrics["sum_rewards"], label="DDQN")
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards')
    plt.title('Average Rewards (Last 100 Ep)')
    plt.legend()

    # 2) Epsilon History
    plt.subplot(3, 3, 2)
    plt.plot(dqn_metrics["epsilon_history"], label="DQN")
    plt.plot(ddqn_metrics["epsilon_history"], label="DDQN")
    plt.xlabel('Training Steps')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.legend()

    # 3) Cumulative Rewards
    plt.subplot(3, 3, 3)
    plt.plot(dqn_metrics["cumulative_rewards"], label="DQN")
    plt.plot(ddqn_metrics["cumulative_rewards"], label="DDQN")
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards Accumulated')
    plt.legend()

    # 4) Smoothed Rewards
    plt.subplot(3, 3, 4)
    plt.plot(dqn_metrics["smoothed_rewards"], label="DQN")
    plt.plot(ddqn_metrics["smoothed_rewards"], label="DDQN")
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title('Smoothed Rewards (Moving Avg)')
    plt.legend()

    # 5) Loss
    plt.subplot(3, 3, 5)
    plt.plot(dqn_metrics["loss_history"], label="DQN")
    plt.plot(ddqn_metrics["loss_history"], label="DDQN")
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()

    # 6) Heatmaps
    # DQN
    plt.subplot(3, 3, 6)
    dqn_visits = dqn_metrics["state_visits"].reshape(4,4)
    sns.heatmap(dqn_visits, annot=True, cmap="coolwarm", fmt=".0f")
    plt.title('DQN State Visits')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    # DDQN
    plt.subplot(3, 3, 7)
    ddqn_visits = ddqn_metrics["state_visits"].reshape(4,4)
    sns.heatmap(ddqn_visits, annot=True, cmap="coolwarm", fmt=".0f")
    plt.title('DDQN State Visits')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    # 7) Success Rate (Moving Average)
    plt.subplot(3, 3, 8)
    plt.plot(dqn_metrics["success_rate"], label="DQN")
    plt.plot(ddqn_metrics["success_rate"], label="DDQN")
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate (Moving Avg)')
    plt.legend()

    # 8) Steps to Goal
    plt.subplot(3, 3, 9)
    plt.plot(dqn_metrics["steps_to_goal"], label="DQN")
    plt.plot(ddqn_metrics["steps_to_goal"], label="DDQN")
    plt.xlabel('Episodes')
    plt.ylabel('Steps to Goal')
    plt.title('Average Steps to Goal')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

if __name__ == '__main__':
    is_slippery = False
    randomMap = False
    frozen_lake = FrozenLakeDQL(randomMap)
    
    # ---- DQN ----
    dqn_metrics = frozen_lake.train(episodes=1000, algo="dqn", is_slippery=is_slippery)
    frozen_lake.test(episodes=1, algo="dqn", is_slippery=is_slippery)

    # ---- DDQN ----
    ddqn_metrics = frozen_lake.train(episodes=1000, algo="ddqn", is_slippery=is_slippery)
    frozen_lake.test(episodes=1, algo="ddqn", is_slippery=is_slippery)

    # ---- Plot Comparison ----
    plot_comparison(dqn_metrics, ddqn_metrics, output_filename="comparison_dqn_ddqn.png")