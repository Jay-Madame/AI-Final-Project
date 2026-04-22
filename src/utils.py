"""
    [] Utility functions for DQN LunarLander
    
    Your job is to implement the following functions:
    
    1. plot_rewards(csv_path, save_path):
        - Read the training log CSV from csv_path
        - Plot episode rewards over time using matplotlib
        - Plot a rolling average (window=50) to show the trend clearly
        - Save the plot as a PNG to save_path
        - Do NOT use plt.show() — save only, since EC2 has no display
        - Example paths:
            csv_path  = "docs/logs/training_log.csv"
            save_path = "docs/logs/reward_curve.png"
    
    2. plot_epsilon(csv_path, save_path):
        - Same as above but plot epsilon decay over episodes
        - save_path = "docs/logs/epsilon_curve.png"

    Notes:
        - Use os.path.join() for all file paths
        - Use matplotlib.use("Agg") at the top — this makes matplotlib
          work without a display on EC2/Docker
        - Both functions should be callable from main.py after training
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

def plot_rewards(csv_path, save_path):
    """
    Read the training log CSV and plot episode rewards over time with a 50-episode rolling average.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error Could not find {csv_path}")
        return

    episode_col = 'Episode'
    reward_col = 'Total Reward'

    rolling_avg = df[reward_col].rolling(window=50).mean()

    plt.figure(figsize=(10, 6))
    
    plt.plot(df[episode_col], df[reward_col], label='Episode Reward', alpha=0.3, color='blue')
    
    plt.plot(df[episode_col], rolling_avg, label='Rolling Average (50)', color='red', linewidth=2)
    
    plt.title('DQN Lunar Lander Training Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()

def plot_epsilon(csv_path, save_path):
    """
    Read the training log CSV and plot epsilon decay over episodes.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return

    episode_col = 'Episode'
    epsilon_col = 'Epsilon'

    plt.figure(figsize=(10, 6))
    plt.plot(df[episode_col], df[epsilon_col], label='Epsilon Decay', color='green', linewidth=2)
    
    plt.title('DQN Lunar Lander: Epsilon Decay Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()