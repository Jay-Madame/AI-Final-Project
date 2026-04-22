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