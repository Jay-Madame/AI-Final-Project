"""
    [] Main entry point for DQN LunarLander
    
    Your job is to implement the argument parsing and call the correct functions.
    
    Requirements:
        - Import DQNAgent from dqn_lunar_lander.py
        - Import train() from training.py
        - Add the following arguments:
            - --train: runs the training loop
            - --eval: runs evaluation on a saved model
            - --checkpoint: path to a saved model for evaluation
              (default: "docs/checkpoints/dqn_lunar_lander_final.pth")
        
    Notes:
        - For --eval, load the model using agent.load(path) before evaluating
        - Make sure render_mode="human" is only used for --eval, not --train
          since EC2 has no display
        - Use os.path.join() for all file paths to ensure compatibility
          between macOS and Linux/Docker on EC2
          
    Example usage:
        python main.py --train
        python main.py --eval --checkpoint docs/checkpoints/dqn_ep100.pth
"""