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
import argparse
import os
import gymnasium as gym

from dqn_lunar_lander import DQNAgent
from training import train


def main():
    parser = argparse.ArgumentParser(description="Main entry point for DQN LunarLander")

    parser.add_argument("--train", action="store_true", help="runs the training loop")
    parser.add_argument("--eval", action="store_true", help="runs evaluation on a saved model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("docs", "checkpoints", "dqn_lunar_lander_final.pth"),
        help="path to a saved model for evaluation"
    )

    args = parser.parse_args()

    if args.train:
        train()

    elif args.eval:
        render_mode = None if not os.environ.get("DISPLAY") else "human"
        env = gym.make("LunarLander-v3", render_mode="human")
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(state_size, action_size)
        agent.load(args.checkpoint)
        agent.epsilon = 0.0

        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Evaluation reward: {total_reward:.2f}")
        env.close()

    else:
        print("Please use --train or --eval")


if __name__ == "__main__":
    main()