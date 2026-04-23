import gymnasium as gym
from dqn_lunar_lander import DQNAgent

import time
import csv
import os

def train():
    start_time = time.time()
    max_hours = 2
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Write to file for logging rewards
    # makes checking the training progress easier and allows for plotting reward trends later
    os.makedirs("docs/logs", exist_ok=True)
    fieldnames = ['Episode', 'Total Reward', 'Epsilon']
    with open (csv_path := "docs/logs/training_log.csv", mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    num_episodes = 2000
    for episode in range(num_episodes):
        # check time to avoid expensive AWS fees
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= max_hours:
            print(f"Time limit reached at episode {episode + 1}. Saving and stopping.")
            agent.save("docs/checkpoints/dqn_lunar_lander_timeout.pth")
            break
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            agent.store(state, action, reward, next_state, done)

            # Update agent (e.g., sample from replay buffer and train)
            agent.learn()

            state = next_state

        with open(csv_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Episode': episode + 1, 'Total Reward': total_reward, 'Epsilon': agent.epsilon})
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.4f}")    
        if (episode + 1) % 100 == 0:
            os.makedirs("docs/checkpoints", exist_ok=True)
            agent.save(f"docs/checkpoints/dqn_lunar_lander_episode_{episode + 1}.pth")  
            print(f"Saved checkpoint at episode {episode + 1}")
    
    env.close()
    agent.save("docs/checkpoints/dqn_lunar_lander_final.pth")
    print("Training complete. Final model saved.")
    elapsed = (time.time() - start_time) / 60
    print(f"Total training time: {elapsed:.2f} minutes")

if __name__ == "__main__":
    train()