"""
    [x] DQN for LunarLander
        - 8 layer input:
            - x position
            - y position
            - x velocity
            - y velocity
            - angle
            - angular velocity
            - left leg contact
            - right leg contact
        - 4 layer output:
            - do nothing
            - fire main engine
            - fire left orientation engine
            - fire right orientation engine
    [x] Replay Buffer
    based from https://medium.com/@coldstart_coder/dqn-algorithm-training-an-ai-to-land-on-the-moon-1a1307748ed9
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state).argmax().item()
        
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int):
        # deque automatically evicts oldest experiences when full
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        # random sampling breaks correlation between consecutive experiences
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
    
class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1e-5,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 500,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # policy_net is trained every step; target_net is a frozen copy
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state: np.ndarray) -> int:
        # explore randomly with probability epsilon, otherwise exploit
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.policy_net.get_action(state_tensor)
    
    def store(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(Experience(state, action, reward, next_state, done))

    def learn(self) -> float | None:
        # wait until buffer has enough experiences to fill a batch
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        experiences = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor([e.state for e in experiences], dtype=torch.float32).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor([e.next_state for e in experiences], dtype=torch.float32).to(self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            # Bellman target: r + γ * max Q(s') — zeroed out on terminal states
            # (1 - dones) ensures no future reward is added when episode ended
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients to prevent instability from large early updates
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # sync target network periodically, not every step to stabilize learning            if self.learn_step % self.target_update_freq == 0:
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_step += 1
        # decay epsilon so the agent explores less as it learns more
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        return loss.item()
    
    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())