from gymnasium.wrappers import TimeLimit

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        # Increased depth of the network with more layers
        self.fc1 = nn.Linear(input_dim, 256)  # First layer with more neurons
        self.bn1 = nn.BatchNorm1d(256)        # Batch normalization
        self.dropout1 = nn.Dropout(0.3)      # Dropout layer

        self.fc2 = nn.Linear(256, 512)  # First layer with more neurons
        self.bn2 = nn.BatchNorm1d(512)        # Batch normalization
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 1000)       # Second hidden layer with more neurons
        self.bn3 = nn.BatchNorm1d(1000)       # Batch normalization
        self.dropout3 = nn.Dropout(0.3)      # Dropout layer
        
        self.fc4 = nn.Linear(1000, 512)       # Third hidden layer
        self.bn4 = nn.BatchNorm1d(512)       # Batch normalization
        self.dropout4 = nn.Dropout(0.3)      # Dropout layer
        
        self.fc5 = nn.Linear(512, 256)       # Fourth hidden layer
        self.bn5 = nn.BatchNorm1d(256)       # Batch normalization
        self.dropout5 = nn.Dropout(0.3)      # Dropout layer
        
        self.fc6 = nn.Linear(256, output_dim)  # Output layer

    def forward(self, x):
        # Forward pass through the layers
        x = F.relu(self.bn1(self.fc1(x)))  # BatchNorm + ReLU
        x = self.dropout1(x)  # Apply dropout
        
        x = F.relu(self.bn2(self.fc2(x)))  # BatchNorm + ReLU
        x = self.dropout2(x)  # Apply dropout
        
        x = F.relu(self.bn3(self.fc3(x)))  # BatchNorm + ReLU
        x = self.dropout3(x)  # Apply dropout
        
        x = F.relu(self.bn4(self.fc4(x)))  # BatchNorm + ReLU
        x = self.dropout4(x)  # Apply dropout

        x = F.relu(self.bn5(self.fc5(x)))  # BatchNorm + ReLU
        x = self.dropout5(x)  # Apply dropout
        
        return self.fc6(x)  # Final output layer

# Implementing the ProjectAgent class
class ProjectAgent:
    def __init__(self, state_dim=6, action_dim=4, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=10000):
        self.state_dim = state_dim + 2  # Add 2 for the drug usage counters
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Counters for drug usage
        self.drug_counters = [0, 0]  # [Drug 1 count, Drug 2 count]

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # DQN model and target model
        self.q_network = DQN(self.state_dim, action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0  # Initial epsilon for exploration
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.iteration = 0

    def act(self, observation, use_random=False):
        # Concatenate state with drug counters
        if self.iteration >= 200:
            self.drug_counters = [0, 0]
            self.iteration = 0

        extended_state = np.concatenate([observation, self.drug_counters])
        
        if use_random and np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        observation = torch.FloatTensor(extended_state).unsqueeze(0).to(self.device)
        
        # Switch to evaluation mode for batch normalization
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(observation)
        self.q_network.train()  # Switch back to training mode
        action = torch.argmax(q_values).item()
        self.update_drug_counters(action)
        self.iteration += 1
        return action

    def remember(self, state, action, reward, next_state, done):
        # Concatenate states with drug counters
        extended_state = np.concatenate([state, self.drug_counters])
        
        # Update counters for the next state
        self.update_drug_counters(action)
        extended_next_state = np.concatenate([next_state, self.drug_counters])

        self.replay_buffer.append((extended_state, action, reward, extended_next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_drug_counters(self, action):
        # Update counters based on the action (assuming actions 0 and 1 correspond to the two drugs)
        if action == 0:  # Drug 1
            self.drug_counters[0] += 1
        elif action == 1:  # Drug 2
            self.drug_counters[1] += 1

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self):
        self.q_network.load_state_dict(torch.load("/home/runner/work/assignment-YassKa71/assignment-YassKa71/src/trained_dqn_model.pth"))
        self.q_network.eval()

# Training loop
def train_agent(env, agent, episodes=500, target_update_freq=10):
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Reset drug counters at the start of each episode
        agent.drug_counters = [0, 0]

        for _ in range(200):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            # Store transition in the replay buffer
            agent.remember(state, action, reward, next_state, done)
            for _ in range(10):
                agent.replay()

            state = next_state
            total_reward += reward

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    agent.save("./trained_dqn_model.pth")
    print("Training complete. Model saved.")

