"""
Reinforcement Learning-based Customer Retention Recommendation System
Using Deep Q-Network (DQN) to learn optimal retention strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
import pickle
import os

# Define retention action space
RETENTION_ACTIONS = {
    0: {
        'name': 'No Action',
        'cost': 0,
        'description': 'Monitor customer, no intervention'
    },
    1: {
        'name': 'Small Discount (10%)',
        'cost': 50,
        'description': '10% monthly discount for 3 months'
    },
    2: {
        'name': 'Medium Discount (20%)',
        'cost': 100,
        'description': '20% monthly discount for 6 months'
    },
    3: {
        'name': 'Large Discount (30%)',
        'cost': 200,
        'description': '30% monthly discount for 12 months'
    },
    4: {
        'name': 'Service Bundle Upgrade',
        'cost': 75,
        'description': 'Free premium services for 6 months'
    },
    5: {
        'name': 'Contract Conversion Offer',
        'cost': 150,
        'description': '2-year contract with price lock + $25 off'
    },
    6: {
        'name': 'Premium Retention Package',
        'cost': 300,
        'description': 'Maximum discount + free upgrades + loyalty rewards'
    },
    7: {
        'name': 'Loyalty Rewards Program',
        'cost': 100,
        'description': 'Points, equipment upgrade, exclusive perks'
    }
}

class DQNNetwork(nn.Module):
    """Deep Q-Network for learning optimal retention actions"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class RetentionEnvironment:
    """
    Simulates customer response to retention actions
    This would ideally use real historical data
    """
    def __init__(self):
        self.action_space = len(RETENTION_ACTIONS)
        
    def simulate_customer_response(self, customer_state, action_id, churn_prob):
        """
        Simulate whether customer accepts offer and stays
        Based on churn probability and action effectiveness
        
        Returns:
            retained: bool - Whether customer was retained
            reward: float - Reward for the action
        """
        action = RETENTION_ACTIONS[action_id]
        action_cost = action['cost']
        
        # Calculate retention probability based on action strength
        # Stronger actions (higher cost) have better retention rates
        base_retention_prob = 1.0 - churn_prob
        
        # Action effectiveness multipliers (learned from domain knowledge)
        action_effectiveness = {
            0: 1.0,   # No action - no improvement
            1: 1.15,  # Small discount - 15% improvement
            2: 1.35,  # Medium discount - 35% improvement
            3: 1.55,  # Large discount - 55% improvement
            4: 1.25,  # Service bundle - 25% improvement
            5: 1.40,  # Contract conversion - 40% improvement
            6: 1.70,  # Premium package - 70% improvement
            7: 1.30   # Loyalty rewards - 30% improvement
        }
        
        # Adjust retention probability based on action
        enhanced_retention_prob = min(0.95, base_retention_prob * action_effectiveness[action_id])
        
        # Additional factors based on customer profile
        tenure = customer_state[0] if len(customer_state) > 0 else 10
        monthly_charges = customer_state[1] if len(customer_state) > 1 else 70
        contract_type = customer_state[2] if len(customer_state) > 2 else 0
        
        # Tenure bonus (longer tenure = easier to retain)
        if tenure > 24:
            enhanced_retention_prob *= 1.2
        elif tenure < 6:
            enhanced_retention_prob *= 0.9
        
        # Contract type consideration
        if contract_type == 2:  # Two year contract
            enhanced_retention_prob *= 1.15
        elif contract_type == 0:  # Month-to-month
            enhanced_retention_prob *= 0.95
        
        # Determine if customer is retained
        retained = random.random() < enhanced_retention_prob
        
        # Calculate reward
        if retained:
            # Customer retained: positive reward = LTV - action cost
            customer_ltv = monthly_charges * 36  # 3-year LTV
            reward = customer_ltv - action_cost
        else:
            # Customer churned: negative reward = -LTV - action cost (wasted)
            customer_ltv = monthly_charges * 36
            reward = -(customer_ltv + action_cost)
        
        # Normalize reward to reasonable range
        reward = reward / 1000.0  # Scale down for training stability
        
        return retained, reward, enhanced_retention_prob

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network Agent for learning optimal retention strategies"""
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-Network and Target Network
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=20000)
        self.batch_size = 128
        self.update_target_every = 200
        self.steps = 0
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
        print(f"✓ RL Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            print(f"✓ RL Agent loaded from {filepath}")
            return True
        return False

def prepare_customer_state(profile, churn_prob):
    """
    Convert customer profile to state vector for RL agent
    """
    state = [
        profile['tenure_months'] / 100.0,  # Normalize tenure
        profile['monthly_charges'] / 150.0,  # Normalize charges
        profile['contract_type'] / 2.0 if isinstance(profile.get('contract_type'), (int, float)) else 0.5,  # Normalize contract
        profile['service_count'] / 8.0,  # Normalize service count
        1.0 if profile['senior_citizen'] else 0.0,  # Binary
        1.0 if profile['has_addons'] else 0.0,  # Binary
        churn_prob,  # Churn probability from prediction model
        profile['total_charges'] / 10000.0,  # Normalize total charges
    ]
    return np.array(state, dtype=np.float32)

def train_rl_agent(agent, environment, num_episodes=1000, customer_profiles=None):
    """
    Train RL agent on customer retention scenarios
    """
    print("\n" + "="*70)
    print("TRAINING REINFORCEMENT LEARNING AGENT")
    print("="*70)
    
    episode_rewards = []
    retention_rates = []
    losses = []
    
    for episode in range(num_episodes):
        episode_reward = 0
        episode_retentions = 0
        episode_attempts = 0
        action_hist = [0] * agent.action_dim
        
        # Simulate multiple customers in each episode
        num_customers = 20
        
        for _ in range(num_customers):
            # Generate synthetic customer (in practice, use real data)
            churn_prob = random.uniform(0.3, 0.9)  # Focus on at-risk customers
            tenure = random.randint(1, 60)
            monthly_charges = random.uniform(50, 120)
            contract_type = random.randint(0, 2)
            
            profile = {
                'tenure_months': tenure,
                'monthly_charges': monthly_charges,
                'contract_type': contract_type,
                'service_count': random.randint(1, 6),
                'senior_citizen': random.random() > 0.8,
                'has_addons': random.random() > 0.5,
                'total_charges': monthly_charges * tenure
            }
            
            state = prepare_customer_state(profile, churn_prob)
            
            # Agent selects action
            action = agent.select_action(state, training=True)
            action_hist[action] += 1

            # Simulate to get retention probability (deterministic)
            # Ignore sampled reward; compute expected reward uplift vs. no action
            _, _, p_action = environment.simulate_customer_response(state, action, churn_prob)
            _, _, p_noact = environment.simulate_customer_response(state, 0, churn_prob)

            # Reconstruct LTV and action cost from normalized state
            monthly_charges = state[1] * 150.0
            ltv = monthly_charges * 36.0
            from math import isfinite
            cost_action = RETENTION_ACTIONS[action]['cost']
            cost_noact = RETENTION_ACTIONS[0]['cost']

            # Expected reward (scaled by 1/1000 like environment)
            def expected_reward(p, cost):
                return (((2.0 * p - 1.0) * ltv) - cost) / 1000.0

            shaped = expected_reward(p_action, cost_action) - expected_reward(p_noact, cost_noact)
            # Optional clipping for stability
            if shaped > 1.0:
                shaped = 1.0
            elif shaped < -1.0:
                shaped = -1.0

            # For reporting, also sample an outcome once
            retained, _, _ = environment.simulate_customer_response(state, action, churn_prob)
            episode_retentions += (1 if retained else 0)
            episode_attempts += 1
            episode_reward += shaped

            # Terminal step after action
            next_state = state.copy()
            done = True

            # Store shaped reward
            agent.replay_buffer.push(state, action, shaped, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        
        episode_rewards.append(episode_reward)
        retention_rate = episode_retentions / episode_attempts
        retention_rates.append(retention_rate)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_retention = np.mean(retention_rates[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Retention Rate: {avg_retention:.2%}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            # Action distribution snapshot
            total_actions = sum(action_hist) if sum(action_hist) else 1
            dist = [round(c / total_actions, 2) for c in action_hist]
            print(f"  Action Dist (last ep): {dist}")
    
    print("\n✅ Training Complete!")
    print(f"Final Retention Rate: {np.mean(retention_rates[-100:]):.2%}")
    print(f"Final Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
    
    return episode_rewards, retention_rates

def get_rl_recommendations(profile, churn_prob, agent, top_k=3):
    """
    Get top-K retention recommendations from trained RL agent
    """
    state = prepare_customer_state(profile, churn_prob)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    # Get Q-values for all actions
    with torch.no_grad():
        q_values = agent.q_network(state_tensor).squeeze().numpy()
    
    # Get top-K actions
    top_actions = np.argsort(q_values)[-top_k:][::-1]
    
    recommendations = []
    for idx, action_id in enumerate(top_actions, 1):
        action = RETENTION_ACTIONS[action_id]
        q_value = q_values[action_id]
        
        # Estimate expected retention probability (normalized Q-value)
        expected_success = min(95, max(50, 50 + q_value * 10))
        
        recommendations.append({
            'priority': idx,
            'action_id': int(action_id),
            'icon': '🤖',
            'action': action['name'],
            'description': action['description'],
            'cost': action['cost'],
            'expected_impact': f'{expected_success:.0f}% retention probability (RL-predicted)',
            'q_value': float(q_value)
        })
    
    return recommendations

if __name__ == "__main__":
    # Initialize environment and agent
    state_dim = 8  # Customer state dimensions
    action_dim = len(RETENTION_ACTIONS)
    
    environment = RetentionEnvironment()
    agent = DQNAgent(state_dim, action_dim)
    
    # Check if pre-trained model exists
    model_path = 'models/rl_agent.pth'
    if os.path.exists(model_path):
        print("Loading pre-trained RL agent...")
        agent.load(model_path)
    else:
        print("Training new RL agent...")
        episode_rewards, retention_rates = train_rl_agent(agent, environment, num_episodes=1000)
        
        # Save trained agent
        os.makedirs('models', exist_ok=True)
        agent.save(model_path)
    
    print("\n✅ RL Recommendation System Ready!")
