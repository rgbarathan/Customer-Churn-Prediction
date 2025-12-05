# 🤖 Reinforcement Learning Recommendation System

## Overview

This system uses **Deep Q-Network (DQN)**, a state-of-the-art Reinforcement Learning algorithm, to learn optimal customer retention strategies. Unlike rule-based systems, the RL agent learns from experience and continuously improves its recommendations.

## How It Works

### 1. **Deep Q-Network (DQN) Architecture**
```
Customer State (8 features) 
    ↓
Input Layer
    ↓
Hidden Layer 1 (128 neurons) + ReLU + Dropout
    ↓
Hidden Layer 2 (128 neurons) + ReLU + Dropout
    ↓
Hidden Layer 3 (64 neurons) + ReLU
    ↓
Output Layer (8 actions) → Q-values for each action
```

### 2. **Action Space** (8 Retention Strategies)
- **Action 0:** No Action (Monitor only)
- **Action 1:** Small Discount (10% for 3 months) - $50 cost
- **Action 2:** Medium Discount (20% for 6 months) - $100 cost
- **Action 3:** Large Discount (30% for 12 months) - $200 cost
- **Action 4:** Service Bundle Upgrade - $75 cost
- **Action 5:** Contract Conversion Offer - $150 cost
- **Action 6:** Premium Retention Package - $300 cost
- **Action 7:** Loyalty Rewards Program - $100 cost

### 3. **State Representation** (Customer Features)
- Tenure (months)
- Monthly Charges
- Contract Type
- Service Count
- Senior Citizen Status
- Has Add-ons
- Churn Probability (from prediction model)
- Total Charges

### 4. **Reward Function**
```
If Customer Retained:
    Reward = Customer LTV - Action Cost

If Customer Churned:
    Reward = -(Customer LTV + Action Cost)
```

This encourages the agent to:
- Maximize customer retention
- Minimize unnecessary costs
- Balance retention probability vs. cost

### 5. **Learning Process**

The agent learns through:
- **Experience Replay:** Stores past experiences and learns from them
- **Epsilon-Greedy Exploration:** Balances trying new strategies vs. using known good ones
- **Target Network:** Stabilizes learning by using a slowly-updating copy
- **Temporal Difference Learning:** Learns from prediction errors

## Advantages Over Rule-Based System

| Feature | Rule-Based | RL-Based |
|---------|-----------|----------|
| **Adaptability** | Fixed rules | Learns and adapts |
| **Personalization** | Template-based | Customer-specific |
| **Cost Optimization** | Manual tuning | Automatic optimization |
| **Performance** | Static | Improves over time |
| **Action Selection** | If-then logic | Learned optimal policy |
| **Handling Complexity** | Limited | Handles non-linear patterns |

## Training the RL Agent

### Option 1: From Main Menu
```bash
python main.py --menu
# Select option 6: Train RL Recommendation System
```

### Option 2: Standalone Training
```bash
python rl_recommendation_system.py
```

### Training Parameters
- **Episodes:** 1000 (default) - Number of training iterations
- **Batch Size:** 64 - Number of experiences per training step
- **Learning Rate:** 0.001 - How fast the agent learns
- **Gamma (γ):** 0.99 - Discount factor for future rewards
- **Epsilon Decay:** 0.995 - Exploration rate decay

### Training Output
```
Episode 100/1000
  Avg Reward: 1.45
  Avg Retention Rate: 72.50%
  Avg Loss: 0.0234
  Epsilon: 0.605

Episode 200/1000
  Avg Reward: 2.13
  Avg Retention Rate: 78.20%
  Avg Loss: 0.0189
  Epsilon: 0.366
  
...

✅ Training Complete!
Final Retention Rate: 85.30%
Final Average Reward: 3.87
```

## Using RL Recommendations

Once trained, the system automatically uses RL recommendations:

```python
# Customer analysis with RL recommendations
customer_id = "7590-VHVEG"
# The system will automatically use RL agent if available
# Output will show: 🤖 Using Reinforcement Learning (AI-Optimized)
```

### Example RL Recommendation Output
```
💡 RECOMMENDED ACTIONS (3):
   🤖 Using Reinforcement Learning (AI-Optimized)
   
   1. [Priority 1] 🤖 Large Discount (30%)
      30% monthly discount for 12 months
      Cost: $200
      Impact: 87% retention probability (RL-predicted)
   
   2. [Priority 2] 🤖 Contract Conversion Offer
      2-year contract with price lock + $25 off
      Cost: $150
      Impact: 82% retention probability (RL-predicted)
   
   3. [Priority 3] 🤖 Service Bundle Upgrade
      Free premium services for 6 months
      Cost: $75
      Impact: 76% retention probability (RL-predicted)
```

## Performance Metrics

### Expected Improvements
- **Retention Rate:** 15-25% improvement over rule-based
- **Cost Efficiency:** 20-30% reduction in unnecessary spending
- **ROI:** 150-200% (vs. 100-120% for rule-based)
- **Personalization:** Customer-specific vs. template-based

### Monitoring
The system tracks:
- Q-values for each action
- Epsilon (exploration rate)
- Average reward per episode
- Retention success rate
- Training loss

## Production Deployment Considerations

### 1. **Data Requirements**
- Historical retention campaign data
- Customer interaction logs
- Actual retention outcomes
- Cost data for each action

### 2. **Online Learning**
- Update agent with real customer responses
- Periodic retraining (monthly/quarterly)
- A/B testing framework

### 3. **Safety Measures**
- Minimum/maximum discount limits
- Budget constraints
- Approval workflows for high-cost actions
- Fallback to rule-based if RL fails

### 4. **Monitoring**
- Track retention rates by action type
- Monitor cost per retention
- Customer satisfaction scores
- Agent performance metrics

## Technical Details

### DQN Algorithm
```
For each episode:
    For each customer:
        1. Observe current state (customer features)
        2. Select action using ε-greedy policy
        3. Execute action, observe reward and next state
        4. Store experience in replay buffer
        5. Sample random batch from buffer
        6. Update Q-network using:
           Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        7. Periodically update target network
```

### Key Hyperparameters
- **State Dimension:** 8
- **Action Dimension:** 8
- **Hidden Layers:** [128, 128, 64]
- **Replay Buffer Size:** 10,000
- **Target Update Frequency:** 100 steps
- **Optimizer:** Adam

## Future Enhancements

### Phase 1 (Current)
- ✅ Basic DQN implementation
- ✅ Simulated environment
- ✅ Integration with churn prediction

### Phase 2 (Planned)
- [ ] Double DQN (improved stability)
- [ ] Dueling DQN (better Q-value estimation)
- [ ] Prioritized Experience Replay
- [ ] Multi-step returns

### Phase 3 (Advanced)
- [ ] Real customer interaction data
- [ ] A3C (Actor-Critic) for continuous actions
- [ ] PPO (Proximal Policy Optimization)
- [ ] Meta-learning across customer segments

## References

- **DQN Paper:** Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- **Double DQN:** van Hasselt et al. (2016)
- **Dueling DQN:** Wang et al. (2016)
- **Application:** Customer retention optimization using RL

## Support

For questions or issues:
1. Check the main README.md
2. Review training logs in console output
3. Verify model file exists: `models/rl_agent.pth`
4. Ensure all dependencies are installed

---

**Note:** The current implementation uses simulated customer responses for training. In production, replace the simulation with actual historical retention campaign data for optimal results.
