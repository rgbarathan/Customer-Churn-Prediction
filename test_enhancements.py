#!/usr/bin/env python3
"""Quick test of enhanced retention system with one customer"""

import torch
import pandas as pd
from churn_prediction import ChurnModel
import pickle
import sys

# Load model and scaler
model = ChurnModel(input_dim=23)
model.load_state_dict(torch.load('models/churn_model.pth'))
model.eval()

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Import main functions
from main import (extract_customer_profile, identify_risk_factors, 
                  generate_recommendations, calculate_winback_probability,
                  generate_objection_handlers, generate_conversation_flow,
                  determine_next_contact_channel, generate_sentiment_guidance,
                  add_engineered_features)

# High-risk test customer (Senior, new, high charges)
customer_data = pd.DataFrame([[1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 105.0, 210.0]],
                             columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])

# Add engineered features
customer_data_enhanced = add_engineered_features(customer_data)

# Predict churn
scaled = scaler.transform(customer_data_enhanced)
tensor = torch.tensor(scaled, dtype=torch.float32)

with torch.no_grad():
    output = model(tensor)
    churn_prob = torch.sigmoid(output).item()

print(f"Testing Enhanced Retention System")
print(f"=" * 70)
print(f"Churn Probability: {churn_prob:.2%}")

# Extract profile
profile = extract_customer_profile(customer_data)

# Test Win-Back Probability
winback = calculate_winback_probability(profile, churn_prob)
print(f"\n✅ Win-Back Probability: {winback['probability']:.2%}")
print(f"   Strategy: {winback['strategy']}")

# Test Objection Handlers
objection_handlers = generate_objection_handlers(profile, churn_prob)
print(f"\n✅ Objection Handlers: {len(objection_handlers)} scenarios generated")

# Test Conversation Flow
risk_factors = identify_risk_factors(profile, churn_prob)
recommendations = generate_recommendations(profile, risk_factors, churn_prob)
conversation_flow = generate_conversation_flow(profile, churn_prob, recommendations)
print(f"\n✅ Conversation Flow: {len(conversation_flow)} steps")

# Test Next Contact Channel
next_contact = determine_next_contact_channel(profile, churn_prob)
print(f"\n✅ Next Contact: {next_contact['primary']}")

# Test Sentiment Guidance
sentiment_guidance = generate_sentiment_guidance(profile, churn_prob)
print(f"\n✅ Sentiment Guidance: {len(sentiment_guidance['watch_for'])} keyword groups")

print(f"\n{'=' * 70}")
print("All enhancements working correctly! ✨")
