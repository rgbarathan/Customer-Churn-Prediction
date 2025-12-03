#!/usr/bin/env python3
"""Quick test to verify demo mode works"""

import sys
import torch
import pandas as pd
from churn_prediction import ChurnModel
import pickle
import os

# Load model and scaler (just like main.py does at module level)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

try:
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    input_dim = len(feature_names)
except FileNotFoundError:
    input_dim = 19

model = ChurnModel(input_dim=input_dim)
model.load_state_dict(torch.load('models/churn_model.pth'))
model.eval()

print("✓ Model loaded successfully")

# Now import and test run_demo
from main import run_demo

print("\n" + "="*70)
print("TESTING DEMO MODE (Option 3)")
print("="*70)
print("\nCalling run_demo() function...\n")

# Test the function
try:
    run_demo()
    print("\n✅ SUCCESS: Demo mode is now working!")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

