#!/bin/bash
# Test script to run demo mode
cd "/Users/rbarat738@cable.comcast.com/Documents/Drexel/Books and Assignments/Assignments/Assignment 5/Project Customer Churn Prediction and QA"
echo "n" | /usr/bin/python3 main.py --demo 2>&1 | head -150
