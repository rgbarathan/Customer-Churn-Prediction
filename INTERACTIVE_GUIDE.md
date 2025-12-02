# 🎯 Interactive Q&A System - User Guide

## Overview

The Customer Churn Prediction and QA System now includes **interactive conversation mode** that allows customer service representatives to have real-time dialogs with high-risk customers.

---

## 🚀 Running Modes

### Mode 1: Standard Mode (Default)
```bash
python main.py
```
**What it does**:
- Trains the churn model
- Loads SQuAD datasets
- Predicts churn for all 5 customers
- For critical-risk customers (>70%), shows automated QA responses
- For high-risk customers (50-70%), shows automated billing recommendations

**When to use**: Daily batch processing, automated analysis, scheduled reports

---

### Mode 2: Demo Mode (Recommended for Testing)
```bash
python main.py --demo
```
**What it does**:
- Runs the full pipeline
- For critical-risk customers, shows a **pre-scripted interactive conversation**
- Demonstrates 3 sample customer service questions
- Shows AI responses with confidence scores
- Displays session summary statistics

**When to use**: Training, demonstrations, testing new features

**Example Output**:
```
Customer 2: 79.63% Churn Risk
Status: 🔴 CRITICAL - This customer needs immediate attention!

📋 DEMO MODE: Showing sample interactive conversation

🎤 CSR: How can I get a discount on my internet bill?
🤖 AI: MyComcast account
   Confidence: 62.50%

🎤 CSR: What bundle packages do you offer?
🤖 AI: loyalty offers
   Confidence: 51.26%

📋 Session Summary:
   Questions asked: 3
   Average confidence: 39.48%
```

---

### Mode 3: Interactive Mode (For Real Conversations)
```bash
python main.py --interactive
```
**What it does**:
- Runs the full pipeline
- For critical-risk customers, opens an **interactive Q&A session**
- Customer service reps can type unlimited questions
- AI provides real-time answers with confidence scores
- Type `quit`, `exit`, or `q` to end the session

**When to use**: Live customer support, real conversations, manual interventions

**Example Session**:
```
🎤 Your question: How can I reduce my internet bill?
🤔 Searching knowledge base (category: billing)...
✅ Answer: Consider bundling services or checking for senior discounts
📊 Confidence: 62.50%

🎤 Your question: What plans do you have?
🤔 Searching knowledge base (category: services)...
✅ Answer: Internet, TV, Phone, and security packages available
📊 Confidence: 58.30%

🎤 Your question: quit
✅ Session ended
```

---

## 💡 How the Interactive System Works

### Step 1: Predict Churn
```
Customer Data → Neural Network → Churn Probability
```

### Step 2: Identify Risk Level
```
If probability > 70%:  🔴 CRITICAL RISK → Interactive Engagement
If probability 50-70%: ⚠️  HIGH RISK → Standard Recommendations
If probability < 50%:  ✅ LOW RISK → No Action
```

### Step 3: Engage with Q&A
For critical customers, the system offers interactive mode:

```
User: "How can I reduce my bill?"
     ↓
System: Detect category (billing/services/support/retention)
     ↓
Knowledge Source: 
  1. Try Comcast KB (curated contexts)
  2. Try SQuAD (39,274 contexts)
  3. Use fallback response
     ↓
Return: Answer + Confidence Score
```

---

## 🎓 Using the Interactive Mode

### For Customer Service Representatives

**Starting a Session**:
```bash
python main.py --interactive
```

**During the Conversation**:
1. Read the customer's churn risk percentage
2. Ask clarifying questions naturally
3. Read the AI's suggested answers
4. Adapt and personalize the response
5. Use confidence scores to guide your approach

**Example Dialog**:
```
CSR: "Hello! I noticed you might be considering leaving us. 
      How can we help you today?"

Customer: "Your prices are too high."

CSR Question to System: "How can we reduce customer costs?"

System Answer: "Bundle discounts available for TV + Internet + Phone"
Confidence: 65.80%

CSR Response: "I completely understand. Great news! We have bundle 
packages that save customers up to 40%. Can I tell you about them?"
```

**Ending a Session**:
- Type: `quit`, `exit`, `q`, or `end`
- Press: `Ctrl+C`

---

## 📊 Understanding Confidence Scores

The system returns confidence scores (0-100%) for each answer:

| Confidence | Meaning | Action |
|-----------|---------|--------|
| **> 80%** | Very confident answer | Use as-is for CSR |
| **60-80%** | Good confidence | Reasonable to use |
| **40-60%** | Moderate confidence | Verify with supervisor |
| **< 40%** | Low confidence | Get human guidance |

### Example:
```
Question: "What are your internet speeds?"
Answer: "100 Mbps to 1200 Mbps available"
Confidence: 85.40%
👍 Use this answer directly

Question: "Do you offer international calling?"
Answer: "Check your account settings"
Confidence: 28.50%
⚠️  Verify this before telling customer
```

---

## 🎯 Best Practices

### 1. **Understand the Customer Context**
```
Before asking AI:
✓ Know their tenure (months with us)
✓ Know their churn risk level
✓ Know what services they use
✓ Know their monthly charges
```

### 2. **Ask Natural Questions**
```
Good: "How can I save money?"
Good: "What internet speeds do you offer?"
Good: "Are there discounts for loyal customers?"

Avoid: "Query: bill reduction" ❌
Avoid: Technical jargon the AI won't understand ❌
```

### 3. **Use Category Hints**
The system auto-detects categories:
- **Billing**: Price, cost, discount, bill, reduce
- **Services**: Internet, TV, phone, speed, plans
- **Retention**: Loyalty, keep, stay, loyalty program
- **Support**: Help, contact, support, issue, problem

### 4. **Handle Low Confidence**
```
When confidence < 50%:
1. Note the low confidence indicator ⚠️
2. Ask your supervisor
3. Don't make promises based on AI answer
4. Log the question for future training
```

### 5. **Track Session Quality**
```
Good session:
✓ Multiple questions answered (3+)
✓ Average confidence > 55%
✓ Customer engaged throughout
✓ Clear resolution or next steps

Poor session:
✗ Few questions (< 2)
✗ Average confidence < 40%
✗ Many low-confidence answers
✗ Customer still uncertain
```

---

## 🔧 Advanced Features

### Category-Based Routing
The system automatically routes questions to relevant knowledge:

```python
if "discount" in question:
    category = "billing"  # Use billing knowledge base
elif "loyalty" in question:
    category = "retention"  # Use retention offers
elif "internet" in question:
    category = "services"  # Use service details
```

### Session History
The system tracks your conversation:
```
Session Summary:
✓ Total questions: 5
✓ Average confidence: 62.40%
✓ Categories: billing (2), retention (2), support (1)
```

### Fallback Mechanism
If the AI can't find a good answer:
```
Level 1: Comcast KB (13 curated contexts)
  └─ Not found? 
Level 2: SQuAD Dataset (39,274 contexts)
  └─ Not found?
Level 3: Fallback response
  └─ "Please contact Comcast support at 1-855-COMCAST"
```

---

## 💻 Command Reference

| Command | Mode | Use Case |
|---------|------|----------|
| `python main.py` | Standard | Batch processing |
| `python main.py --demo` | Demo | Training/testing |
| `python main.py --interactive` | Interactive | Live support |

---

## 📋 Sample Interactive Sessions

### Session 1: Billing Questions
```
CSR: How can we reduce this customer's bill?
System: Consider bundling services
Confidence: 62.50%

CSR: What bundle packages are available?
System: TV + Internet + Phone bundles with discounts
Confidence: 51.26%

CSR: Do we have discounts for senior citizens?
System: Contact our retention team for personalized offers
Confidence: 4.70%

Session Summary: 3 questions, 39.48% avg confidence
```

### Session 2: Service Questions
```
CSR: What internet speeds do we offer?
System: 100 Mbps to 1200 Mbps depending on plan
Confidence: 78.90%

CSR: Can we upgrade their speed?
System: Yes, upgrades available in most areas
Confidence: 71.20%

CSR: Is there an installation fee?
System: Check with technical team for location-specific details
Confidence: 35.60%

Session Summary: 3 questions, 61.90% avg confidence
```

---

## ⚠️ Troubleshooting

### "EOF when reading a line"
**Cause**: Running in non-interactive mode (piped input)
**Solution**: Run `python main.py --demo` for demo mode

### Low Confidence Answers
**Cause**: Question not well-matched in knowledge base
**Solution**: Rephrase question or contact supervisor

### Model Not Loading
**Cause**: Missing `models/churn_model.pth`
**Solution**: Run `python churn_prediction.py` to train model

### SQuAD Data Not Loading
**Cause**: Missing `archive/*.json` files
**Solution**: Ensure archive folder contains both train and dev files

---

## 📊 Metrics to Track

### Per-Session Metrics
- Total questions asked
- Average confidence score
- Categories used
- Time to resolution

### Performance Metrics
- % of critical customers engaged interactively
- Average customer satisfaction (if tracked)
- Conversion to retained customer
- Cost savings vs. customer acquisition cost

---

## 🎓 Training Your CSRs

### 1. Demo Session
```bash
python main.py --demo
# Show how the system responds
```

### 2. Practice Session
```bash
python main.py --interactive
# Let CSRs practice asking natural questions
```

### 3. Real Conversations
```bash
python main.py --interactive
# Use for actual customer retention calls
```

---

## 📞 Support

**For Technical Issues**:
- Check error messages
- Review confidence scores
- Consult supervisor

**For Customer Service Issues**:
- Use `--demo` mode for examples
- Reference customer profile data
- Track session history

---

## ✨ Summary

The interactive Q&A system empowers customer service teams to:

✅ **Quickly identify** at-risk customers  
✅ **Ask natural questions** to the AI  
✅ **Get instant answers** with confidence scores  
✅ **Personalize responses** to each customer  
✅ **Track conversations** for quality assurance  
✅ **Scale support** without hiring more reps  

**Start now**: `python main.py --demo`

---

*Last Updated: December 1, 2025*
*Version: 2.0 (Interactive)*
