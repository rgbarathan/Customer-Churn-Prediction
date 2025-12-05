# 🎯 Customer Churn Prediction & Retention System

A production-ready machine learning system that **predicts customer churn** and **provides AI-powered retention strategies** using advanced neural networks, reinforcement learning, and intelligent recommendation systems with comprehensive evaluation metrics.

## 📋 Project Overview

**Objective**: Help telecommunications companies identify at-risk customers and engage them with personalized, data-driven retention strategies.

**Key Achievements**:
- 🧠 **Enhanced Deep Learning Model**: 85.19% ROC-AUC, 72.7% recall, 57.2% precision
- 🎯 **36 Engineered Features**: Advanced feature engineering with ratios, interactions, and risk indicators
- 🤖 **RL-Based Recommendations**: DQN agent trained for optimal retention strategies (50.8% success rate)
- ⚖️ **Optimized Decision Threshold**: 0.48 for balanced precision-recall (64.0% F1-Score)
- 📊 **Multi-Tier Risk Segmentation**: Critical (70%+), High (60%+), Medium (50%+), Custom thresholds
- 💪 **Conversion Tracking**: 42% retention rate, 297.6% ROI, $290 average cost per retention
- 🎯 **Relevance Scoring**: Measures how well recommendations match specific customer risk factors
- 📈 **Comprehensive Evaluation**: Model metrics, recommendation quality, conversion analysis
- 🗣️ **Agent Support**: Conversation playbooks, objection handlers, channel optimization
- 💼 **Business Impact**: $1.19M+ potential annual revenue from 2,374 identified high-risk customers

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **RAM**: 4GB recommended
- **Disk Space**: 500MB for dependencies + models

### Installation

```bash
# Clone repository
git clone https://github.com/rgbarathan/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import pandas; import sklearn; print('✅ All packages installed!')"
```

### Running the Application

```bash
# Start interactive menu
python main.py --menu
```

---

## 📋 Main Menu Options

```
CUSTOMER CHURN PREDICTION & RETENTION SYSTEM
═══════════════════════════════════════════════

📋 MAIN MENU:
   1. 📊 Churn Prediction Model - Evaluation Metrics
   2. 🎯 Evaluate Recommendation System Quality
   3. 💪 Enhanced Metrics (Conversion + Relevance) ⭐ NEW
   4. 🔍 Analyze Single Customer (by ID)
   5. 📈 Generate High-Risk Customer Report
   6. 🎬 Run Demo (3 Test Customers)
   7. 🤖 Train RL Recommendation System (Advanced)
   8. 🚪 Exit
```

---

## 🎓 Core Features

### 1. Model Evaluation (Option 1)

**What it does**: Displays comprehensive model performance metrics

**Key Metrics**:
- **Accuracy**: 78.32%
- **Precision**: 57.20% (optimized threshold 0.48)
- **Recall**: 72.66% (catches 73% of churners)
- **F1-Score**: 64.01% (balanced performance)
- **ROC-AUC**: 85.19% (excellent discrimination)

**Business Impact**:
- 2,374 customers flagged as high-risk at 48% threshold
- $3.17M potential savings from correct predictions
- $1.19M revenue at risk from missed churners

**Usage**:
```bash
python main.py --menu
# Select: 1
```

---

### 2. Recommendation Quality (Option 2)

**What it does**: Evaluates the recommendation generation system

**Key Metrics**:
- **Coverage**: 100% (all customers get recommendations)
- **Diversity**: 84.9% (varied recommendation types)
- **Success Rate**: 50.8% (based on RL agent simulation)
- **ROI**: 492% (highly profitable retention efforts)

**Recommendation Types**:
- Contract conversion offers
- Price discounts (15-40%)
- Service bundles
- Loyalty rewards
- Onboarding support

**Usage**:
```bash
python main.py --menu
# Select: 2
```

---

### 3. Enhanced Metrics - Conversion & Relevance ⭐ NEW (Option 3)

**What it does**: Tracks actual retention outcomes and measures recommendation relevance to customer risks

#### Conversion Tracking

**Purpose**: Measure if customers are actually retained after recommendations

**Current Performance**:
- **Conversion Rate**: 42.0% (target: ≥50%)
- **Customers Retained**: 42/100 in test sample
- **ROI**: 297.6% (highly profitable)
- **Cost per Retention**: $290
- **Net Benefit**: $87,187 from 100 customers
- **Prediction Accuracy**: 51.18%

#### Relevance Scoring

**Purpose**: Measure how well recommendations match specific risk factors

**Current Performance**:
- **Average Relevance**: 43.5% (target: ≥60%)
- **High Relevance (≥70%)**: 0/100 customers
- **Medium Relevance (40-70%)**: 63/100 customers
- **Low Relevance (<40%)**: 37/100 customers

**Risk-to-Recommendation Mapping**:
- "High Monthly Charges" → discount, price, savings offers
- "No Contract" → contract commitment incentives
- "Senior Citizen" → senior-specific discounts
- "Short Tenure" → onboarding support, welcome packages
- "No Services" → bundle offers, add-ons

**Sample Output**:
```
Customer: 5119-NZPTV (Churn Prob: 57.6%)
Overall Relevance: 51.50%

✅ Well Addressed Risks:
   - No Contract Commitment [HIGH] → Contract offer (92.5% relevance)
   - Senior Citizen [MEDIUM] → Senior discount

⚠️ Unaddressed Risks:
   - Very Short Tenure [CRITICAL] → No tenure-specific offer
   - No Value-Added Services [MEDIUM] → No service bundle

Recommendations:
   1. Premium Retention Package (Relevance: 42.50%)
   2. Large Discount (30%) (Relevance: 57.50%)
   3. Contract Conversion Offer (Relevance: 92.50%)
```

**Usage**:
```bash
python main.py --menu
# Select: 3
```

**Improvement Opportunities**:
1. Boost relevance from 43.5% to 60%+ → +$15K per 100 customers
2. Improve conversion from 42% to 55%+ → +$35K per 100 customers
3. **Combined impact**: +$1.19M annually across 2,374 high-risk customers

---

### 4. Single Customer Analysis (Option 4)

**What it does**: Deep-dive analysis of individual customer with actionable recommendations

**Output Includes**:
- Customer profile (tenure, charges, services, contract)
- Churn probability and risk level
- Identified risk factors (Critical, High, Medium)
- Prioritized recommendations with success rates
- Conversation playbook with timing
- Objection handling scripts
- Win-back probability calculation
- Optimal contact channel
- Sentiment monitoring keywords

**Usage**:
```bash
python main.py --menu
# Select: 4
# Enter customer ID: 1234 (range: 1-7043)
```

---

### 5. High-Risk Customer Report (Option 5)

**What it does**: Generates actionable list of high-risk customers with retention strategies

**Risk Thresholds**:
1. **Critical (70%+)**: 0 customers - immediate intervention needed
2. **High (60%+)**: 448 customers - urgent action required
3. **Medium (50%+)**: ~1,500 customers - proactive outreach
4. **Custom**: Set your own threshold

**Customer Profile Pattern (60%+ risk)**:
- Tenure: 1-4 months (new customers)
- Contract: Month-to-month (no commitment)
- Internet: Fiber optic (high charges)
- Monthly Charges: $88-$101 (premium pricing)
- Total Revenue at Risk: $3.17M (36-month LTV)

**Output**:
- Excel report with customer list
- Top 10 highest-risk customers
- Recommended actions per customer
- Priority ranking for agent assignment

**Usage**:
```bash
python main.py --menu
# Select: 5
# Choose threshold: 1 (Critical), 2 (High), 3 (Medium), 4 (Custom)
```

---

### 6. Demo Mode (Option 6)

**What it does**: Demonstrates system capabilities with 3 sample customers

**Shows**:
- Full customer profiles
- Risk analysis
- Personalized recommendations
- Conversation scripts
- Expected outcomes

**Usage**:
```bash
python main.py --menu
# Select: 6
```

---

### 7. Train RL Agent (Option 7)

**What it does**: Retrains the reinforcement learning recommendation agent

**When to use**:
- Initial setup (RL agent already pre-trained)
- Improve recommendation quality
- Retrain with updated data
- Experiment with different strategies

**Training Parameters**:
- Episodes: 500-2000 (default: 1000)
- Environment: Customer response simulation
- Algorithm: Deep Q-Network (DQN)
- State space: 8 dimensions (customer features)
- Action space: 12 retention strategies

**Note**: System uses RL-based recommendations by default. Option 7 is for retraining/improvement only.

**Usage**:
```bash
python main.py --menu
# Select: 7
# Enter episodes: 1000 (or press Enter for default)
```

---

## 📊 System Performance Summary

### Model Performance
| Metric | Value | Status |
|--------|-------|--------|
| ROC-AUC | 85.19% | ✅ Excellent |
| Recall | 72.66% | ✅ Good |
| Precision | 57.20% | ⚠️ Moderate |
| F1-Score | 64.01% | ✅ Balanced |
| Accuracy | 78.32% | ✅ Good |

### Recommendation Performance
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Coverage | 100% | 100% | ✅ Perfect |
| Diversity | 84.9% | ≥80% | ✅ Excellent |
| Success Rate | 50.8% | ≥50% | ✅ Good |
| ROI | 492% | ≥200% | ✅ Excellent |

### Enhanced Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Conversion Rate | 42.0% | ≥50% | ⚠️ Below Target |
| Relevance Score | 43.5% | ≥60% | ⚠️ Below Target |
| Cost/Retention | $290 | <$400 | ✅ Good |
| ROI | 297.6% | ≥200% | ✅ Excellent |

---

## 🏗️ Technical Architecture

### Core Components

1. **`main.py`** (1,842 lines)
   - Main application entry point
   - Interactive menu system
   - All 8 menu options
   - Customer analysis logic
   - Recommendation generation
   - RL integration

2. **`churn_prediction_enhanced.py`** (300+ lines)
   - Enhanced neural network model
   - 36 feature engineering pipeline
   - Training with Focal Loss
   - Model architecture: 128→64→32 neurons

3. **`rl_recommendation_system.py`** (650+ lines)
   - DQN agent implementation
   - Customer response simulation
   - Training environment
   - Action space: 12 retention strategies

4. **`enhanced_recommendation_metrics.py`** (650+ lines)
   - Conversion tracking system
   - Relevance scoring engine
   - Financial impact analysis
   - Risk-to-recommendation mapping

### Models & Data

```
models/
├── churn_model.pth           # Enhanced neural network (36 features)
├── scaler.pkl                # Feature scaler
├── label_encoders.pkl        # Categorical encoders
├── rl_agent.pth             # Pre-trained RL agent
├── decision_threshold.json   # Optimized threshold (0.48)
├── calibration.json         # Temperature scaling
├── conversion_tracking.json  # Retention outcomes
└── training_history.json    # Training metrics
```

### Feature Engineering

**Original Features (19)**:
- Demographics: gender, senior_citizen, partner, dependents
- Services: phone, internet, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies
- Billing: tenure, monthly_charges, total_charges, contract, payment_method, paperless_billing

**Engineered Features (17)**:
- Service counts: total_services, has_premium_services
- Financial ratios: avg_charge_per_service, charge_per_month_tenure
- Risk indicators: is_new_customer, is_high_value, contract_risk_score
- Interaction features: service_engagement_score, payment_reliability
- And 10 more advanced features

**Total**: 36 input features for enhanced model

---

## 📈 Business Impact

### Current Results

**Identified High-Risk Customers**: 2,374 at 48% threshold
- 0 customers at >70% risk (Critical)
- 448 customers at 60-69% risk (High)
- ~1,500 customers at 50-59% risk (Medium)

**Financial Opportunity**:
- **Revenue at Risk**: $3.17M (36-month LTV)
- **Potential Savings**: With 55% conversion rate = $1.74M saved
- **Investment Required**: $290 per retention × 2,374 = $688,460
- **Expected ROI**: 253% = $1.05M net benefit

### Key Customer Segment (448 High-Risk)

**Profile**:
- New customers (1-4 months tenure)
- Fiber optic internet (high charges)
- Month-to-month contracts (no commitment)
- Monthly charges: $88-$101

**Recommended Actions**:
1. Welcome packages for new customers
2. Contract conversion incentives (12-24 months)
3. Fiber optimization programs
4. Loyalty discounts (20-30% off)

---

## 📚 Documentation

### Comprehensive Guides

1. **ENHANCED_METRICS_GUIDE.md** - Complete explanation of conversion tracking and relevance scoring
2. **ENHANCED_METRICS_SUMMARY.md** - Quick reference for enhanced metrics
3. **THRESHOLD_TESTING_RESULTS.md** - Analysis of 16 different decision thresholds
4. **PRECISION_IMPROVEMENT_GUIDE.md** - Strategies for improving model precision
5. **RL_IMPLEMENTATION_SUMMARY.md** - Reinforcement learning system details
6. **ASSIGNMENT_ANSWERS.md** - Complete project documentation for academic submission
7. **ASSIGNMENT_EVALUATION.md** - Evaluation criteria and scoring

---

## 🔧 Advanced Usage

### Threshold Optimization

Test different decision thresholds to balance precision vs recall:

```bash
python test_thresholds.py
# Interactive tool tests 16 thresholds from 0.35 to 0.70
# Current optimal: 0.48 (64.0% F1-Score)
```

### Conversion Tracking

Track retention outcomes over time:

```bash
# Data saved to: models/conversion_tracking.json
{
  "total_contacted": 100,
  "total_retained": 42,
  "total_cost": 29300,
  "total_revenue_saved": 116487,
  "roi": 297.6
}
```

### Custom Recommendations

Modify recommendation strategies in `main.py`:
- Adjust discount percentages
- Add new recommendation types
- Change priority rankings
- Update success rate estimates

---

## 🎯 Next Steps & Improvements

### Immediate Priorities

1. **Improve Relevance** (43.5% → 60%)
   - Add tenure-specific recommendations
   - Add service-specific offers
   - Better risk-to-recommendation mapping
   - Expected impact: +$15K per 100 customers

2. **Boost Conversion** (42% → 55%)
   - Increase discount amounts for high-risk
   - Add urgency (48-hour expiration)
   - Multi-touch campaigns (email→SMS→phone)
   - Expected impact: +$35K per 100 customers

3. **Optimize Costs** (maintain <$400/retention)
   - Email/SMS for medium-risk (cost: $5-10)
   - Phone for 65%+ risk only
   - Tiered offers (start 15%, escalate to 35%)

### Long-term Enhancements

- A/B testing framework for recommendations
- Real-time dashboard for agent monitoring
- Integration with CRM systems
- Automated campaign triggers
- Customer feedback loop
- Continuous model retraining

---

## 🤝 Contributing

This project was developed as part of academic coursework. Contributions and suggestions are welcome!

**Team Members**:
- Raja Gopal Barathan
- Arun Mohan

---

## 📄 License

This project is for educational purposes. Dataset from IBM Watson Analytics (Kaggle).

---

## 📞 Support

For questions or issues:
1. Check documentation in guides (ENHANCED_METRICS_GUIDE.md, etc.)
2. Run Option 6 (Demo) to see system in action
3. Review ASSIGNMENT_ANSWERS.md for detailed explanations

---

## ✅ System Status

- ✅ Enhanced model trained (85.19% ROC-AUC)
- ✅ RL agent trained and active
- ✅ Decision threshold optimized (0.48)
- ✅ Conversion tracking implemented
- ✅ Relevance scoring implemented
- ✅ All menu options functional
- ✅ 2,374 high-risk customers identified
- ⚠️ Relevance needs improvement (43.5% → target 60%)
- ⚠️ Conversion needs improvement (42% → target 50%)
- 💰 System is profitable (297.6% ROI)

**Ready for production use with continuous optimization!**
