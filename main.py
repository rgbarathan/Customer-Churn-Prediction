
import torch
import pandas as pd
try:
    from churn_prediction_enhanced import ChurnModel, advanced_feature_engineering
    ENHANCED_MODEL = True
except ImportError:
    from churn_prediction import ChurnModel
    ENHANCED_MODEL = False
import os
import pickle
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import json

# Import RL Recommendation System
try:
    from rl_recommendation_system import (
        DQNAgent, RetentionEnvironment, RETENTION_ACTIONS,
        get_rl_recommendations, prepare_customer_state, train_rl_agent
    )
    RL_AVAILABLE = True
    print("✓ RL Recommendation System imported successfully")
except ImportError as e:
    RL_AVAILABLE = False
    print(f"⚠️  RL Recommendation System not available: {e}")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load scaler and feature names
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

try:
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    input_dim = len(feature_names)
except FileNotFoundError:
    print("Warning: feature_names.pkl not found. Using default 19 features.")
    print("Please retrain the model by running churn_prediction.py")
    feature_names = None
    input_dim = 19

# Load churn model with appropriate architecture
if ENHANCED_MODEL and input_dim > 25:
    model = ChurnModel(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.4)
    print(f"✓ Using enhanced model architecture")
else:
    model = ChurnModel(input_dim=input_dim)
    print(f"✓ Using standard model architecture")

model.load_state_dict(torch.load('models/churn_model.pth'))
model.eval()

print("✓ Loaded churn prediction model")
print(f"✓ Model expects {input_dim} input features")
print("✓ Initialized retention recommendation engine")

# Load optional calibration and threshold artifacts
temperature = 1.0
decision_threshold = 0.5
try:
    with open('models/calibration.json', 'r') as f:
        temperature = float(json.load(f).get('temperature', 1.0))
    print(f"✓ Loaded calibration temperature T={temperature:.3f}")
except Exception:
    pass
try:
    with open('models/decision_threshold.json', 'r') as f:
        decision_threshold = float(json.load(f).get('threshold', 0.5))
    print(f"✓ Using decision threshold {decision_threshold:.3f}")
except Exception:
    pass

# Initialize RL Agent for recommendations
rl_agent = None
rl_env = None
# Health flag for RL agent (fallback to rule-based if unhealthy)
RL_HEALTHY = False
if RL_AVAILABLE:
    try:
        state_dim = 8
        action_dim = len(RETENTION_ACTIONS)
        rl_env = RetentionEnvironment()
        rl_agent = DQNAgent(state_dim, action_dim)
        
        # Try to load pre-trained RL agent
        rl_model_path = 'models/rl_agent.pth'
        if rl_agent.load(rl_model_path):
            print("✓ Loaded pre-trained RL agent for recommendations")
        else:
            print("⚠️  No pre-trained RL agent found - will use rule-based recommendations")
            print("   Run option 7 from menu to train RL agent")
        
        # Quick health check of RL agent to detect degenerate behavior
        def _rl_random_profile():
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
            churn_prob = random.uniform(0.3, 0.9)
            return profile, churn_prob

        def _assess_rl_health(agent, samples: int = 200, dominance_threshold: float = 0.95) -> bool:
            try:
                counts = [0] * action_dim
                for _ in range(samples):
                    profile, churn = _rl_random_profile()
                    state = prepare_customer_state(profile, churn)
                    action = agent.select_action(state, training=False)
                    counts[action] += 1
                dominant = max(counts) / max(1, sum(counts))
                if dominant >= dominance_threshold and counts.index(max(counts)) == 0:
                    # Dominated by "No Action" → unhealthy
                    return False
                return True
            except Exception as _:
                return False

        RL_HEALTHY = _assess_rl_health(rl_agent)
        if RL_HEALTHY:
            print("✓ RL agent health check passed - using RL-based recommendations")
        else:
            print("⚠️  RL agent health check failed - defaulting to rule-based recommendations")
    except Exception as e:
        print(f"⚠️  Could not initialize RL agent: {e}")
        rl_agent = None
        RL_HEALTHY = False

def evaluate_model_performance():
    """Evaluate model performance on the test dataset and display comprehensive metrics."""
    print("\n" + "="*70)
    print("MODEL EVALUATION METRICS")
    print("="*70)
    
    try:
        # Load the full dataset
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        print(f"\n📊 Dataset loaded: {len(df)} customers")
        
        # Store original labels before encoding
        original_labels = df['Churn'].copy()
        
        # Prepare data (same as training)
        df_encoded = df.drop(columns=['customerID', 'Churn'], errors='ignore')
        
        # CRITICAL: Convert TotalCharges to numeric BEFORE encoding loop
        df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
        df_encoded.fillna(0, inplace=True)
        
        # Load saved label encoders for consistency
        try:
            with open('models/label_encoders.pkl', 'rb') as f:
                label_encoders = pickle.load(f)
            use_saved_encoders = True
        except FileNotFoundError:
            label_encoders = {}
            use_saved_encoders = False
        
        # Encode categorical columns using saved encoders
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if use_saved_encoders and col in label_encoders:
                df_encoded[col] = label_encoders[col].transform(df_encoded[col])
            else:
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
        
        # Add engineered features
        df_encoded = add_engineered_features(df_encoded)
        
        # Encode labels
        y_true = LabelEncoder().fit_transform(original_labels)
        
        # Scale and predict
        X_scaled = scaler.transform(df_encoded)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        print("🔄 Running predictions on all customers...")
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.sigmoid(outputs / temperature).numpy().flatten()
            predictions = (probs >= decision_threshold).astype(int)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_true, probs)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Display metrics
        print("\n" + "="*70)
        print("📈 CLASSIFICATION METRICS")
        print("="*70)
        print(f"\n✓ Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  └─ Correct predictions out of all predictions")
        print(f"\n✓ Precision:          {precision:.4f} ({precision*100:.2f}%)")
        print(f"  └─ Of predicted churns, how many actually churned")
        print(f"\n✓ Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
        print(f"  └─ Of actual churns, how many we correctly identified")
        print(f"\n✓ F1-Score:           {f1:.4f} ({f1*100:.2f}%)")
        print(f"  └─ Harmonic mean of precision and recall")
        print(f"\n✓ ROC-AUC Score:      {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        print(f"  └─ Area under ROC curve (model's discriminative ability)")
        print(f"\n✓ Specificity:        {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"  └─ Of actual non-churns, how many we correctly identified")
        
        print("\n" + "="*70)
        print("📊 CONFUSION MATRIX")
        print("="*70)
        print(f"\n                    Predicted")
        print(f"                No Churn  |  Churn")
        print(f"             ─────────────┼─────────────")
        print(f"   No Churn  │    {tn:5d}     │   {fp:5d}    │ {tn+fp:5d}")
        print(f"Actual       │             │             │")
        print(f"   Churn     │    {fn:5d}     │   {tp:5d}    │ {fn+tp:5d}")
        print(f"             ─────────────┼─────────────")
        print(f"             │   {tn+fn:5d}     │  {fp+tp:5d}    │ {len(y_true):5d}")
        
        print(f"\n   True Negatives (TN):  {tn:5d} - Correctly predicted non-churn")
        print(f"   False Positives (FP): {fp:5d} - Incorrectly predicted churn")
        print(f"   False Negatives (FN): {fn:5d} - Missed actual churns (⚠️ COSTLY)")
        print(f"   True Positives (TP):  {tp:5d} - Correctly predicted churn")
        
        print("\n" + "="*70)
        print("📉 ERROR ANALYSIS")
        print("="*70)
        print(f"\n✓ False Positive Rate:  {fpr:.4f} ({fpr*100:.2f}%)")
        print(f"  └─ Non-churns incorrectly flagged as churn risk")
        print(f"\n✓ False Negative Rate:  {fnr:.4f} ({fnr*100:.2f}%)")
        print(f"  └─ Churns we failed to identify (⚠️ Most costly error)")
        print(f"\n✓ Negative Predictive Value: {npv:.4f} ({npv*100:.2f}%)")
        print(f"  └─ When we predict no churn, how often we're correct")
        
        # Business impact analysis
        print("\n" + "="*70)
        print("💰 BUSINESS IMPACT ANALYSIS")
        print("="*70)
        
        # Calculate average customer value from dataset
        avg_monthly = df['MonthlyCharges'].mean()
        avg_ltv = avg_monthly * 36  # 3-year lifetime value
        
        print(f"\n📊 Customer Economics:")
        print(f"   Average Monthly Charges: ${avg_monthly:.2f}")
        print(f"   Estimated 3-Year LTV:    ${avg_ltv:.2f}")
        
        print(f"\n🎯 Model Performance Impact:")
        print(f"   Correctly Identified Churners (TP):     {tp:5d} customers")
        print(f"   └─ Potential savings: ${tp * avg_ltv:,.2f}")
        print(f"   └─ (If retention efforts are successful)")
        
        print(f"\n⚠️  Missed Churners (FN):                 {fn:5d} customers")
        print(f"   └─ Potential lost revenue: ${fn * avg_ltv:,.2f}")
        print(f"   └─ (Customers we failed to identify)")
        
        print(f"\n💸 False Alarms (FP):                    {fp:5d} customers")
        print(f"   └─ Unnecessary retention costs")
        print(f"   └─ (Offered discounts to non-churners)")
        
        # Calculate churn distribution
        churn_count = y_true.sum()
        no_churn_count = len(y_true) - churn_count
        
        print("\n" + "="*70)
        print("📊 DATASET DISTRIBUTION")
        print("="*70)
        print(f"\n   Total Customers:     {len(y_true):5d}")
        print(f"   Churned:             {churn_count:5d} ({churn_count/len(y_true)*100:.2f}%)")
        print(f"   Not Churned:         {no_churn_count:5d} ({no_churn_count/len(y_true)*100:.2f}%)")
        print(f"   Class Imbalance:     {no_churn_count/churn_count:.2f}:1 (non-churn:churn)")
        
        # Prediction distribution
        pred_churn = predictions.sum()
        pred_no_churn = len(predictions) - pred_churn
        print(f"\n   Predicted Churners:  {pred_churn:5d} ({pred_churn/len(predictions)*100:.2f}%)")
        print(f"   Predicted Stable:    {pred_no_churn:5d} ({pred_no_churn/len(predictions)*100:.2f}%)")
        
        # Risk categories analysis
        print("\n" + "="*70)
        print("🎯 RISK STRATIFICATION ANALYSIS")
        print("="*70)
        
        critical_risk = (probs > 0.7).sum()
        high_risk = ((probs > decision_threshold) & (probs <= 0.7)).sum()
        medium_risk = ((probs > 0.3) & (probs <= decision_threshold)).sum()
        low_risk = (probs <= 0.3).sum()
        
        print(f"\n   🔴 Critical Risk (>70%):     {critical_risk:5d} customers ({critical_risk/len(probs)*100:.2f}%)")
        print(f"   🟠 High Risk (>{decision_threshold:.0%}-70%): {high_risk:5d} customers ({high_risk/len(probs)*100:.2f}%)")
        print(f"   🟡 Medium Risk (30%-{decision_threshold:.0%}): {medium_risk:5d} customers ({medium_risk/len(probs)*100:.2f}%)")
        print(f"   🟢 Low Risk (<30%):          {low_risk:5d} customers ({low_risk/len(probs)*100:.2f}%)")
        
        print("\n" + "="*70)
        print("📋 DETAILED CLASSIFICATION REPORT")
        print("="*70)
        print("\n" + classification_report(y_true, predictions, target_names=['No Churn', 'Churn'], digits=4))
        
        print("\n" + "="*70)
        print("✅ MODEL EVALUATION COMPLETE")
        print("="*70)
        
        # Summary interpretation with more context
        print("\n💡 KEY TAKEAWAYS & RECOMMENDATIONS:")
        
        # Overall Performance
        print(f"\n   📊 Overall Performance:")
        if accuracy > 0.8:
            print(f"      ✓ Strong accuracy ({accuracy:.2%}) - Model is reliable")
        elif accuracy > 0.7:
            print(f"      ⚠️  Moderate accuracy ({accuracy:.2%}) - Room for improvement")
        else:
            print(f"      ❌ Low accuracy ({accuracy:.2%}) - Needs significant improvement")
        
        # Recall Analysis (Most Important for Churn)
        print(f"\n   🎯 Churn Detection (Recall): {recall:.2%}")
        if recall > 0.7:
            print(f"      ✓ Good at catching churners - Missing only {fnr*100:.1f}% of at-risk customers")
        elif recall > 0.5:
            print(f"      ⚠️  Missing {fnr*100:.1f}% of churners - Consider lowering prediction threshold")
        else:
            print(f"      ❌ Missing {fnr*100:.1f}% of churners - Critical issue for retention strategy")
        
        # Precision Analysis
        print(f"\n   🎪 Prediction Reliability (Precision): {precision:.2%}")
        if precision > 0.7:
            print(f"      ✓ Reliable predictions - Only {fpr*100:.1f}% false alarms")
        elif precision > 0.5:
            print(f"      ⚠️  {fpr*100:.1f}% false alarm rate - Some wasted retention efforts")
        else:
            print(f"      ❌ High false alarm rate ({fpr*100:.1f}%) - Too many unnecessary interventions")
        
        # F1-Score (Balance)
        print(f"\n   ⚖️  Model Balance (F1-Score): {f1:.2%}")
        if f1 > 0.7:
            print(f"      ✓ Well-balanced between catching churners and avoiding false alarms")
        elif f1 > 0.5:
            print(f"      ⚠️  Moderate balance - May need to optimize threshold")
        else:
            print(f"      ❌ Poor balance - Need to retrain or adjust decision threshold")
        
        # ROC-AUC Score
        print(f"\n   📈 Discriminative Ability (ROC-AUC): {roc_auc:.2%}")
        if roc_auc > 0.8:
            print(f"      ✓ Excellent - Model can distinguish churners from non-churners")
        elif roc_auc > 0.7:
            print(f"      ✓ Good discrimination capability")
        else:
            print(f"      ⚠️  Limited discrimination - Model struggles to separate classes")
        
        # Business Recommendation
        print(f"\n   💼 BUSINESS RECOMMENDATION:")
        if recall < 0.6:
            print(f"      🚨 PRIORITY: Improve churn detection - Currently missing ${fn * avg_ltv:,.0f} in at-risk revenue")
        if precision < 0.6:
            print(f"      💸 Consider: Reduce false alarms - Currently wasting retention budget on {fp} stable customers")
        if f1 > 0.7 and roc_auc > 0.8:
            print(f"      ✅ Model is production-ready - Deploy for retention strategy")
        elif f1 > 0.6:
            print(f"      ⚠️  Model is usable but monitor performance closely")
        else:
            print(f"      ❌ Model needs improvement before full deployment")
        
        print("\n")
        
    except FileNotFoundError:
        print("\n❌ Error: Dataset file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found!")
        print("Please ensure the dataset is in the current directory.")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def evaluate_recommendations_quality():
    """Evaluate the quality and effectiveness of retention recommendations."""
    print("\n" + "="*70)
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("="*70)
    
    try:
        # Load dataset
        df = load_dataset()
        if df is None:
            return
        
        print(f"\n🔄 Analyzing recommendation system on {len(df)} customers...")
        
        # Prepare data
        df_original = df.copy()
        df_encoded = df.drop(columns=['customerID', 'Churn'], errors='ignore')
        
        # CRITICAL: Convert TotalCharges to numeric BEFORE encoding loop
        df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
        df_encoded.fillna(0, inplace=True)
        
        # Load saved label encoders for consistency
        try:
            with open('models/label_encoders.pkl', 'rb') as f:
                label_encoders = pickle.load(f)
            use_saved_encoders = True
        except FileNotFoundError:
            label_encoders = {}
            use_saved_encoders = False
        
        # Encode categorical columns using saved encoders
        from sklearn.preprocessing import LabelEncoder
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if use_saved_encoders and col in label_encoders:
                df_encoded[col] = label_encoders[col].transform(df_encoded[col])
            else:
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
        
        # Predict for all customers
        df_encoded_enhanced = add_engineered_features(df_encoded)
        scaled_data = scaler.transform(df_encoded_enhanced)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(tensor_data)
            churn_probs = torch.sigmoid(outputs / temperature).numpy().flatten()
        
        # Focus on high-risk customers (>= tuned decision threshold)
        high_risk_mask = churn_probs >= decision_threshold
        high_risk_count = high_risk_mask.sum()
        
        print(f"✓ Found {high_risk_count} high-risk customers (≥{decision_threshold:.0%} churn probability)")
        
        # Analyze recommendations for a sample of high-risk customers
        sample_size = min(100, high_risk_count)
        high_risk_indices = np.where(high_risk_mask)[0]
        sample_indices = np.random.choice(high_risk_indices, sample_size, replace=False)
        
        print(f"✓ Evaluating recommendations for {sample_size} sample customers...\n")
        
        # Metrics to collect
        recommendation_counts = []
        priority_distributions = []
        risk_factor_counts = []
        coverage_scores = []
        expected_success_rates = []
        recommendation_types = {
            'pricing': 0,
            'contract': 0,
            'service': 0,
            'loyalty': 0,
            'engagement': 0
        }
        
        for idx in sample_indices:
            # Get customer data
            customer_data = df_encoded.iloc[[idx]]
            churn_prob = churn_probs[idx]
            
            # Generate recommendations
            profile = extract_customer_profile(customer_data)
            risk_factors = identify_risk_factors(profile, churn_prob)
            recommendations = generate_recommendations(profile, risk_factors, churn_prob)
            winback = calculate_winback_probability(profile, churn_prob)
            
            # Collect metrics
            recommendation_counts.append(len(recommendations))
            risk_factor_counts.append(len(risk_factors))
            
            # Priority distribution
            priorities = [rec['priority'] for rec in recommendations]
            priority_distributions.extend(priorities)
            
            # Coverage: how many risk factors have corresponding recommendations
            coverage = min(1.0, len(recommendations) / max(1, len(risk_factors)))
            coverage_scores.append(coverage)
            
            # Expected success rate
            expected_success_rates.append(winback['probability'])
            
            # Categorize recommendations
            for rec in recommendations:
                action = rec['action'].lower()
                if 'cost' in action or 'discount' in action or 'price' in action or 'charge' in action:
                    recommendation_types['pricing'] += 1
                elif 'contract' in action or 'commitment' in action:
                    recommendation_types['contract'] += 1
                elif 'service' in action or 'bundle' in action or 'add-on' in action:
                    recommendation_types['service'] += 1
                elif 'loyalty' in action or 'reward' in action or 'tenure' in action:
                    recommendation_types['loyalty'] += 1
                else:
                    recommendation_types['engagement'] += 1
        
        # Calculate aggregate metrics
        avg_recommendations = np.mean(recommendation_counts)
        avg_risk_factors = np.mean(risk_factor_counts)
        avg_coverage = np.mean(coverage_scores)
        avg_success_rate = np.mean(expected_success_rates)
        
        # Display results
        print("="*70)
        print("📊 RECOMMENDATION SYSTEM METRICS")
        print("="*70)
        
        print(f"\n📋 RECOMMENDATION GENERATION:")
        print(f"   Average Recommendations per Customer:  {avg_recommendations:.2f}")
        print(f"   Average Risk Factors per Customer:     {avg_risk_factors:.2f}")
        print(f"   Min Recommendations:                   {min(recommendation_counts)}")
        print(f"   Max Recommendations:                   {max(recommendation_counts)}")
        
        print(f"\n🎯 COVERAGE ANALYSIS:")
        print(f"   Average Coverage Score:                {avg_coverage:.2%}")
        print(f"   └─ How well recommendations address all risk factors")
        perfect_coverage = sum(1 for c in coverage_scores if c >= 1.0)
        print(f"   Perfect Coverage (100%):               {perfect_coverage}/{sample_size} customers ({perfect_coverage/sample_size*100:.1f}%)")
        
        print(f"\n✨ RECOMMENDATION QUALITY:")
        priority_1 = priority_distributions.count(1)
        priority_2 = priority_distributions.count(2)
        priority_3 = priority_distributions.count(3)
        total_recs = len(priority_distributions)
        
        print(f"   Priority 1 (Critical):                 {priority_1}/{total_recs} ({priority_1/total_recs*100:.1f}%)")
        print(f"   Priority 2 (High):                     {priority_2}/{total_recs} ({priority_2/total_recs*100:.1f}%)")
        print(f"   Priority 3 (Medium):                   {priority_3}/{total_recs} ({priority_3/total_recs*100:.1f}%)")
        
        print(f"\n📈 EXPECTED SUCCESS RATE:")
        print(f"   Average Win-Back Probability:          {avg_success_rate:.2%}")
        high_success = sum(1 for s in expected_success_rates if s > 0.7)
        medium_success = sum(1 for s in expected_success_rates if 0.4 < s <= 0.7)
        low_success = sum(1 for s in expected_success_rates if s <= 0.4)
        print(f"   High Success Rate (>70%):              {high_success}/{sample_size} ({high_success/sample_size*100:.1f}%)")
        print(f"   Medium Success Rate (40-70%):          {medium_success}/{sample_size} ({medium_success/sample_size*100:.1f}%)")
        print(f"   Low Success Rate (<40%):               {low_success}/{sample_size} ({low_success/sample_size*100:.1f}%)")
        
        print(f"\n🎨 RECOMMENDATION DIVERSITY:")
        total_type_recs = sum(recommendation_types.values())
        print(f"   Pricing/Discount Offers:               {recommendation_types['pricing']:3d} ({recommendation_types['pricing']/total_type_recs*100:.1f}%)")
        print(f"   Contract/Commitment Offers:            {recommendation_types['contract']:3d} ({recommendation_types['contract']/total_type_recs*100:.1f}%)")
        print(f"   Service/Bundle Upgrades:               {recommendation_types['service']:3d} ({recommendation_types['service']/total_type_recs*100:.1f}%)")
        print(f"   Loyalty Rewards:                       {recommendation_types['loyalty']:3d} ({recommendation_types['loyalty']/total_type_recs*100:.1f}%)")
        print(f"   Engagement/Other:                      {recommendation_types['engagement']:3d} ({recommendation_types['engagement']/total_type_recs*100:.1f}%)")
        
        # Calculate diversity score (Shannon entropy)
        from math import log
        diversity_score = 0
        for count in recommendation_types.values():
            if count > 0:
                p = count / total_type_recs
                diversity_score -= p * log(p)
        max_diversity = log(len(recommendation_types))
        normalized_diversity = diversity_score / max_diversity
        
        print(f"\n   Diversity Score:                       {normalized_diversity:.2%}")
        print(f"   └─ Higher is better (more balanced recommendation types)")
        
        # Business impact simulation
        print("\n" + "="*70)
        print("💰 PROJECTED BUSINESS IMPACT")
        print("="*70)
        
        avg_monthly = df['MonthlyCharges'].mean()
        avg_ltv = avg_monthly * 36
        
        # Calculate potential savings
        total_at_risk = high_risk_count
        total_ltv_at_risk = total_at_risk * avg_ltv
        expected_saved = total_at_risk * avg_success_rate
        expected_revenue_saved = expected_saved * avg_ltv
        
        print(f"\n📊 Retention Opportunity:")
        print(f"   Customers at High Risk:                {total_at_risk:,}")
        print(f"   Total LTV at Risk:                     ${total_ltv_at_risk:,.2f}")
        print(f"   Expected Customers Saved:              {expected_saved:.0f}")
        print(f"   Expected Revenue Saved:                ${expected_revenue_saved:,.2f}")
        print(f"   Success Rate:                          {avg_success_rate:.2%}")
        
        # Cost-benefit analysis
        avg_incentive_cost = 200  # Estimated average cost per retention offer
        total_incentive_cost = total_at_risk * avg_incentive_cost
        net_benefit = expected_revenue_saved - total_incentive_cost
        roi = (net_benefit / total_incentive_cost) * 100 if total_incentive_cost > 0 else 0
        
        print(f"\n💵 Cost-Benefit Analysis:")
        print(f"   Estimated Retention Costs:             ${total_incentive_cost:,.2f}")
        print(f"   └─ (${avg_incentive_cost} avg per customer)")
        print(f"   Expected Revenue Saved:                ${expected_revenue_saved:,.2f}")
        print(f"   Net Benefit:                           ${net_benefit:,.2f}")
        print(f"   Return on Investment (ROI):            {roi:.1f}%")
        
        # Recommendation system performance summary
        print("\n" + "="*70)
        print("✅ RECOMMENDATION SYSTEM ASSESSMENT")
        print("="*70)
        
        print("\n💡 KEY FINDINGS:")
        
        if avg_coverage >= 0.9:
            print("   ✓ Excellent coverage - recommendations address most risk factors")
        elif avg_coverage >= 0.7:
            print("   ✓ Good coverage - most risk factors have recommendations")
        else:
            print("   ⚠️ Coverage could be improved - some risk factors not addressed")
        
        if normalized_diversity >= 0.7:
            print("   ✓ Good diversity - varied recommendation strategies")
        else:
            print("   ⚠️ Low diversity - recommendations too similar")
        
        if avg_success_rate >= 0.6:
            print("   ✓ High expected success rate - strong retention potential")
        elif avg_success_rate >= 0.4:
            print("   ✓ Moderate success rate - reasonable retention outcomes")
        else:
            print("   ⚠️ Low success rate - may need strategy adjustment")
        
        if roi > 100:
            print("   ✓ Excellent ROI - highly cost-effective recommendations")
        elif roi > 0:
            print("   ✓ Positive ROI - recommendations are profitable")
        else:
            print("   ⚠️ Negative ROI - need to optimize costs")
        
        if priority_1/total_recs >= 0.3:
            print("   ✓ Good prioritization - critical actions highlighted")
        else:
            print("   ⚠️ Weak prioritization - unclear action urgency")
        
        # Specific recommendations for improvement
        print("\n🔧 RECOMMENDATIONS FOR SYSTEM IMPROVEMENT:")
        
        improvements = []
        if avg_coverage < 0.8:
            improvements.append("   • Increase recommendation coverage to address all risk factors")
        if normalized_diversity < 0.6:
            improvements.append("   • Diversify recommendation strategies across categories")
        if avg_success_rate < 0.5:
            improvements.append("   • Adjust win-back probability calculations")
        if recommendation_types['pricing'] / total_type_recs > 0.5:
            improvements.append("   • Reduce over-reliance on pricing discounts")
        if avg_recommendations < 3:
            improvements.append("   • Generate more recommendations per customer")
        
        if improvements:
            for imp in improvements:
                print(imp)
        else:
            print("   ✓ System is performing well - no critical improvements needed")
        
        print("\n" + "="*70)
        print("✅ RECOMMENDATION EVALUATION COMPLETE")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during recommendation evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def add_engineered_features(customer_data):
    """Add engineered features to customer data."""
    if ENHANCED_MODEL:
        return advanced_feature_engineering(customer_data)
    
    # Legacy feature engineering
    if input_dim == 19:
        return customer_data  # Old model, no engineered features
    
    df = customer_data.copy()
    
    # 1. Tenure to charges ratio
    df['tenure_to_charges_ratio'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1e-6)
    
    # 2. Service density (services per dollar)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_count'] = df[service_cols].sum(axis=1)
    df['service_density'] = df['service_count'] / (df['MonthlyCharges'] + 1e-6)
    
    # 3. Payment reliability
    df['payment_reliability'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1e-6)
    
    return df

def extract_customer_profile(customer_data):
    """Extract customer profile from raw data for analysis."""
    row = customer_data.iloc[0]
    
    # Map encoded values back to meaningful labels
    # LabelEncoder maps categories alphabetically. For InternetService the dataset categories
    # are typically ['DSL', 'Fiber optic', 'No'] which map to {0:'DSL',1:'Fiber optic',2:'No'}.
    contract_types = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
    internet_types = {0: 'DSL', 1: 'Fiber optic', 2: 'No'}
    
    profile = {
        'senior_citizen': bool(row['SeniorCitizen']),
        'has_partner': bool(row['Partner']),
        'has_dependents': bool(row['Dependents']),
        'tenure_months': int(row['tenure']),
        'has_phone': bool(row['PhoneService']),
        'internet_service': internet_types.get(int(row['InternetService']), 'Unknown'),
        'has_online_security': bool(row['OnlineSecurity']),
        'has_online_backup': bool(row['OnlineBackup']),
        'has_device_protection': bool(row['DeviceProtection']),
        'has_tech_support': bool(row['TechSupport']),
        'contract_type': contract_types.get(int(row['Contract']), 'Unknown'),
        'monthly_charges': float(row['MonthlyCharges']),
        'total_charges': float(row['TotalCharges'])
    }
    
    # Calculate derived metrics
    profile['service_count'] = sum([
        profile['has_phone'],
        profile['internet_service'] != 'No',
        profile['has_online_security'],
        profile['has_online_backup'],
        profile['has_device_protection'],
        profile['has_tech_support']
    ])
    
    profile['has_addons'] = any([
        profile['has_online_security'],
        profile['has_online_backup'],
        profile['has_device_protection'],
        profile['has_tech_support']
    ])
    
    return profile

def identify_risk_factors(profile, churn_prob):
    """Identify specific factors contributing to churn risk."""
    risk_factors = []
    
    # Tenure-based risks
    if profile['tenure_months'] < 3:
        risk_factors.append({
            'factor': 'Very Short Tenure',
            'severity': 'CRITICAL',
            'description': f"Only {profile['tenure_months']} month(s) - highest churn risk period"
        })
    elif profile['tenure_months'] < 12:
        risk_factors.append({
            'factor': 'Short Tenure',
            'severity': 'HIGH',
            'description': f"{profile['tenure_months']} months - still in high-risk period"
        })
    
    # Pricing risks
    if profile['monthly_charges'] > 90:
        risk_factors.append({
            'factor': 'High Monthly Charges',
            'severity': 'HIGH',
            'description': f"${profile['monthly_charges']:.2f}/month may cause price sensitivity"
        })
    
    # Contract risks
    if profile['contract_type'] == 'Month-to-month':
        risk_factors.append({
            'factor': 'No Contract Commitment',
            'severity': 'HIGH',
            'description': "Easy to cancel without penalties"
        })
    
    # Service engagement risks
    if profile['service_count'] <= 1:
        risk_factors.append({
            'factor': 'Low Service Engagement',
            'severity': 'MEDIUM',
            'description': f"Only {profile['service_count']} service(s) - low switching costs"
        })
    
    if not profile['has_addons']:
        risk_factors.append({
            'factor': 'No Value-Added Services',
            'severity': 'MEDIUM',
            'description': "Missing security, backup, protection - easy to replicate"
        })
    
    # Senior citizen
    if profile['senior_citizen']:
        risk_factors.append({
            'factor': 'Senior Citizen',
            'severity': 'MEDIUM',
            'description': "May be more price-sensitive"
        })
    
    # Premium service with short tenure
    if profile['internet_service'] == 'Fiber optic' and profile['tenure_months'] < 6:
        risk_factors.append({
            'factor': 'New Premium Customer',
            'severity': 'HIGH',
            'description': "Fiber customer with high expectations"
        })
    
    return risk_factors

def generate_recommendations(profile, risk_factors, churn_prob):
    """Generate actionable retention recommendations."""
    
    # Try RL-based recommendations first
    if rl_agent is not None and RL_AVAILABLE and RL_HEALTHY:
        try:
            rl_recommendations = get_rl_recommendations(profile, churn_prob, rl_agent, top_k=5)
            # Add RL badge to differentiate
            for rec in rl_recommendations:
                rec['method'] = 'RL-Based'
            return rl_recommendations
        except Exception as e:
            print(f"⚠️  RL recommendation failed: {e}, falling back to rule-based")
    
    # Fallback to rule-based recommendations
    recommendations = []
    
    # Critical risk - immediate action
    if churn_prob > 0.6:
        recommendations.append({
            'priority': 1,
            'icon': '🚨',
            'action': 'Immediate Outreach Required',
            'description': 'Contact within 24 hours with exclusive retention offer',
            'expected_impact': '65% retention success rate',
            'method': 'Rule-Based'
        })
    
    # Address specific risk factors
    for risk in risk_factors:
        if risk['factor'] == 'High Monthly Charges':
            if profile['senior_citizen']:
                savings = 30
                offer = f"Senior Bundle Special: Reduce to ${profile['monthly_charges'] - savings:.2f}/month"
            else:
                savings = 25
                offer = f"Loyalty Discount: ${savings}/month off"
            
            recommendations.append({
                'priority': 2,
                'icon': '💰',
                'action': f"Reduce Monthly Cost (Currently ${profile['monthly_charges']:.2f})",
                'description': offer,
                'expected_impact': f'Save ${savings}/month - 72% success rate',
                'method': 'Rule-Based'
            })
        
        if risk['factor'] == 'No Contract Commitment':
            recommendations.append({
                'priority': 2,
                'icon': '📝',
                'action': 'Convert to Long-Term Contract',
                'description': '24-month: $25/month off + Price Lock Guarantee',
                'expected_impact': '58% conversion rate',
                'method': 'Rule-Based'
            })
        
        if risk['factor'] in ['Very Short Tenure', 'Short Tenure']:
            recommendations.append({
                'priority': 1 if profile['tenure_months'] < 3 else 2,
                'icon': '🆕',
                'action': f"New Customer Retention (Tenure: {profile['tenure_months']} mo)",
                'description': '50% off next 3 months + Free premium channels',
                'expected_impact': '68% retention success',
                'method': 'Rule-Based'
            })
        
        if risk['factor'] == 'Senior Citizen':
            recommendations.append({
                'priority': 2,
                'icon': '👴',
                'action': 'Senior Advantage Program',
                'description': '$15/month discount + Free tech support',
                'expected_impact': '75% enrollment success',
                'method': 'Rule-Based'
            })
        
        if risk['factor'] in ['Low Service Engagement', 'No Value-Added Services']:
            recommendations.append({
                'priority': 3,
                'icon': '📦',
                'action': f"Increase Service Bundle (Current: {profile['service_count']})",
                'description': 'Free add-ons for 6 months (Security, Backup, Streaming)',
                'expected_impact': 'Increase LTV by $500-1000',
                'method': 'Rule-Based'
            })
    
    # Loyalty rewards for long-term customers
    if profile['tenure_months'] > 24:
        discount = min(30, profile['tenure_months'] // 12 * 10)
        recommendations.append({
            'priority': 2,
            'icon': '⭐',
            'action': f"Loyalty Rewards ({profile['tenure_months']} months tenure)",
            'description': f'${discount}/month discount + Equipment upgrade',
            'expected_impact': '82% retention success',
            'method': 'Rule-Based'
        })
    
    # Remove duplicates and sort by priority
    seen = set()
    unique_recs = []
    for rec in recommendations:
        if rec['action'] not in seen:
            seen.add(rec['action'])
            unique_recs.append(rec)
    
    unique_recs.sort(key=lambda x: x['priority'])
    return unique_recs

def calculate_winback_probability(profile, churn_prob):
    """Calculate probability of successfully retaining the customer."""
    base_score = 1.0 - churn_prob  # Start with inverse of churn probability
    
    # Adjust based on tenure (longer tenure = easier to retain)
    if profile['tenure_months'] > 24:
        tenure_factor = 1.3  # 30% boost for loyal customers
    elif profile['tenure_months'] > 12:
        tenure_factor = 1.15
    elif profile['tenure_months'] > 6:
        tenure_factor = 1.0
    else:
        tenure_factor = 0.85  # Harder to retain new customers
    
    # Adjust based on contract type (committed customers easier to retain)
    if profile['contract_type'] == 'Two year':
        contract_factor = 1.25
    elif profile['contract_type'] == 'One year':
        contract_factor = 1.1
    else:
        contract_factor = 0.9  # Month-to-month harder to retain
    
    # Adjust based on engagement (more services = easier to retain)
    if profile['service_count'] >= 5:
        engagement_factor = 1.2
    elif profile['service_count'] >= 3:
        engagement_factor = 1.1
    else:
        engagement_factor = 0.95
    
    # Adjust based on payment reliability (high charges with good tenure = loyal)
    payment_reliability = profile['total_charges'] / (profile['tenure_months'] * profile['monthly_charges'] + 1e-6)
    if payment_reliability > 0.95:
        payment_factor = 1.15  # Consistent payer
    elif payment_reliability > 0.8:
        payment_factor = 1.0
    else:
        payment_factor = 0.9  # Payment issues
    
    # Calculate final win-back probability
    winback_prob = min(0.95, base_score * tenure_factor * contract_factor * engagement_factor * payment_factor)
    
    # Determine strategy
    if winback_prob > 0.7:
        strategy = "AGGRESSIVE_SAVE"
        strategy_desc = "High retention likelihood - offer premium incentives"
    elif winback_prob > 0.4:
        strategy = "NEGOTIATION"
        strategy_desc = "Moderate retention likelihood - negotiate and escalate if needed"
    else:
        strategy = "BEST_EFFORT"
        strategy_desc = "Low retention likelihood - document for future win-back campaign"
    
    return {
        'probability': winback_prob,
        'strategy': strategy,
        'description': strategy_desc
    }

def generate_objection_handlers(profile, churn_prob):
    """Generate objection handling scripts for common customer concerns."""
    handlers = []
    
    # Price objection
    if profile['monthly_charges'] > 80:
        max_discount = min(40, int(profile['monthly_charges'] * 0.3))
        handlers.append({
            'objection': '💰 "Too expensive / Can\'t afford it"',
            'response': f"I completely understand budget concerns. I can reduce your bill by ${max_discount}/month right now.",
            'action': f"Apply ${max_discount}/month discount immediately",
            'fallback': f"Alternative: 50% off for 3 months (saves ${int(profile['monthly_charges']*0.5*3)})",
            'escalation': 'If declined: Transfer to retention specialist with authority for higher discounts'
        })
    
    # Competitor objection
    handlers.append({
        'objection': '🏢 "Competitor has better deal"',
        'response': "I\'d love to match or beat that offer. What are they offering specifically?",
        'action': "Price match + $100 gift card + premium channel upgrade",
        'fallback': "Alternative: Match their price + additional service add-ons",
        'escalation': 'If competitor offer > our best: Escalate to manager for approval'
    })
    
    # Service quality objection
    if profile['internet_service'] != 'No':
        handlers.append({
            'objection': '📶 "Service quality issues / Slow internet"',
            'response': "I sincerely apologize for that experience. Let\'s fix this immediately.",
            'action': "Schedule priority tech visit (within 24hrs) + 1 month bill credit",
            'fallback': "Alternative: Upgrade to fiber (if available) at current price",
            'escalation': 'URGENT: Create service ticket + notify technical operations'
        })
    
    # Contract flexibility objection
    if profile['contract_type'] != 'Month-to-month':
        handlers.append({
            'objection': '📋 "Locked into contract / Want flexibility"',
            'response': "I understand wanting flexibility. How about we convert you to month-to-month with a loyalty discount?",
            'action': "Convert to month-to-month + maintain contract discount for 6 months",
            'fallback': "Alternative: Reduce contract to 1-year with same discount",
            'escalation': 'If insistent: Waive early termination fee + retention offer'
        })
    
    # Moving/Relocation objection
    handlers.append({
        'objection': '🏠 "Moving / Relocating"',
        'response': "We can transfer your service to your new address with all current benefits.",
        'action': "Free installation at new address + 2 months at 50% off",
        'fallback': "Alternative: Suspend service for up to 3 months (if temporary move)",
        'escalation': 'If outside service area: Offer early termination fee waiver'
    })
    
    return handlers

def generate_conversation_flow(profile, churn_prob, recommendations):
    """Generate step-by-step conversation playbook for agent."""
    ltv = profile['monthly_charges'] * 36
    
    flow = {
        'step1': {
            'phase': 'BUILD RAPPORT',
            'timing': '0-2 minutes',
            'objective': 'Establish connection and gather context',
            'script': [
                f"Thank you for being a Comcast customer for {profile['tenure_months']} months, we truly value your business.",
                "I'm calling today because you're one of our valued customers and I want to make sure you're getting the best experience.",
                f"How has your {'internet' if profile['internet_service'] != 'No' else 'phone'} service been working for you lately?"
            ],
            'listen_for': ['Satisfaction level', 'Any complaints', 'Specific needs'],
            'notes': '🎯 Goal: Get customer talking, identify pain points'
        },
        'step2': {
            'phase': 'IDENTIFY PAIN POINTS',
            'timing': '2-5 minutes',
            'objective': 'Uncover reasons for potential churn',
            'script': [
                f"I see you're currently on our {profile['contract_type']} plan at ${profile['monthly_charges']:.2f}/month.",
                "What's most important to you: saving money, getting more services, or improving reliability?",
                "Have you looked at other providers or considered making any changes?"
            ],
            'listen_for': ['Price concerns', 'Service issues', 'Competitor mentions', 'Life changes'],
            'notes': '⚠️ CRITICAL: Listen actively and take notes on objections'
        },
        'step3': {
            'phase': 'PRESENT PRIMARY OFFER',
            'timing': '5-8 minutes',
            'objective': 'Present most relevant retention offer',
            'primary_offer': recommendations[0] if recommendations else None,
            'script': [
                "Based on what you've shared, I have an exclusive offer designed specifically for you.",
                f"[Present: {recommendations[0]['description'] if recommendations else 'best available offer'}]",
                "This offer is available for the next 48 hours and can save you significant money.",
                "Would you like me to apply this to your account right now?"
            ],
            'if_yes': 'Proceed to Step 4 (Closing)',
            'if_no': 'Present Alternative Offer A',
            'alternative_a': recommendations[1] if len(recommendations) > 1 else None,
            'alternative_b': recommendations[2] if len(recommendations) > 2 else None,
            'notes': '💡 If customer hesitates: Use objection handlers'
        },
        'step4': {
            'phase': 'CLOSING',
            'timing': '8-10 minutes',
            'objective': 'Secure commitment or schedule follow-up',
            'script': [
                "Great! Let me summarize what we've agreed on today:",
                f"[Recap: Discount, services, contract terms]",
                "I'll apply this immediately - you'll see it on your next bill.",
                "Is there anything else I can help you with today?"
            ],
            'if_accepted': 'Apply offer + send confirmation email + schedule follow-up call in 30 days',
            'if_rejected': 'Schedule callback in 24-48 hours + escalate to retention specialist',
            'follow_up': 'Send SMS confirmation with offer details within 1 hour',
            'notes': '✅ Document outcome in CRM + set reminder for follow-up'
        }
    }
    
    return flow

def determine_next_contact_channel(profile, churn_prob):
    """Determine best channel for follow-up contact."""
    # High churn risk = immediate phone call
    if churn_prob > 0.7:
        return {
            'primary': '📞 Phone Call',
            'timing': 'Within 24 hours',
            'reason': 'Critical risk requires immediate personal contact',
            'backup': 'SMS if no answer (max 3 attempts)',
            'message': 'Urgent: Special retention offer expires soon. Call us at 1-800-COMCAST'
        }
    
    # Senior citizens often prefer phone
    elif profile['senior_citizen']:
        return {
            'primary': '📞 Phone Call',
            'timing': 'Within 48 hours (prefer morning 9-11am)',
            'reason': 'Senior customer - phone preferred for clear communication',
            'backup': 'Physical mail with large print offer details',
            'message': 'Important information about your Comcast account'
        }
    
    # High-value customers get personalized contact
    elif profile['monthly_charges'] > 100:
        return {
            'primary': '📞 Phone Call',
            'timing': 'Within 48 hours',
            'reason': 'High-value customer deserves personal attention',
            'backup': '📧 Email with video message from account manager',
            'message': 'Exclusive VIP retention offer - Limited time'
        }
    
    # Tech-savvy customers (multiple services, young)
    elif profile['service_count'] >= 4 and not profile['senior_citizen']:
        return {
            'primary': '📱 SMS/Text',
            'timing': 'Within 2 hours',
            'reason': 'Tech-savvy customer prefers quick digital contact',
            'backup': '📧 Email with clickable offer link',
            'message': '🎁 Special offer just for you! Click here to save $XX/month [LINK]'
        }
    
    # Default to email
    else:
        return {
            'primary': '📧 Email',
            'timing': 'Within 24 hours',
            'reason': 'Standard follow-up with detailed offer information',
            'backup': '📞 Phone call if no response in 48 hours',
            'message': 'Your Personalized Comcast Retention Offer - Don\'t Miss Out!'
        }

def generate_sentiment_guidance(profile, churn_prob):
    """Provide real-time sentiment monitoring guidance for agent."""
    guidance = {
        'watch_for': [],
        'if_frustrated': {},
        'if_confused': {},
        'if_interested': {},
        'if_angry': {}
    }
    
    # Keywords to monitor
    guidance['watch_for'] = [
        '❌ Negative: "cancel", "disconnect", "switching", "expensive", "poor service", "fed up"',
        '⚠️ Warning: "thinking about", "considering", "looking at", "competitor", "better deal"',
        '✅ Positive: "interested", "sounds good", "maybe", "tell me more", "what else"',
        '💰 Price-focused: "cost", "price", "afford", "budget", "cheaper", "discount"'
    ]
    
    # If customer sounds frustrated
    guidance['if_frustrated'] = {
        'indicators': 'Raised voice, sighing, repetitive complaints, interrupting',
        'immediate_action': '🚨 STOP SELLING - Switch to empathy mode',
        'script': "I can hear your frustration, and I want to make this right. Let\'s focus on solving your specific issue first.",
        'next_step': 'Address root cause before presenting offers',
        'escalation': 'If frustration continues: "Let me connect you with my supervisor who can authorize additional solutions."'
    }
    
    # If customer sounds confused
    guidance['if_confused'] = {
        'indicators': 'Asking for clarification, long pauses, "I don\'t understand"',
        'immediate_action': '⚠️ Simplify explanation - avoid technical jargon',
        'script': "Let me explain that more simply: [use plain language]",
        'next_step': 'Confirm understanding: "Does that make sense?"',
        'tip': 'Use analogies and concrete examples'
    }
    
    # If customer shows interest
    guidance['if_interested'] = {
        'indicators': 'Asking questions, "tell me more", engaged tone',
        'immediate_action': '✅ Strike while hot - present offer details',
        'script': "I\'m glad you\'re interested! Let me walk you through exactly what you\'ll get...",
        'next_step': 'Create urgency: "This offer expires in 48 hours"',
        'closing': 'Ask for commitment: "Can I apply this to your account right now?"'
    }
    
    # If customer is angry
    guidance['if_angry'] = {
        'indicators': 'Yelling, threats to cancel, demanding, aggressive language',
        'immediate_action': '🚨 URGENT - De-escalation protocol',
        'script': "I sincerely apologize. This is not the experience you deserve. Let me personally ensure we fix this.",
        'next_step': 'Immediate resolution: Credit account + schedule tech visit + escalate',
        'escalation': 'Transfer to senior retention specialist ASAP',
        'document': 'Flag account as VIP recovery - management follow-up required'
    }
    
    return guidance

def display_retention_insights(customer_name, customer_data, churn_prob):
    """Display comprehensive retention insights for agent."""
    profile = extract_customer_profile(customer_data)
    risk_factors = identify_risk_factors(profile, churn_prob)
    recommendations = generate_recommendations(profile, risk_factors, churn_prob)
    
    # NEW: Calculate win-back probability
    winback = calculate_winback_probability(profile, churn_prob)
    
    # NEW: Generate objection handlers
    objection_handlers = generate_objection_handlers(profile, churn_prob)
    
    # NEW: Generate conversation flow
    conversation_flow = generate_conversation_flow(profile, churn_prob, recommendations)
    
    # NEW: Determine next contact channel
    next_contact = determine_next_contact_channel(profile, churn_prob)
    
    # NEW: Generate sentiment guidance
    sentiment_guidance = generate_sentiment_guidance(profile, churn_prob)
    
    # Determine risk level
    if churn_prob > 0.7:
        risk_level = '🔴 CRITICAL'
        urgency = 'URGENT - Contact within 24 hours'
    elif churn_prob > 0.5:
        risk_level = '🟠 HIGH'
        urgency = 'HIGH - Contact within 48 hours'
    elif churn_prob > 0.3:
        risk_level = '🟡 MEDIUM'
        urgency = 'MODERATE - Contact within 1 week'
    else:
        risk_level = '🟢 LOW'
        urgency = 'ROUTINE - Standard service'
    
    print(f"\n{'='*70}")
    print(f"RETENTION INSIGHTS - {customer_name}")
    print(f"{'='*70}")
    
    # Customer Profile
    print(f"\n📋 CUSTOMER PROFILE:")
    print(f"   Tenure: {profile['tenure_months']} months")
    print(f"   Monthly Charges: ${profile['monthly_charges']:.2f}")
    print(f"   Total Charges: ${profile['total_charges']:.2f}")
    print(f"   Contract: {profile['contract_type']}")
    print(f"   Services: {profile['service_count']} active")
    print(f"   Internet: {profile['internet_service']}")
    print(f"   Senior: {'Yes' if profile['senior_citizen'] else 'No'}")
    print(f"   Add-ons: {'Yes' if profile['has_addons'] else 'No'}")
    
    # Churn Risk
    print(f"\n🎯 CHURN RISK ANALYSIS:")
    print(f"   Risk Level: {risk_level}")
    print(f"   Churn Probability: {churn_prob:.2%}")
    print(f"   Urgency: {urgency}")
    ltv = profile['monthly_charges'] * 36
    print(f"   Estimated LTV: ${ltv:.2f}")
    
    # NEW: Win-Back Probability
    print(f"\n🎲 WIN-BACK PROBABILITY:")
    print(f"   Retention Likelihood: {winback['probability']:.2%}")
    print(f"   Strategy: {winback['strategy']}")
    print(f"   Approach: {winback['description']}")
    if winback['strategy'] == 'AGGRESSIVE_SAVE':
        print(f"   💰 Authorization: Up to ${min(500, int(ltv * 0.15))} in incentives")
    elif winback['strategy'] == 'NEGOTIATION':
        print(f"   💰 Authorization: Up to ${min(300, int(ltv * 0.10))} in incentives")
    else:
        print(f"   💰 Authorization: Standard offers only (up to ${min(150, int(ltv * 0.05))})")
    
    # Risk Factors
    if risk_factors:
        print(f"\n⚠️  RISK FACTORS ({len(risk_factors)}):")
        for i, risk in enumerate(risk_factors, 1):
            print(f"   {i}. [{risk['severity']}] {risk['factor']}")
            print(f"      {risk['description']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDED ACTIONS ({len(recommendations)}):")
    if recommendations and 'method' in recommendations[0]:
        rec_method = recommendations[0]['method']
        if rec_method == 'RL-Based':
            print(f"   🤖 Using Reinforcement Learning (AI-Optimized)")
        else:
            print(f"   📋 Using Rule-Based System")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. [Priority {rec.get('priority', 'N/A')}] {rec['icon']} {rec['action']}")
        print(f"      {rec['description']}")
        if 'cost' in rec:
            print(f"      Cost: ${rec['cost']}")
        print(f"      Impact: {rec['expected_impact']}")
    
    # Primary Offer with Time Urgency
    if profile['senior_citizen'] and profile['monthly_charges'] > 90:
        primary_offer = f"Senior Bundle: ${profile['monthly_charges'] - 30:.2f}/month + Free tech support"
    elif profile['tenure_months'] < 6:
        primary_offer = "New Customer Special: 50% off 3 months + Free premium channels"
    elif profile['tenure_months'] > 24:
        discount = min(30, profile['tenure_months'] // 12 * 10)
        primary_offer = f"Loyalty Reward: ${discount}/month off + Equipment upgrade"
    elif profile['contract_type'] == 'Month-to-month':
        primary_offer = "Contract Conversion: $25/month off + 24-month Price Lock"
    else:
        primary_offer = f"Retention Offer: ${min(25, int(profile['monthly_charges'] * 0.2))}/month discount"
    
    print(f"\n📞 PRIMARY RETENTION OFFER:")
    print(f"   {primary_offer}")
    print(f"   ⏰ EXPIRES: 48 hours from contact")
    print(f"   📅 Follow-up: Schedule callback if customer needs time to decide")
    
    # NEW: Conversation Flow Playbook
    print(f"\n🗣️  CONVERSATION PLAYBOOK (Step-by-Step):")
    for step_key in ['step1', 'step2', 'step3', 'step4']:
        step = conversation_flow[step_key]
        print(f"\n   📍 {step['phase']} ({step['timing']})")
        print(f"      Goal: {step['objective']}")
        if 'script' in step:
            for idx, line in enumerate(step['script'], 1):
                print(f"      {idx}. \"{line}\"")
        if 'listen_for' in step:
            print(f"      👂 Listen for: {', '.join(step['listen_for'])}")
        if 'notes' in step:
            print(f"      {step['notes']}")
    
    # NEW: Objection Handling
    if objection_handlers:
        print(f"\n🛡️  OBJECTION HANDLING SCRIPTS ({len(objection_handlers)} scenarios):")
        for idx, handler in enumerate(objection_handlers, 1):
            print(f"\n   {idx}. {handler['objection']}")
            print(f"      Agent Says: \"{handler['response']}\"")
            print(f"      ➡️  Action: {handler['action']}")
            print(f"      � Fallback: {handler['fallback']}")
            print(f"      ⬆️  Escalation: {handler['escalation']}")
    
    # NEW: Next Contact Channel
    print(f"\n📱 NEXT-BEST-CONTACT CHANNEL:")
    print(f"   Recommended: {next_contact['primary']}")
    print(f"   Timing: {next_contact['timing']}")
    print(f"   Reason: {next_contact['reason']}")
    print(f"   Backup: {next_contact['backup']}")
    print(f"   Message: \"{next_contact['message']}\"")
    
    # NEW: Real-Time Sentiment Monitoring
    print(f"\n🎭 REAL-TIME SENTIMENT MONITORING:")
    print(f"   Keywords to Watch:")
    for keyword_group in sentiment_guidance['watch_for']:
        print(f"      • {keyword_group}")
    
    print(f"\n   If Customer is FRUSTRATED:")
    print(f"      🚨 {sentiment_guidance['if_frustrated']['immediate_action']}")
    print(f"      Say: \"{sentiment_guidance['if_frustrated']['script']}\"")
    
    print(f"\n   If Customer is INTERESTED:")
    print(f"      ✅ {sentiment_guidance['if_interested']['immediate_action']}")
    print(f"      Say: \"{sentiment_guidance['if_interested']['script']}\"")
    print(f"      Close: \"{sentiment_guidance['if_interested']['closing']}\"")
    
    print(f"\n   If Customer is ANGRY:")
    print(f"      🚨 {sentiment_guidance['if_angry']['immediate_action']}")
    print(f"      Say: \"{sentiment_guidance['if_angry']['script']}\"")
    print(f"      ⚠️  {sentiment_guidance['if_angry']['escalation']}")
    
    # Traditional Talking Points
    print(f"\n💬 ADDITIONAL TALKING POINTS:")
    print(f"   • Thank you for being a Comcast customer ({profile['tenure_months']} months)")
    if churn_prob > 0.5:
        print(f"   • I want to ensure you're getting the best value")
        print(f"   • I have exclusive offers designed for valued customers like you")
    if profile['senior_citizen']:
        print(f"   • As a senior, you qualify for special discounts and support")
    if profile['tenure_months'] > 24:
        print(f"   • Your loyalty means everything - let me show you our appreciation")
    print(f"   • What's most important to you: lower cost, more services, or better support?")
    
    print(f"\n{'='*70}\n")

def predict_churn(customer_data):
    """Predict churn probability for customer data."""
    # Add engineered features
    customer_data_enhanced = add_engineered_features(customer_data)
    
    # Scale the customer data
    customer_data_scaled = scaler.transform(customer_data_enhanced)
    customer_tensor = torch.tensor(customer_data_scaled, dtype=torch.float32)
    
    # Get prediction
    with torch.no_grad():
        output = model(customer_tensor)
        # Apply sigmoid if using BCEWithLogitsLoss
        if input_dim > 19:
            churn_prob = torch.sigmoid(output).item()
        else:
            churn_prob = output.item()
    
    return churn_prob

def load_dataset():
    """Load the customer dataset."""
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        return df
    except FileNotFoundError:
        print("Error: Dataset file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found!")
        return None

def analyze_single_customer():
    """Interactive mode to analyze a single customer."""
    print("\n" + "="*70)
    print("ANALYZE SINGLE CUSTOMER")
    print("="*70)
    
    df = load_dataset()
    if df is None:
        return
    
    print(f"\nTotal customers in database: {len(df)}")
    
    # Show sample customer IDs
    print(f"\nSample Customer IDs: {', '.join(df['customerID'].head(10).tolist())}")
    
    customer_id = input("\nEnter Customer ID (or 'back' to return): ").strip()
    if customer_id.lower() == 'back':
        return
    
    # Find customer
    customer = df[df['customerID'] == customer_id]
    
    if customer.empty:
        print(f"\n❌ Customer ID '{customer_id}' not found!")
        return
    
    # Prepare data for prediction (drop customerID and Churn if exists)
    customer_data = customer.drop(columns=['customerID', 'Churn'], errors='ignore')
    
    # CRITICAL: Convert TotalCharges to numeric BEFORE encoding loop
    customer_data['TotalCharges'] = pd.to_numeric(customer_data['TotalCharges'], errors='coerce')
    customer_data.fillna(0, inplace=True)
    
    # Load saved label encoders for consistency
    try:
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        use_saved_encoders = True
    except FileNotFoundError:
        label_encoders = {}
        use_saved_encoders = False
    
    # Encode categorical columns (same as training)
    from sklearn.preprocessing import LabelEncoder
    for col in customer_data.select_dtypes(include=['object']).columns:
        if use_saved_encoders and col in label_encoders:
            customer_data[col] = label_encoders[col].transform(customer_data[col])
        else:
            customer_data[col] = LabelEncoder().fit_transform(customer_data[col])
    
    # Predict
    churn_prob = predict_churn(customer_data)
    
    # Display insights
    display_retention_insights(f"Customer {customer_id}", customer_data, churn_prob)

def analyze_high_risk_customers():
    """Generate report of all high-risk customers."""
    print("\n" + "="*70)
    print("HIGH-RISK CUSTOMERS REPORT")
    print("="*70)
    
    df = load_dataset()
    if df is None:
        return
    
    # Preset threshold choices
    print("\n📊 Select risk threshold:")
    print("   1. 🔴 Critical Only (70%+) - Immediate action required")
    print("   2. 🟠 High Risk (60%+) - Proactive outreach recommended")
    print("   3. 🟡 Medium Risk (50%+) - Broader prevention strategy")
    print("   4. 🟢 All At-Risk (30%+) - Early intervention & campaigns")
    
    choice = input("\nSelect option (1-4, default=2): ").strip()
    
    threshold_map = {
        '1': 0.7,
        '2': 0.6,
        '3': 0.5,
        '4': 0.3,
        '': 0.6  # Default to option 2
    }
    
    threshold = threshold_map.get(choice, 0.6)
    
    risk_labels = {
        0.7: "🔴 CRITICAL (70%+)",
        0.6: "🟠 HIGH RISK (60%+)",
        0.5: "🟡 MEDIUM RISK (50%+)",
        0.3: "🟢 AT-RISK (30%+)"
    }
    
    print(f"\nAnalyzing {len(df)} customers for {risk_labels.get(threshold, 'risk')}... (this may take a moment)")
    
    # Prepare data
    df_original = df.copy()
    df_encoded = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    
    # CRITICAL: Convert TotalCharges to numeric BEFORE encoding loop
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    df_encoded.fillna(0, inplace=True)
    
    # Load saved label encoders for consistency
    try:
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        use_saved_encoders = True
    except FileNotFoundError:
        label_encoders = {}
        use_saved_encoders = False
    
    # Encode categorical columns
    from sklearn.preprocessing import LabelEncoder
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if use_saved_encoders and col in label_encoders:
            df_encoded[col] = label_encoders[col].transform(df_encoded[col])
        else:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    
    # Predict for all customers
    df_encoded_enhanced = add_engineered_features(df_encoded)
    scaled_data = scaler.transform(df_encoded_enhanced)
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(tensor_data)
        if input_dim > 19:
            churn_probs = torch.sigmoid(outputs).numpy().flatten()
        else:
            churn_probs = outputs.numpy().flatten()
    
    # Filter high-risk customers
    high_risk_mask = churn_probs >= threshold
    high_risk_customers = df_original[high_risk_mask].copy()
    high_risk_customers['ChurnProbability'] = churn_probs[high_risk_mask]
    high_risk_customers = high_risk_customers.sort_values('ChurnProbability', ascending=False)
    
    print(f"\n{risk_labels.get(threshold, '⚠️')} Found {len(high_risk_customers)} high-risk customers (>{threshold:.0%} churn probability)")
    
    if len(high_risk_customers) == 0:
        return
    
    # Display top 10
    print("\n" + "="*70)
    print("TOP 10 HIGHEST RISK CUSTOMERS")
    print("="*70)
    
    for idx, (_, row) in enumerate(high_risk_customers.head(10).iterrows(), 1):
        print(f"\n{idx}. Customer ID: {row['customerID']}")
        print(f"   Churn Probability: {row['ChurnProbability']:.2%}")
        print(f"   Tenure: {row['tenure']} months | Monthly: ${row['MonthlyCharges']:.2f}")
        print(f"   Contract: {row['Contract']} | Internet: {row['InternetService']}")
    
    # Ask to save report
    save = input("\nSave full report to CSV? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"high_risk_customers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        high_risk_customers.to_csv(filename, index=False)
        print(f"✅ Report saved to: {filename}")

def run_demo():
    """Run the demo with 3 test customers selected from the real dataset and encoded like training."""
    print("\n" + "="*70)
    print("DEMO MODE - 3 TEST CUSTOMERS")
    print("="*70)

    # Load and prepare dataset
    df = load_dataset()
    if df is None:
        print("\n❌ Cannot run demo without dataset.")
        return

    df_original = df.copy()
    df_enc = df.drop(columns=['customerID', 'Churn'], errors='ignore')

    # Fit LabelEncoders on the full dataset (same approach used elsewhere)
    encoders = {}
    for col in df_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder().fit(df_enc[col])
        df_enc[col] = le.transform(df_enc[col])
        encoders[col] = le

    # Ensure numeric conversions
    df_enc['TotalCharges'] = pd.to_numeric(df_enc['TotalCharges'], errors='coerce')
    df_enc.fillna(0, inplace=True)

    # Precompute predictions for all customers using consistent preprocessing
    df_enc_enh = add_engineered_features(df_enc)
    scaled_all = scaler.transform(df_enc_enh)
    tens_all = torch.tensor(scaled_all, dtype=torch.float32)
    with torch.no_grad():
        out_all = model(tens_all)
        probs_all = torch.sigmoid(out_all).numpy().flatten() if input_dim > 19 else out_all.numpy().flatten()

    # Helper to package a single index
    def predict_row(idx):
        row_df = df_enc.iloc[[idx]].copy()
        prob = float(probs_all[idx])
        return prob, row_df

    # Pick example customers based on original (non-encoded) features
    # 1) Low-risk: long tenure (>=24) and multiple add-ons
    low_mask = (
        (df_original['tenure'] >= 24) &
        (df_original['Contract'].isin(['One year', 'Two year'])) &
        (df_original['MonthlyCharges'] <= df_original['MonthlyCharges'].median())
    )
    # Prefer an actual low-probability customer (lowest predicted prob)
    low_idx = int(np.argmin(probs_all))
    prob_low, row_low = predict_row(low_idx)
    print("=" * 70)
    print("CUSTOMER 1: LOW-RISK (Loyal Customer)")
    print("=" * 70)
    print(f"Tenure: {int(df_original.iloc[low_idx]['tenure'])} months | Monthly Charges: ${float(df_original.iloc[low_idx]['MonthlyCharges']):.0f} | Total Charges: ${float(df_original.iloc[low_idx]['TotalCharges']):.0f}")
    print(f"Churn probability: {prob_low:.2%}")
    print(f"Status: {'🟢 LOW RISK' if prob_low <= 0.3 else '🟡 MEDIUM/HIGH'}\n")

    # 2) High-risk #1: senior, month-to-month, very new (<=3), high charges
    hr1_mask = (
        (df_original['SeniorCitizen'] == 1) &
        (df_original['Contract'] == 'Month-to-month') &
        (df_original['tenure'] <= 3) &
        (df_original['MonthlyCharges'] >= df_original['MonthlyCharges'].quantile(0.75))
    )
    # Choose highest predicted churn probability as high-risk #1
    hr1_idx = int(np.argmax(probs_all))
    prob_hr1, row_hr1 = predict_row(hr1_idx)
    risk_cat_1 = '🔴 CRITICAL RISK' if prob_hr1 > 0.7 else ('🟠 HIGH RISK' if prob_hr1 > 0.5 else ('🟡 MEDIUM RISK' if prob_hr1 > 0.3 else '🟢 LOW RISK'))
    print("=" * 70)
    print(f"CUSTOMER 2: {risk_cat_1} (Senior Citizen - New Customer)")
    print("=" * 70)
    print(f"Tenure: {int(df_original.iloc[hr1_idx]['tenure'])} months | Monthly Charges: ${float(df_original.iloc[hr1_idx]['MonthlyCharges']):.0f} | Total Charges: ${float(df_original.iloc[hr1_idx]['TotalCharges']):.0f}")
    print(f"Contract: {df_original.iloc[hr1_idx]['Contract']} | Internet: {df_original.iloc[hr1_idx]['InternetService']}")
    print(f"Senior Citizen: {'Yes' if int(df_original.iloc[hr1_idx]['SeniorCitizen'])==1 else 'No'}")
    print(f"Churn probability: {prob_hr1:.2%}\n")

    # 3) High-risk #2: month-to-month, new (<=3), internet only, no add-ons, high charges
    no_addons_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    hr2_mask = (
        (df_original['Contract'] == 'Month-to-month') &
        (df_original['tenure'] <= 3) &
        (df_original['InternetService'] != 'No') &
        (df_original['MonthlyCharges'] >= df_original['MonthlyCharges'].quantile(0.75))
    )
    for c in no_addons_cols:
        if c in df_original.columns:
            hr2_mask &= (df_original[c] == 'No')
    # Choose the second-highest predicted churn probability as high-risk #2
    # Ensure distinct from hr1_idx
    order = np.argsort(probs_all)
    hr2_idx = int(order[-2]) if int(order[-1]) == hr1_idx else int(order[-1])
    prob_hr2, row_hr2 = predict_row(hr2_idx)
    risk_cat_2 = '🔴 CRITICAL RISK' if prob_hr2 > 0.7 else ('🟠 HIGH RISK' if prob_hr2 > 0.5 else ('🟡 MEDIUM RISK' if prob_hr2 > 0.3 else '🟢 LOW RISK'))
    print("=" * 70)
    print(f"CUSTOMER 3: {risk_cat_2} (Month-to-Month, No Add-ons)")
    print("=" * 70)
    print(f"Tenure: {int(df_original.iloc[hr2_idx]['tenure'])} months | Monthly Charges: ${float(df_original.iloc[hr2_idx]['MonthlyCharges']):.0f} | Total Charges: ${float(df_original.iloc[hr2_idx]['TotalCharges']):.0f}")
    print(f"Contract: {df_original.iloc[hr2_idx]['Contract']} | Internet: {df_original.iloc[hr2_idx]['InternetService']}")
    print(f"Churn probability: {prob_hr2:.2%}\n")

    # Build list for insight rendering (use encoded rows for pipeline consistency)
    high_risk_customers = [
        {"name": "Customer 2", "data": row_hr1, "prob": prob_hr1},
        {"name": "Customer 3", "data": row_hr2, "prob": prob_hr2}
    ]

    print("\n" + "=" * 70)
    print("AI-POWERED RETENTION INSIGHTS SYSTEM")
    print("Generating Actionable Recommendations for Customer Service Agents")
    print("=" * 70)
    print("\nThis system uses AI to analyze customer data and provide:")
    print("✓ Churn risk assessment and urgency levels")
    print("✓ Specific risk factors identified from customer behavior")
    print("✓ Prioritized actionable recommendations with expected outcomes")
    print("✓ Pre-scripted talking points for agent conversations")
    print("✓ Primary retention offers tailored to each customer")

    # Process all high-risk customers
    for customer in high_risk_customers:
        display_retention_insights(customer["name"], customer["data"], customer["prob"])
        
        # Add a separator for readability
        if customer != high_risk_customers[-1]:
            print("\n" + "-" * 70)
            input("\nPress Enter to view next customer insights...")

    print("\n" + "=" * 70)
    print("RETENTION INSIGHTS SUMMARY")
    print("=" * 70)

    # Generate summary statistics
    critical_count = sum(1 for c in high_risk_customers if c["prob"] > 0.7)
    high_count = sum(1 for c in high_risk_customers if 0.5 < c["prob"] <= 0.7)
    total_revenue_at_risk = sum(
        extract_customer_profile(c["data"])['monthly_charges'] * 36 
        for c in high_risk_customers
    )

    print(f"\n📊 SUMMARY:")
    print(f"   Total Customers Analyzed: {len(high_risk_customers)}")
    print(f"   🔴 Critical Risk: {critical_count} customers")
    print(f"   🟠 High Risk: {high_count} customers")
    print(f"   💰 Total Revenue at Risk (3-year): ${total_revenue_at_risk:.2f}")
    print(f"   📞 Immediate Actions Required: {critical_count}")
    print(f"\n✅ All retention insights generated successfully!")
    print(f"   Agents can now contact customers with personalized offers.")

    print("\n" + "=" * 70)

def train_rl_recommendation_system():
    """Train the Reinforcement Learning recommendation system"""
    print("\n" + "="*70)
    print("TRAIN REINFORCEMENT LEARNING RECOMMENDATION SYSTEM")
    print("="*70)
    
    if not RL_AVAILABLE:
        print("\n❌ RL system not available. Please check rl_recommendation_system.py")
        return
    
    print("\n🤖 Initializing RL Agent Training...")
    print("This will train a Deep Q-Network to learn optimal retention strategies")
    print("Training uses simulated customer responses (in production, use real data)\n")
    
    # Ask user for training parameters
    try:
        episodes = input("Number of training episodes (default=1000, recommended 500-2000): ").strip()
        episodes = int(episodes) if episodes else 1000
    except:
        episodes = 1000
    
    print(f"\n📚 Training for {episodes} episodes...")
    print("This may take a few minutes...\n")
    
    # Initialize fresh agent
    state_dim = 8
    action_dim = len(RETENTION_ACTIONS)
    agent = DQNAgent(state_dim, action_dim)
    environment = RetentionEnvironment()
    
    # Train the agent
    try:
        episode_rewards, retention_rates = train_rl_agent(agent, environment, num_episodes=episodes)
        
        # Save the trained agent
        model_path = 'models/rl_agent.pth'
        agent.save(model_path)
        
        # Update global RL agent
        global rl_agent
        rl_agent = agent
        
        print("\n" + "="*70)
        print("✅ RL AGENT TRAINING COMPLETE")
        print("="*70)
        print(f"\n📊 Training Summary:")
        print(f"   Final Retention Rate: {np.mean(retention_rates[-100:]):.2%}")
        print(f"   Final Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"   Model saved to: {model_path}")
        print(f"\n💡 The system will now use RL-based recommendations automatically!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

def enhanced_metrics_evaluation():
    """Run enhanced recommendation evaluation with conversion tracking and relevance scoring"""
    try:
        from enhanced_recommendation_metrics import evaluate_with_enhanced_metrics
        evaluate_with_enhanced_metrics()
    except Exception as e:
        print(f"\n❌ Error running enhanced metrics: {str(e)}")
        import traceback
        traceback.print_exc()

def main_menu():
    """Interactive main menu for the retention system."""
    while True:
        print("\n" + "="*70)
        print("CUSTOMER CHURN PREDICTION & RETENTION SYSTEM")
        print("="*70)
        print("\n📋 MAIN MENU:")
        print("   1. 📊 Churn Prediction Model - Evaluation Metrics")
        print("   2. 🎯 Evaluate Recommendation System Quality")
        print("   3. � Enhanced Metrics (Conversion + Relevance) ⭐ NEW")
        print("   4. �🔍 Analyze Single Customer (by ID)")
        print("   5. 📈 Generate High-Risk Customer Report")
        print("   6. 🎬 Run Demo (3 Test Customers)")
        print("   7. 🤖 Train RL Recommendation System (Advanced)")
        print("   8. 🚪 Exit")
        
        choice = input("\nSelect an option (1-8): ").strip()
        
        if choice == '1':
            evaluate_model_performance()
        elif choice == '2':
            evaluate_recommendations_quality()
        elif choice == '3':
            enhanced_metrics_evaluation()
        elif choice == '4':
            analyze_single_customer()
        elif choice == '5':
            analyze_high_risk_customers()
        elif choice == '6':
            run_demo()
        elif choice == '7':
            train_rl_recommendation_system()
        elif choice == '8':
            print("\n✅ Thank you for using the Retention System. Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please select 1-8.")

# Check if running in interactive mode or demo mode
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # Run demo mode directly
        run_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == '--menu':
        # Run interactive menu
        main_menu()
    else:
        # Default: run demo then show menu
        print("\n" + "="*70)
        print("Running in DEMO MODE (showing 3 test customers)")
        print("Use '--menu' argument to skip demo and go straight to menu")
        print("="*70)
        input("\nPress Enter to continue...")
        run_demo()
        
        # Ask if user wants to continue to interactive menu
        print("\n" + "="*70)
        continue_menu = input("\nWould you like to access the interactive menu? (y/n): ").strip().lower()
        if continue_menu == 'y':
            main_menu()
        else:
            print("\n✅ Thank you for using the Retention System. Goodbye!")

