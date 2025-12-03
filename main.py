
import torch
import pandas as pd
from churn_prediction import ChurnModel
import os
import pickle
import sys

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

# Load churn model
model = ChurnModel(input_dim=input_dim)
model.load_state_dict(torch.load('models/churn_model.pth'))
model.eval()

print("✓ Loaded churn prediction model")
print(f"✓ Model expects {input_dim} input features")
print("✓ Initialized retention recommendation engine")

def add_engineered_features(customer_data):
    """Add engineered features to customer data."""
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
    contract_types = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
    internet_types = {0: 'No', 1: 'DSL', 2: 'Fiber optic'}
    
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
    recommendations = []
    
    # Critical risk - immediate action
    if churn_prob > 0.6:
        recommendations.append({
            'priority': 1,
            'icon': '🚨',
            'action': 'Immediate Outreach Required',
            'description': 'Contact within 24 hours with exclusive retention offer',
            'expected_impact': '65% retention success rate'
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
                'expected_impact': f'Save ${savings}/month - 72% success rate'
            })
        
        if risk['factor'] == 'No Contract Commitment':
            recommendations.append({
                'priority': 2,
                'icon': '📝',
                'action': 'Convert to Long-Term Contract',
                'description': '24-month: $25/month off + Price Lock Guarantee',
                'expected_impact': '58% conversion rate'
            })
        
        if risk['factor'] in ['Very Short Tenure', 'Short Tenure']:
            recommendations.append({
                'priority': 1 if profile['tenure_months'] < 3 else 2,
                'icon': '🆕',
                'action': f"New Customer Retention (Tenure: {profile['tenure_months']} mo)",
                'description': '50% off next 3 months + Free premium channels',
                'expected_impact': '68% retention success'
            })
        
        if risk['factor'] == 'Senior Citizen':
            recommendations.append({
                'priority': 2,
                'icon': '👴',
                'action': 'Senior Advantage Program',
                'description': '$15/month discount + Free tech support',
                'expected_impact': '75% enrollment success'
            })
        
        if risk['factor'] in ['Low Service Engagement', 'No Value-Added Services']:
            recommendations.append({
                'priority': 3,
                'icon': '📦',
                'action': f"Increase Service Bundle (Current: {profile['service_count']})",
                'description': 'Free add-ons for 6 months (Security, Backup, Streaming)',
                'expected_impact': 'Increase LTV by $500-1000'
            })
    
    # Loyalty rewards for long-term customers
    if profile['tenure_months'] > 24:
        discount = min(30, profile['tenure_months'] // 12 * 10)
        recommendations.append({
            'priority': 2,
            'icon': '⭐',
            'action': f"Loyalty Rewards ({profile['tenure_months']} months tenure)",
            'description': f'${discount}/month discount + Equipment upgrade',
            'expected_impact': '82% retention success'
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
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. [Priority {rec['priority']}] {rec['icon']} {rec['action']}")
        print(f"      {rec['description']}")
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
    
    # Encode categorical columns (same as training)
    from sklearn.preprocessing import LabelEncoder
    for col in customer_data.select_dtypes(include=['object']).columns:
        customer_data[col] = LabelEncoder().fit_transform(customer_data[col])
    
    # Handle TotalCharges
    customer_data['TotalCharges'] = pd.to_numeric(customer_data['TotalCharges'], errors='coerce')
    customer_data.fillna(0, inplace=True)
    
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
    
    # Encode categorical columns
    from sklearn.preprocessing import LabelEncoder
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    df_encoded.fillna(0, inplace=True)
    
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
    """Run the demo with 5 test customers."""
    print("\n" + "="*70)
    print("DEMO MODE - 5 TEST CUSTOMERS")
    print("="*70)

    # Example combined workflow
    # Low-risk customer (loyal, long tenure)
    customer_data_low = pd.DataFrame([[1, 0, 29, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 70.0, 150.0]],
                              columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])

    # Scale the customer data using the same scaler
    customer_data_low_enhanced = add_engineered_features(customer_data_low)
    customer_data_scaled = scaler.transform(customer_data_low_enhanced)
    customer_tensor = torch.tensor(customer_data_scaled, dtype=torch.float32)
    with torch.no_grad():
        output = model(customer_tensor)
        churn_prob_low = torch.sigmoid(output).item() if input_dim > 19 else output.item()
    print("=" * 70)
    print("CUSTOMER 1: LOW-RISK (Loyal Customer)")
    print("=" * 70)
    print(f"Tenure: 29 months | Monthly Charges: $70 | Total Charges: $150")
    print(f"Services: Phone, Internet, Security, Backup, Tech Support")
    print(f"Churn probability: {churn_prob_low:.2%}")
    print(f"Status: ✅ RETAIN - Low risk customer\n")

    # High-risk customer #1 (new, high charges, minimal services)
    customer_data_high1 = pd.DataFrame([[1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 105.0, 210.0]],
                              columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])

    customer_data_scaled_high1 = scaler.transform(add_engineered_features(customer_data_high1))
    customer_tensor_high1 = torch.tensor(customer_data_scaled_high1, dtype=torch.float32)
    with torch.no_grad():
        output = model(customer_tensor_high1)
        churn_prob_high1 = torch.sigmoid(output).item() if input_dim > 19 else output.item()
    print("=" * 70)
    print("CUSTOMER 2: HIGH-RISK #1 (Senior Citizen - New Customer)")
    print("=" * 70)
    print(f"Tenure: 2 months | Monthly Charges: $105 | Total Charges: $210")
    print(f"Services: Internet only (no phone, security, backup, tech support)")
    print(f"Senior Citizen: Yes | High monthly charges")
    print(f"Churn probability: {churn_prob_high1:.2%}")
    print(f"Status: ⚠️  AT RISK - Needs intervention\n")

    # High-risk customer #2 (month-to-month, high charges, no extras)
    customer_data_high2 = pd.DataFrame([[0, 0, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 95.0, 285.0]],
                              columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])

    customer_data_scaled_high2 = scaler.transform(add_engineered_features(customer_data_high2))
    customer_tensor_high2 = torch.tensor(customer_data_scaled_high2, dtype=torch.float32)
    with torch.no_grad():
        output = model(customer_tensor_high2)
        churn_prob_high2 = torch.sigmoid(output).item() if input_dim > 19 else output.item()
    print("=" * 70)
    print("CUSTOMER 3: HIGH-RISK #2 (Month-to-Month Contract)")
    print("=" * 70)
    print(f"Tenure: 3 months | Monthly Charges: $95 | Total Charges: $285")
    print(f"Services: Internet only | Month-to-month contract (no commitment)")
    print(f"Contract Type: Month-to-Month (high flexibility to leave)")
    print(f"Churn probability: {churn_prob_high2:.2%}")
    print(f"Status: ⚠️  AT RISK - Easy to switch providers\n")

    # High-risk customer #3 (no internet service extras, paperless billing)
    customer_data_high3 = pd.DataFrame([[1, 0, 5, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 88.0, 440.0]],
                              columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])

    customer_data_scaled_high3 = scaler.transform(add_engineered_features(customer_data_high3))
    customer_tensor_high3 = torch.tensor(customer_data_scaled_high3, dtype=torch.float32)
    with torch.no_grad():
        output = model(customer_tensor_high3)
        churn_prob_high3 = torch.sigmoid(output).item() if input_dim > 19 else output.item()
    print("=" * 70)
    print("CUSTOMER 4: HIGH-RISK #3 (Low Engagement - No Add-ons)")
    print("=" * 70)
    print(f"Tenure: 5 months | Monthly Charges: $88 | Total Charges: $440")
    print(f"Services: Internet + Phone only (no security, backup, tech support)")
    print(f"No additional services despite 5 months tenure")
    print(f"Churn probability: {churn_prob_high3:.2%}")
    print(f"Status: ⚠️  AT RISK - Low engagement\n")

    # High-risk customer #4 (fiber optic, high charges, minimal tenure)
    customer_data_high4 = pd.DataFrame([[0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 115.0, 115.0]],
                              columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])

    customer_data_scaled_high4 = scaler.transform(add_engineered_features(customer_data_high4))
    customer_tensor_high4 = torch.tensor(customer_data_scaled_high4, dtype=torch.float32)
    with torch.no_grad():
        output = model(customer_tensor_high4)
        churn_prob_high4 = torch.sigmoid(output).item() if input_dim > 19 else output.item()
    print("=" * 70)
    print("CUSTOMER 5: HIGH-RISK #4 (Very New - Highest Charges)")
    print("=" * 70)
    print(f"Tenure: 1 month | Monthly Charges: $115 | Total Charges: $115")
    print(f"Services: Phone + Fiber Internet (premium service)")
    print(f"Fiber optic (premium but may be dissatisfied with speed/quality)")
    print(f"Churn probability: {churn_prob_high4:.2%}")
    print(f"Status: ⚠️  CRITICAL RISK - Brand new customer with high bill\n")

    # RETENTION INSIGHTS SYSTEM - AI-Powered Recommendations for Agents
    high_risk_customers = [
        {"name": "Customer 2", "data": customer_data_high1, "prob": churn_prob_high1},
        {"name": "Customer 3", "data": customer_data_high2, "prob": churn_prob_high2},
        {"name": "Customer 4", "data": customer_data_high3, "prob": churn_prob_high3},
        {"name": "Customer 5", "data": customer_data_high4, "prob": churn_prob_high4}
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

def main_menu():
    """Interactive main menu for the retention system."""
    while True:
        print("\n" + "="*70)
        print("CUSTOMER CHURN PREDICTION & RETENTION SYSTEM")
        print("="*70)
        print("\n📋 MAIN MENU:")
        print("   1. Analyze Single Customer (by ID)")
        print("   2. Generate High-Risk Customer Report")
        print("   3. Run Demo (5 Test Customers)")
        print("   4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            analyze_single_customer()
        elif choice == '2':
            analyze_high_risk_customers()
        elif choice == '3':
            run_demo()
        elif choice == '4':
            print("\n✅ Thank you for using the Retention System. Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please select 1-4.")

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
        print("Running in DEMO MODE (showing 5 test customers)")
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

