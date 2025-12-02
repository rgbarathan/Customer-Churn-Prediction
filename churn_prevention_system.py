"""
Customer Churn Prevention & Retention Recommendation System

This system analyzes customer profiles and provides actionable insights
and next-best-action recommendations for customer service representatives
to prevent churn and improve retention.
"""

import torch
import pandas as pd
from transformers import pipeline
from churn_prediction import ChurnModel
import os
import pickle
import sys

class ChurnPreventionSystem:
    """
    AI-powered system that analyzes customer churn risk and generates
    personalized retention recommendations for CSR agents.
    """
    
    def __init__(self):
        """Initialize churn model and recommendation engine."""
        # Load churn prediction model
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.model = ChurnModel(input_dim=19)
        self.model.load_state_dict(torch.load('models/churn_model.pth'))
        self.model.eval()
        
        # Initialize NLP model for generating personalized messages
        self.text_generator = pipeline('text-generation', 
                                      model='gpt2',
                                      max_length=100,
                                      device=0 if torch.cuda.is_available() else -1)
        
        print("✓ Loaded churn prediction model")
        print("✓ Initialized retention recommendation engine")
    
    def analyze_customer(self, customer_data):
        """
        Analyze customer and provide comprehensive retention insights.
        
        Args:
            customer_data: DataFrame with customer features
            
        Returns:
            Dict with churn risk, risk factors, and recommendations
        """
        # Predict churn probability
        customer_scaled = self.scaler.transform(customer_data)
        customer_tensor = torch.tensor(customer_scaled, dtype=torch.float32)
        churn_prob = self.model(customer_tensor).item()
        
        # Extract customer profile
        profile = self._extract_profile(customer_data)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(profile, churn_prob)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(profile, risk_factors, churn_prob)
        
        # Create retention strategy
        strategy = self._create_retention_strategy(profile, churn_prob, recommendations)
        
        return {
            'customer_profile': profile,
            'churn_probability': churn_prob,
            'risk_level': self._get_risk_level(churn_prob),
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'retention_strategy': strategy,
            'priority_score': self._calculate_priority(churn_prob, profile),
            'estimated_ltv': self._estimate_lifetime_value(profile),
            'retention_success_rate': self._estimate_retention_success(profile, churn_prob)
        }
    
    def _extract_profile(self, customer_data):
        """Extract customer profile from raw data."""
        row = customer_data.iloc[0]
        
        # Map encoded values back to meaningful labels
        contract_types = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
        internet_types = {0: 'No', 1: 'DSL', 2: 'Fiber optic'}
        payment_types = {0: 'Bank transfer', 1: 'Credit card', 2: 'Electronic check', 
                        3: 'Mailed check', 4: 'Other', 5: 'Unknown'}
        
        profile = {
            'gender': 'Male' if row['gender'] == 1 else 'Female',
            'senior_citizen': bool(row['SeniorCitizen']),
            'has_partner': bool(row['Partner']),
            'has_dependents': bool(row['Dependents']),
            'tenure_months': int(row['tenure']),
            'has_phone': bool(row['PhoneService']),
            'has_multiple_lines': bool(row['MultipleLines']),
            'internet_service': internet_types.get(int(row['InternetService']), 'Unknown'),
            'has_online_security': bool(row['OnlineSecurity']),
            'has_online_backup': bool(row['OnlineBackup']),
            'has_device_protection': bool(row['DeviceProtection']),
            'has_tech_support': bool(row['TechSupport']),
            'has_streaming_tv': bool(row['StreamingTV']),
            'has_streaming_movies': bool(row['StreamingMovies']),
            'contract_type': contract_types.get(int(row['Contract']), 'Unknown'),
            'paperless_billing': bool(row['PaperlessBilling']),
            'payment_method': payment_types.get(int(row['PaymentMethod']), 'Unknown'),
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
            profile['has_tech_support'],
            profile['has_streaming_tv'],
            profile['has_streaming_movies']
        ])
        
        profile['has_addons'] = any([
            profile['has_online_security'],
            profile['has_online_backup'],
            profile['has_device_protection'],
            profile['has_tech_support']
        ])
        
        profile['average_monthly_cost'] = profile['total_charges'] / max(profile['tenure_months'], 1)
        
        return profile
    
    def _identify_risk_factors(self, profile, churn_prob):
        """Identify specific factors contributing to churn risk."""
        risk_factors = []
        
        # Tenure-based risks
        if profile['tenure_months'] < 3:
            risk_factors.append({
                'factor': 'Very Short Tenure',
                'severity': 'CRITICAL',
                'description': f"Customer has only been with us for {profile['tenure_months']} month(s). Early-stage churn is highest.",
                'impact': 'HIGH'
            })
        elif profile['tenure_months'] < 12:
            risk_factors.append({
                'factor': 'Short Tenure',
                'severity': 'HIGH',
                'description': f"Customer has been with us for {profile['tenure_months']} months. Still in high-risk period.",
                'impact': 'MEDIUM'
            })
        
        # Pricing risks
        if profile['monthly_charges'] > 90:
            risk_factors.append({
                'factor': 'High Monthly Charges',
                'severity': 'HIGH',
                'description': f"Monthly charges of ${profile['monthly_charges']:.2f} may be causing price sensitivity.",
                'impact': 'HIGH'
            })
        
        # Contract risks
        if profile['contract_type'] == 'Month-to-month':
            risk_factors.append({
                'factor': 'No Contract Commitment',
                'severity': 'HIGH',
                'description': "Month-to-month contract makes it easy to leave without penalties.",
                'impact': 'HIGH'
            })
        
        # Service engagement risks
        if profile['service_count'] <= 1:
            risk_factors.append({
                'factor': 'Low Service Engagement',
                'severity': 'MEDIUM',
                'description': f"Only using {profile['service_count']} service(s). Low engagement means lower switching costs.",
                'impact': 'MEDIUM'
            })
        
        if not profile['has_addons']:
            risk_factors.append({
                'factor': 'No Value-Added Services',
                'severity': 'MEDIUM',
                'description': "No security, backup, device protection, or tech support. Easy to replicate elsewhere.",
                'impact': 'MEDIUM'
            })
        
        # Senior citizen risks
        if profile['senior_citizen']:
            risk_factors.append({
                'factor': 'Senior Citizen',
                'severity': 'MEDIUM',
                'description': "Senior citizens may be more price-sensitive and less tech-savvy.",
                'impact': 'MEDIUM'
            })
        
        # Premium service risks
        if profile['internet_service'] == 'Fiber optic' and profile['tenure_months'] < 6:
            risk_factors.append({
                'factor': 'New Premium Service Customer',
                'severity': 'HIGH',
                'description': "Fiber optic customer with short tenure - high expectations may not be met.",
                'impact': 'HIGH'
            })
        
        # Payment method risks
        if profile['payment_method'] in ['Electronic check', 'Mailed check']:
            risk_factors.append({
                'factor': 'Manual Payment Method',
                'severity': 'LOW',
                'description': "Manual payments indicate lower commitment compared to auto-pay.",
                'impact': 'LOW'
            })
        
        return risk_factors
    
    def _generate_recommendations(self, profile, risk_factors, churn_prob):
        """Generate specific, actionable recommendations for CSR."""
        recommendations = []
        
        # Priority 1: Immediate Actions (Critical Risk)
        if churn_prob > 0.6:
            recommendations.append({
                'priority': 1,
                'category': 'URGENT ACTION',
                'action': '🚨 Immediate Outreach Required',
                'description': 'Contact customer within 24 hours with retention offer',
                'talking_points': [
                    'Express appreciation for their business',
                    'Ask about service satisfaction and pain points',
                    'Present exclusive retention offer'
                ],
                'success_rate': '65%'
            })
        
        # Address specific risk factors
        for risk in risk_factors:
            if risk['factor'] == 'High Monthly Charges':
                recommendations.append({
                    'priority': 2 if churn_prob > 0.5 else 3,
                    'category': 'PRICING',
                    'action': f"💰 Reduce Monthly Cost (Current: ${profile['monthly_charges']:.2f})",
                    'description': 'Offer bundle discount or loyalty pricing',
                    'specific_offers': [
                        f"Bundle discount: Save $20-30/month by adding services",
                        f"Loyalty discount: $15/month off for 12-month commitment",
                        f"Senior discount: $15/month (if applicable)" if profile['senior_citizen'] else None
                    ],
                    'expected_savings': '$20-45/month',
                    'success_rate': '72%'
                })
            
            if risk['factor'] == 'No Contract Commitment':
                recommendations.append({
                    'priority': 2,
                    'category': 'CONTRACT',
                    'action': '📝 Convert to Fixed-Term Contract',
                    'description': 'Offer incentive for 12 or 24-month contract',
                    'specific_offers': [
                        '12-month contract: $15/month discount + free equipment upgrade',
                        '24-month contract: $25/month discount + price lock guarantee',
                        'Waive any setup fees or equipment charges'
                    ],
                    'expected_savings': '$15-25/month',
                    'success_rate': '58%'
                })
            
            if risk['factor'] in ['Low Service Engagement', 'No Value-Added Services']:
                recommendations.append({
                    'priority': 3,
                    'category': 'UPSELL',
                    'action': f"📦 Increase Service Bundle (Current: {profile['service_count']} service(s))",
                    'description': 'Add complementary services to increase stickiness',
                    'specific_offers': [
                        'Free premium channels for 6 months (HBO, Showtime)',
                        'Free security suite and backup service (3 months)',
                        'Free tech support package trial',
                        'Streaming device at no cost'
                    ],
                    'expected_value': 'Increase LTV by $500-1000',
                    'success_rate': '45%'
                })
            
            if risk['factor'] in ['Very Short Tenure', 'Short Tenure']:
                recommendations.append({
                    'priority': 1 if profile['tenure_months'] < 3 else 2,
                    'category': 'ONBOARDING',
                    'action': f"🆕 Enhanced Onboarding (Tenure: {profile['tenure_months']} months)",
                    'description': 'Provide white-glove service during critical early period',
                    'specific_offers': [
                        'Schedule personal check-in call',
                        'Free in-home tech support visit',
                        '50% off next 3 months (buyer\'s remorse protection)',
                        'Assign dedicated account manager'
                    ],
                    'expected_impact': 'Reduce early churn by 40%',
                    'success_rate': '68%'
                })
            
            if risk['factor'] == 'Senior Citizen':
                recommendations.append({
                    'priority': 2,
                    'category': 'DEMOGRAPHIC',
                    'action': '👴 Senior Citizen Program Enrollment',
                    'description': 'Apply senior-specific benefits and support',
                    'specific_offers': [
                        'Senior Advantage: $15/month discount (automatic)',
                        'Free 24/7 tech support hotline',
                        'Priority customer service (no wait times)',
                        'Large-print bills and simplified interface'
                    ],
                    'expected_savings': '$15-25/month',
                    'success_rate': '75%'
                })
        
        # Long-tenure customers (loyalty rewards)
        if profile['tenure_months'] > 24:
            recommendations.append({
                'priority': 2,
                'category': 'LOYALTY',
                'action': f"⭐ Loyalty Rewards (Tenure: {profile['tenure_months']} months)",
                'description': 'Recognize and reward long-term customer',
                'specific_offers': [
                    f"Loyalty discount: ${min(30, profile['tenure_months'] // 12 * 10)}/month",
                    'Free equipment upgrade (latest modem/router)',
                    'Complimentary premium channel package',
                    'No-fee service upgrades for life'
                ],
                'expected_value': 'Increase retention by 35%',
                'success_rate': '82%'
            })
        
        # Fiber customers (premium service)
        if profile['internet_service'] == 'Fiber optic':
            recommendations.append({
                'priority': 2,
                'category': 'PREMIUM',
                'action': '🚀 Premium Customer Experience',
                'description': 'Ensure premium service expectations are met',
                'specific_offers': [
                    'Dedicated premium support hotline',
                    'Priority service appointments',
                    'Free speed upgrade to next tier',
                    'Quarterly service quality check-ins'
                ],
                'expected_impact': 'Premium customer retention',
                'success_rate': '70%'
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations
    
    def _create_retention_strategy(self, profile, churn_prob, recommendations):
        """Create comprehensive retention strategy with timeline and actions."""
        strategy = {
            'urgency_level': self._get_urgency(churn_prob),
            'action_timeline': self._get_action_timeline(churn_prob),
            'primary_offer': self._get_primary_offer(profile, churn_prob),
            'talking_points': self._get_talking_points(profile, churn_prob),
            'escalation_path': self._get_escalation_path(churn_prob),
            'follow_up_plan': self._get_follow_up_plan(profile, churn_prob),
            'success_metrics': {
                'target_retention_rate': self._estimate_retention_success(profile, churn_prob),
                'expected_ltv_increase': f"${self._estimate_lifetime_value(profile) * 0.25:.2f}",
                'roi_of_intervention': '3.5x'
            }
        }
        
        return strategy
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level."""
        if probability > 0.7:
            return '🔴 CRITICAL'
        elif probability > 0.5:
            return '🟠 HIGH'
        elif probability > 0.3:
            return '🟡 MEDIUM'
        else:
            return '🟢 LOW'
    
    def _get_urgency(self, probability):
        """Determine urgency level."""
        if probability > 0.7:
            return 'URGENT - Act within 24 hours'
        elif probability > 0.5:
            return 'HIGH - Act within 48 hours'
        elif probability > 0.3:
            return 'MODERATE - Act within 1 week'
        else:
            return 'ROUTINE - Monthly check-in'
    
    def _get_action_timeline(self, probability):
        """Get specific action timeline."""
        if probability > 0.7:
            return {
                'immediate': 'Call customer today',
                '24_hours': 'Send retention offer via email',
                '48_hours': 'Follow-up call if no response',
                '72_hours': 'Escalate to retention manager'
            }
        elif probability > 0.5:
            return {
                'day_1': 'Email personalized retention offer',
                'day_2': 'Call customer with offer details',
                'week_1': 'Follow-up and finalize offer',
                'week_2': 'Check satisfaction after offer applied'
            }
        else:
            return {
                'week_1': 'Send customer appreciation message',
                'month_1': 'Proactive check-in call',
                'quarter_1': 'Review account for optimization'
            }
    
    def _get_primary_offer(self, profile, probability):
        """Determine best primary offer based on profile."""
        if profile['senior_citizen'] and profile['monthly_charges'] > 90:
            return f"Senior Bundle Special: Reduce bill to ${profile['monthly_charges'] - 30:.2f}/month + free tech support"
        elif profile['tenure_months'] < 6:
            return "New Customer Retention: 50% off next 3 months + free premium channels"
        elif profile['tenure_months'] > 24:
            return f"Loyalty Appreciation: ${min(30, profile['tenure_months'] // 12 * 10)}/month off + equipment upgrade"
        elif profile['contract_type'] == 'Month-to-month':
            return "Contract Conversion: $25/month off with 24-month commitment + price lock"
        elif profile['service_count'] <= 1:
            return "Bundle Savings: Add 2+ services and save $40/month"
        else:
            return f"Personalized Retention Offer: ${min(25, int(profile['monthly_charges'] * 0.2))}/month discount"
    
    def _get_talking_points(self, profile, probability):
        """Generate CSR talking points."""
        points = [
            "Thank you for being a valued Comcast customer",
            f"I see you've been with us for {profile['tenure_months']} months"
        ]
        
        if probability > 0.5:
            points.append("I want to make sure you're getting the best value from your service")
            points.append("I have some exclusive offers that might interest you")
        
        if profile['senior_citizen']:
            points.append("As a senior customer, you qualify for special discounts")
        
        if profile['tenure_months'] > 24:
            points.append("Your loyalty means everything to us - let me show you how we can reward that")
        
        points.append("What matters most to you: lowering your bill, adding services, or improving service quality?")
        
        return points
    
    def _get_escalation_path(self, probability):
        """Determine escalation path."""
        if probability > 0.7:
            return 'Escalate to Senior Retention Specialist immediately'
        elif probability > 0.5:
            return 'Route to Retention Team within 24 hours'
        elif probability > 0.3:
            return 'Standard retention process'
        else:
            return 'No escalation needed'
    
    def _get_follow_up_plan(self, profile, probability):
        """Create follow-up schedule."""
        if probability > 0.6:
            return {
                'initial_followup': '3 days after offer accepted',
                'second_followup': '2 weeks after acceptance',
                'ongoing': 'Monthly check-ins for 6 months'
            }
        elif probability > 0.3:
            return {
                'initial_followup': '1 week after offer',
                'second_followup': '1 month after acceptance',
                'ongoing': 'Quarterly check-ins'
            }
        else:
            return {
                'initial_followup': 'Not required',
                'ongoing': 'Annual satisfaction survey'
            }
    
    def _calculate_priority(self, churn_prob, profile):
        """Calculate intervention priority score (0-100)."""
        base_score = churn_prob * 100
        
        # Adjust for customer value
        if profile['monthly_charges'] > 100:
            base_score += 10
        if profile['tenure_months'] > 24:
            base_score += 5
        if profile['service_count'] >= 3:
            base_score += 5
        
        return min(100, int(base_score))
    
    def _estimate_lifetime_value(self, profile):
        """Estimate customer lifetime value."""
        average_lifetime_months = 36  # Industry average
        monthly_value = profile['monthly_charges']
        return monthly_value * average_lifetime_months
    
    def _estimate_retention_success(self, profile, churn_prob):
        """Estimate probability of successful retention."""
        base_success = 1 - churn_prob
        
        # Adjust based on tenure (longer tenure = easier to retain)
        if profile['tenure_months'] > 24:
            base_success += 0.1
        elif profile['tenure_months'] < 3:
            base_success -= 0.1
        
        # Adjust based on engagement
        if profile['service_count'] >= 3:
            base_success += 0.05
        
        return f"{min(95, base_success * 100):.1f}%"


def format_customer_insights(analysis):
    """Format analysis results for display."""
    print("\n" + "=" * 80)
    print("CUSTOMER RETENTION INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)
    
    profile = analysis['customer_profile']
    
    # Customer Overview
    print("\n📋 CUSTOMER PROFILE")
    print("-" * 80)
    print(f"Tenure: {profile['tenure_months']} months")
    print(f"Monthly Charges: ${profile['monthly_charges']:.2f}")
    print(f"Total Charges: ${profile['total_charges']:.2f}")
    print(f"Contract: {profile['contract_type']}")
    print(f"Services: {profile['service_count']} active service(s)")
    print(f"Internet: {profile['internet_service']}")
    print(f"Senior Citizen: {'Yes' if profile['senior_citizen'] else 'No'}")
    print(f"Add-ons: {'Yes' if profile['has_addons'] else 'No (Security, Backup, etc.)'}")
    
    # Churn Risk
    print(f"\n🎯 CHURN RISK ANALYSIS")
    print("-" * 80)
    print(f"Risk Level: {analysis['risk_level']}")
    print(f"Churn Probability: {analysis['churn_probability']:.2%}")
    print(f"Priority Score: {analysis['priority_score']}/100")
    print(f"Estimated LTV: ${analysis['estimated_ltv']:.2f}")
    print(f"Retention Success Rate: {analysis['retention_success_rate']}")
    
    # Risk Factors
    print(f"\n⚠️  IDENTIFIED RISK FACTORS")
    print("-" * 80)
    for i, risk in enumerate(analysis['risk_factors'], 1):
        print(f"\n{i}. [{risk['severity']}] {risk['factor']}")
        print(f"   {risk['description']}")
        print(f"   Impact: {risk['impact']}")
    
    # Recommendations
    print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
    print("-" * 80)
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"\n{i}. [Priority {rec['priority']}] {rec['action']}")
        print(f"   Category: {rec['category']}")
        print(f"   {rec['description']}")
        if 'specific_offers' in rec:
            print(f"   Offers:")
            for offer in rec['specific_offers']:
                if offer:
                    print(f"      • {offer}")
        if 'expected_savings' in rec:
            print(f"   Expected Savings: {rec['expected_savings']}")
        if 'success_rate' in rec:
            print(f"   Success Rate: {rec['success_rate']}")
    
    # Retention Strategy
    print(f"\n📞 RETENTION STRATEGY")
    print("-" * 80)
    strategy = analysis['retention_strategy']
    print(f"Urgency: {strategy['urgency_level']}")
    print(f"Primary Offer: {strategy['primary_offer']}")
    print(f"Escalation: {strategy['escalation_path']}")
    
    print(f"\n🗓️  Action Timeline:")
    for timeframe, action in strategy['action_timeline'].items():
        print(f"   • {timeframe.replace('_', ' ').title()}: {action}")
    
    print(f"\n💬 Talking Points for CSR:")
    for point in strategy['talking_points']:
        print(f"   • {point}")
    
    print(f"\n📊 Success Metrics:")
    for metric, value in strategy['success_metrics'].items():
        print(f"   • {metric.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 80)


# Main execution
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    print("=" * 80)
    print("CUSTOMER CHURN PREVENTION & RETENTION RECOMMENDATION SYSTEM")
    print("=" * 80)
    
    # Initialize system
    system = ChurnPreventionSystem()
    
    # Test customers
    print("\n\nAnalyzing high-risk customers...")
    
    # Customer 1: Senior citizen with high charges, new customer
    print("\n" + "🔴" * 40)
    print("CUSTOMER 1: Senior Citizen - New Customer - High Charges")
    print("🔴" * 40)
    customer1 = pd.DataFrame([[1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 105.0, 210.0]],
                             columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
                                    'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                                    'Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])
    
    analysis1 = system.analyze_customer(customer1)
    format_customer_insights(analysis1)
    
    # Customer 2: New fiber customer with highest charges
    print("\n" + "🔴" * 40)
    print("CUSTOMER 2: New Premium Fiber Customer - Buyer's Remorse Risk")
    print("🔴" * 40)
    customer2 = pd.DataFrame([[0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 115.0, 115.0]],
                             columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
                                    'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                                    'Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])
    
    analysis2 = system.analyze_customer(customer2)
    format_customer_insights(analysis2)
    
    print("\n" + "=" * 80)
    print("✓ Analysis Complete - Recommendations Ready for Agent Action")
    print("=" * 80)
