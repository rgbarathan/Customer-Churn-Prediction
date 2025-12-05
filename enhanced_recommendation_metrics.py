"""
Enhanced Recommendation Metrics: Conversion Tracking & Relevance Scoring

This module adds two critical missing metrics:
1. Conversion Tracking - Simulates/tracks actual retention outcomes
2. Relevance Scoring - Measures how well recommendations match risk factors

These metrics provide deeper insights into recommendation quality beyond
basic coverage and diversity metrics.
"""

import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import json
import os

try:
    from churn_prediction_enhanced import ChurnModel, advanced_feature_engineering
    ENHANCED_MODEL = True
except ImportError:
    ENHANCED_MODEL = False


# ============================================================================
# CONVERSION TRACKING SYSTEM
# ============================================================================

class ConversionTracker:
    """
    Tracks simulated/actual retention outcomes for recommendations.
    
    In production, this would track:
    - Which customers were contacted
    - Which recommendations were offered
    - Whether customer was retained
    - Cost of retention effort
    - Revenue saved/lost
    """
    
    def __init__(self, tracking_file='models/conversion_tracking.json'):
        self.tracking_file = tracking_file
        self.conversions = self.load_tracking_data()
    
    def load_tracking_data(self):
        """Load existing conversion tracking data"""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {
            'campaigns': [],
            'individual_outcomes': [],
            'summary_stats': {
                'total_contacted': 0,
                'total_retained': 0,
                'total_churned': 0,
                'total_cost': 0,
                'total_revenue_saved': 0
            }
        }
    
    def save_tracking_data(self):
        """Save conversion tracking data"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.conversions, f, indent=2)
    
    def simulate_conversion(self, customer_id, churn_prob, recommendations, 
                          customer_profile, actual_churned=None):
        """
        Simulate conversion outcome based on recommendation quality.
        
        In production, replace with actual outcome data.
        """
        # Base retention probability (inversely related to churn probability)
        base_retention_prob = 1 - churn_prob
        
        # Calculate recommendation quality boost
        quality_boost = self._calculate_quality_boost(recommendations, customer_profile, churn_prob)
        
        # Final retention probability
        final_retention_prob = min(0.95, base_retention_prob + quality_boost)
        
        # Simulate outcome (in production, use actual_churned parameter)
        if actual_churned is not None:
            retained = not actual_churned
        else:
            retained = np.random.random() < final_retention_prob
        
        # Calculate metrics
        rec_costs = [rec.get('cost', 200) for rec in recommendations]
        total_cost = sum(rec_costs[:1])  # Assume only top recommendation used
        
        ltv = customer_profile.get('monthly_charges', 65) * 36
        revenue_saved = ltv if retained else 0
        
        # Record outcome
        outcome = {
            'customer_id': customer_id,
            'timestamp': datetime.now().isoformat(),
            'churn_probability': float(churn_prob),
            'predicted_retention_prob': float(final_retention_prob),
            'recommendations_offered': len(recommendations),
            'top_recommendation': recommendations[0]['action'] if recommendations else None,
            'cost': total_cost,
            'retained': retained,
            'revenue_saved': revenue_saved,
            'roi': ((revenue_saved - total_cost) / total_cost * 100) if total_cost > 0 else 0
        }
        
        self.conversions['individual_outcomes'].append(outcome)
        
        # Update summary
        self.conversions['summary_stats']['total_contacted'] += 1
        if retained:
            self.conversions['summary_stats']['total_retained'] += 1
        else:
            self.conversions['summary_stats']['total_churned'] += 1
        self.conversions['summary_stats']['total_cost'] += total_cost
        self.conversions['summary_stats']['total_revenue_saved'] += revenue_saved
        
        return outcome
    
    def _calculate_quality_boost(self, recommendations, customer_profile, churn_prob):
        """Calculate retention probability boost from recommendation quality"""
        if not recommendations:
            return 0.0
        
        boost = 0.0
        
        # Factor 1: Relevance to customer profile
        if customer_profile.get('high_charges', False) and \
           any('discount' in rec['action'].lower() or 'price' in rec['action'].lower() 
               for rec in recommendations):
            boost += 0.10
        
        if customer_profile.get('month_to_month', False) and \
           any('contract' in rec['action'].lower() for rec in recommendations):
            boost += 0.12
        
        if customer_profile.get('senior_citizen', False) and \
           any('senior' in rec['action'].lower() for rec in recommendations):
            boost += 0.08
        
        # Factor 2: Recommendation diversity
        unique_types = len(set(rec.get('type', 'unknown') for rec in recommendations))
        if unique_types >= 3:
            boost += 0.05
        
        # Factor 3: Higher risk needs more aggressive offers
        if churn_prob > 0.6 and any(rec.get('priority', 3) == 1 for rec in recommendations):
            boost += 0.08
        
        return min(boost, 0.30)  # Cap at 30% boost
    
    def get_conversion_metrics(self):
        """Calculate aggregate conversion metrics"""
        if not self.conversions['individual_outcomes']:
            return None
        
        outcomes = self.conversions['individual_outcomes']
        
        # Overall conversion rate
        conversion_rate = sum(1 for o in outcomes if o['retained']) / len(outcomes)
        
        # Average cost per retention
        retained_outcomes = [o for o in outcomes if o['retained']]
        avg_cost_per_retention = (
            sum(o['cost'] for o in retained_outcomes) / len(retained_outcomes)
            if retained_outcomes else 0
        )
        
        # ROI
        total_cost = sum(o['cost'] for o in outcomes)
        total_revenue = sum(o['revenue_saved'] for o in outcomes)
        roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        # Prediction accuracy
        prediction_errors = [
            abs(o['predicted_retention_prob'] - (1 if o['retained'] else 0))
            for o in outcomes
        ]
        prediction_accuracy = 1 - np.mean(prediction_errors)
        
        return {
            'total_customers': len(outcomes),
            'conversion_rate': conversion_rate,
            'customers_retained': sum(1 for o in outcomes if o['retained']),
            'customers_lost': sum(1 for o in outcomes if not o['retained']),
            'total_cost': total_cost,
            'total_revenue_saved': total_revenue,
            'net_benefit': total_revenue - total_cost,
            'roi': roi,
            'avg_cost_per_retention': avg_cost_per_retention,
            'prediction_accuracy': prediction_accuracy
        }


# ============================================================================
# RELEVANCE SCORING SYSTEM
# ============================================================================

class RelevanceScorer:
    """
    Scores how well recommendations match identified risk factors.
    
    Higher relevance = recommendations directly address customer's specific issues
    """
    
    # Mapping of risk factors to relevant recommendation types
    RISK_TO_REC_MAPPING = {
        'High Monthly Charges': ['discount', 'price', 'reduce', 'savings', 'cost'],
        'No Contract Commitment': ['contract', 'commitment', 'lock', 'year'],
        'Month-to-month': ['contract', 'commitment', 'lock', 'year'],
        'No Value-Added Services': ['service', 'bundle', 'add-on', 'protection', 'security'],
        'Electronic Check Payment': ['payment', 'automatic', 'autopay', 'reliable'],
        'Senior Citizen': ['senior', 'assistance', 'support', 'discount'],
        'Short Tenure': ['welcome', 'onboarding', 'loyalty', 'commitment'],
        'Fiber Optic': ['upgrade', 'optimize', 'speed', 'performance'],
        'No Support': ['support', 'assistance', 'help', 'service'],
        'Multiple Services': ['bundle', 'package', 'discount', 'combined'],
        'No Phone Service': ['bundle', 'package', 'phone', 'add'],
        'No Internet Service': ['internet', 'bundle', 'add', 'connectivity'],
        'Paperless Billing': ['convenience', 'eco', 'discount'],
        'No Online Security': ['security', 'protection', 'safety', 'backup'],
        'No Tech Support': ['support', 'assistance', 'help', 'technical'],
        'New Customer': ['welcome', 'onboarding', 'start', 'new'],
        'High Value Customer': ['premium', 'vip', 'exclusive', 'priority'],
        'Price Sensitive': ['discount', 'savings', 'reduce', 'lower']
    }
    
    def calculate_relevance_score(self, recommendations, risk_factors):
        """
        Calculate overall relevance score for recommendations.
        
        Returns score from 0.0 (not relevant) to 1.0 (perfectly relevant)
        """
        if not recommendations or not risk_factors:
            return 0.0
        
        relevance_scores = []
        
        for rec in recommendations:
            rec_score = self._score_single_recommendation(rec, risk_factors)
            relevance_scores.append(rec_score)
        
        # Overall relevance is average of individual scores
        return np.mean(relevance_scores)
    
    def _score_single_recommendation(self, recommendation, risk_factors):
        """Score a single recommendation against all risk factors"""
        rec_action = recommendation.get('action', '').lower()
        rec_description = recommendation.get('description', '').lower()
        rec_text = f"{rec_action} {rec_description}"
        
        # Check if recommendation addresses any risk factor
        addresses_risk = False
        max_match_score = 0.0
        
        for risk in risk_factors:
            risk_factor = risk.get('factor', '')
            match_score = self._calculate_match_score(rec_text, risk_factor)
            
            if match_score > 0:
                addresses_risk = True
                max_match_score = max(max_match_score, match_score)
        
        # Score components
        direct_address_score = max_match_score  # 0.0 to 1.0
        specificity_score = self._calculate_specificity(recommendation)  # 0.0 to 1.0
        
        # Weighted combination
        relevance = (0.7 * direct_address_score + 0.3 * specificity_score)
        
        return relevance
    
    def _calculate_match_score(self, rec_text, risk_factor):
        """Calculate how well recommendation matches specific risk factor"""
        if risk_factor not in self.RISK_TO_REC_MAPPING:
            # Generic match - check if risk keywords appear in recommendation
            risk_keywords = risk_factor.lower().split()
            matches = sum(1 for keyword in risk_keywords if keyword in rec_text)
            return min(matches / len(risk_keywords), 1.0)
        
        # Use mapping
        relevant_keywords = self.RISK_TO_REC_MAPPING[risk_factor]
        matches = sum(1 for keyword in relevant_keywords if keyword in rec_text)
        
        if matches == 0:
            return 0.0
        elif matches == 1:
            return 0.5
        elif matches == 2:
            return 0.8
        else:
            return 1.0
    
    def _calculate_specificity(self, recommendation):
        """
        Score how specific/actionable the recommendation is.
        
        Specific recommendations include:
        - Exact dollar amounts
        - Specific time frames
        - Clear actions
        - Expected outcomes
        """
        score = 0.0
        rec_text = f"{recommendation.get('action', '')} {recommendation.get('description', '')}"
        
        # Has dollar amount
        if '$' in rec_text or 'dollar' in rec_text.lower():
            score += 0.25
        
        # Has percentage
        if '%' in rec_text or 'percent' in rec_text.lower():
            score += 0.25
        
        # Has time frame
        time_keywords = ['month', 'year', 'week', 'day', 'hours']
        if any(keyword in rec_text.lower() for keyword in time_keywords):
            score += 0.25
        
        # Has expected outcome
        if recommendation.get('expected_impact') or 'expected' in rec_text.lower():
            score += 0.25
        
        return score
    
    def get_detailed_relevance_analysis(self, recommendations, risk_factors):
        """Get detailed breakdown of relevance scores"""
        analysis = {
            'overall_score': self.calculate_relevance_score(recommendations, risk_factors),
            'recommendation_scores': [],
            'unaddressed_risks': [],
            'well_addressed_risks': []
        }
        
        # Score each recommendation
        for i, rec in enumerate(recommendations):
            rec_score = self._score_single_recommendation(rec, risk_factors)
            analysis['recommendation_scores'].append({
                'recommendation': rec.get('action', 'Unknown'),
                'relevance_score': rec_score,
                'priority': rec.get('priority', 'Unknown')
            })
        
        # Identify unaddressed risks
        for risk in risk_factors:
            risk_factor = risk.get('factor', '')
            addressed = False
            
            for rec in recommendations:
                rec_text = f"{rec.get('action', '')} {rec.get('description', '')}".lower()
                if self._calculate_match_score(rec_text, risk_factor) > 0.3:
                    addressed = True
                    analysis['well_addressed_risks'].append({
                        'risk': risk_factor,
                        'severity': risk.get('severity', 'Unknown')
                    })
                    break
            
            if not addressed:
                analysis['unaddressed_risks'].append({
                    'risk': risk_factor,
                    'severity': risk.get('severity', 'Unknown')
                })
        
        return analysis


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_with_enhanced_metrics():
    """Run enhanced recommendation evaluation with conversion and relevance tracking"""
    print("\n" + "="*80)
    print("ENHANCED RECOMMENDATION EVALUATION")
    print("Conversion Tracking & Relevance Scoring")
    print("="*80)
    
    # Import necessary functions from main
    import sys
    sys.path.append('.')
    from main import (load_dataset, model, scaler, temperature, decision_threshold,
                      add_engineered_features, extract_customer_profile,
                      identify_risk_factors, generate_recommendations)
    
    # Initialize trackers
    conversion_tracker = ConversionTracker()
    relevance_scorer = RelevanceScorer()
    
    # Load data
    print("\n📂 Loading customer data...")
    df = load_dataset()
    if df is None:
        return
    
    # Prepare data
    df_encoded = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    df_encoded.fillna(0, inplace=True)
    
    # Load label encoders
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            df_encoded[col] = label_encoders[col].transform(df_encoded[col])
    
    # Get predictions
    df_enhanced = advanced_feature_engineering(df_encoded)
    scaled_data = scaler.transform(df_enhanced)
    
    with torch.no_grad():
        outputs = model(torch.tensor(scaled_data, dtype=torch.float32))
        churn_probs = torch.sigmoid(outputs / temperature).numpy().flatten()
    
    # Get actual churn labels
    y_true = LabelEncoder().fit_transform(df['Churn'])
    
    # Focus on high-risk customers
    high_risk_mask = churn_probs >= decision_threshold
    high_risk_indices = np.where(high_risk_mask)[0]
    
    print(f"✓ Found {len(high_risk_indices)} high-risk customers")
    
    # Sample for analysis
    sample_size = min(100, len(high_risk_indices))
    sample_indices = np.random.choice(high_risk_indices, sample_size, replace=False)
    
    print(f"✓ Analyzing {sample_size} customers for detailed metrics...\n")
    
    # Collect metrics
    relevance_scores = []
    conversion_outcomes = []
    detailed_analyses = []
    
    for idx in sample_indices:
        customer_id = df.iloc[idx]['customerID']
        customer_data = df_encoded.iloc[[idx]]
        churn_prob = churn_probs[idx]
        actual_churned = bool(y_true[idx])
        
        # Get profile and recommendations
        profile = extract_customer_profile(customer_data)
        risk_factors = identify_risk_factors(profile, churn_prob)
        recommendations = generate_recommendations(profile, risk_factors, churn_prob)
        
        # Calculate relevance
        relevance_score = relevance_scorer.calculate_relevance_score(recommendations, risk_factors)
        relevance_scores.append(relevance_score)
        
        # Simulate/track conversion
        outcome = conversion_tracker.simulate_conversion(
            customer_id, churn_prob, recommendations, profile, actual_churned
        )
        conversion_outcomes.append(outcome)
        
        # Get detailed analysis for sample
        if len(detailed_analyses) < 5:
            analysis = relevance_scorer.get_detailed_relevance_analysis(recommendations, risk_factors)
            detailed_analyses.append({
                'customer_id': customer_id,
                'churn_prob': churn_prob,
                'analysis': analysis
            })
    
    # Save tracking data
    conversion_tracker.save_tracking_data()
    
    # Display results
    print("="*80)
    print("📊 ENHANCED METRICS RESULTS")
    print("="*80)
    
    # Relevance Metrics
    print("\n🎯 RELEVANCE SCORING:")
    print(f"   Average Relevance Score:           {np.mean(relevance_scores):.2%}")
    print(f"   Min Relevance:                     {np.min(relevance_scores):.2%}")
    print(f"   Max Relevance:                     {np.max(relevance_scores):.2%}")
    
    high_relevance = sum(1 for s in relevance_scores if s >= 0.7)
    medium_relevance = sum(1 for s in relevance_scores if 0.4 <= s < 0.7)
    low_relevance = sum(1 for s in relevance_scores if s < 0.4)
    
    print(f"\n   High Relevance (≥70%):             {high_relevance}/{sample_size} ({high_relevance/sample_size*100:.1f}%)")
    print(f"   Medium Relevance (40-70%):         {medium_relevance}/{sample_size} ({medium_relevance/sample_size*100:.1f}%)")
    print(f"   Low Relevance (<40%):              {low_relevance}/{sample_size} ({low_relevance/sample_size*100:.1f}%)")
    
    # Conversion Metrics
    conv_metrics = conversion_tracker.get_conversion_metrics()
    
    print("\n💪 CONVERSION TRACKING:")
    print(f"   Conversion Rate (Retained):        {conv_metrics['conversion_rate']:.2%}")
    print(f"   Customers Retained:                {conv_metrics['customers_retained']}/{conv_metrics['total_customers']}")
    print(f"   Customers Lost:                    {conv_metrics['customers_lost']}/{conv_metrics['total_customers']}")
    print(f"   Prediction Accuracy:               {conv_metrics['prediction_accuracy']:.2%}")
    
    print(f"\n💰 FINANCIAL IMPACT:")
    print(f"   Total Investment:                  ${conv_metrics['total_cost']:,.0f}")
    print(f"   Total Revenue Saved:               ${conv_metrics['total_revenue_saved']:,.0f}")
    print(f"   Net Benefit:                       ${conv_metrics['net_benefit']:,.0f}")
    print(f"   ROI:                               {conv_metrics['roi']:.1f}%")
    print(f"   Avg Cost per Retention:            ${conv_metrics['avg_cost_per_retention']:,.0f}")
    
    # Detailed examples
    print("\n" + "="*80)
    print("📋 DETAILED RELEVANCE ANALYSIS (Sample)")
    print("="*80)
    
    for detail in detailed_analyses[:3]:
        print(f"\nCustomer: {detail['customer_id']} (Churn Prob: {detail['churn_prob']:.1%})")
        analysis = detail['analysis']
        print(f"   Overall Relevance: {analysis['overall_score']:.2%}")
        
        if analysis['well_addressed_risks']:
            print(f"   ✅ Well Addressed Risks:")
            for risk in analysis['well_addressed_risks'][:3]:
                print(f"      - {risk['risk']} [{risk['severity']}]")
        
        if analysis['unaddressed_risks']:
            print(f"   ⚠️  Unaddressed Risks:")
            for risk in analysis['unaddressed_risks'][:3]:
                print(f"      - {risk['risk']} [{risk['severity']}]")
        
        print(f"   Recommendations:")
        for rec in analysis['recommendation_scores'][:3]:
            print(f"      {rec['priority']}. {rec['recommendation']} (Relevance: {rec['relevance_score']:.2%})")
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS & RECOMMENDATIONS")
    print("="*80)
    
    avg_relevance = np.mean(relevance_scores)
    conv_rate = conv_metrics['conversion_rate']
    
    print(f"\n📊 Overall Assessment:")
    if avg_relevance >= 0.7 and conv_rate >= 0.6:
        print(f"   ✅ EXCELLENT - High relevance and strong conversion")
    elif avg_relevance >= 0.5 and conv_rate >= 0.4:
        print(f"   ✓ GOOD - Solid performance with room for improvement")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT - Focus on relevance and conversion")
    
    print(f"\n🎯 Recommendations:")
    if avg_relevance < 0.6:
        print(f"   1. Improve recommendation relevance - currently {avg_relevance:.1%}")
        print(f"      → Better match recommendations to specific risk factors")
    
    if conv_rate < 0.5:
        print(f"   2. Boost conversion rate - currently {conv_rate:.1%}")
        print(f"      → Offer more aggressive incentives for high-risk customers")
    
    if conv_metrics['avg_cost_per_retention'] > 500:
        print(f"   3. Reduce cost per retention - currently ${conv_metrics['avg_cost_per_retention']:.0f}")
        print(f"      → Optimize offer amounts and targeting")
    
    print(f"\n✅ Tracking data saved to: {conversion_tracker.tracking_file}")
    print(f"✅ Enhanced metrics evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    evaluate_with_enhanced_metrics()
