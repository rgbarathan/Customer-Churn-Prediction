"""
Threshold Testing & Optimization Tool

This script tests multiple decision thresholds and provides detailed
comparisons to help you choose the best one for your business needs.

It will evaluate thresholds from 0.35 to 0.70 and show:
- Precision, Recall, F1-Score for each
- Business impact (costs, savings, profit)
- Visual comparison chart
- Recommendation based on your priorities
"""

import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score)
import json

try:
    from churn_prediction_enhanced import ChurnModel, advanced_feature_engineering
    ENHANCED_MODEL = True
except ImportError:
    ENHANCED_MODEL = False

# Business constants
FP_COST = 200          # Cost of false positive (wasted retention offer)
FN_COST = 2331         # Cost of false negative (lost customer LTV)
TP_BENEFIT = 1166      # Benefit of true positive (50% retention success)
RETENTION_SUCCESS = 0.5  # Assume 50% of retention efforts succeed


def load_model_and_data():
    """Load trained model and dataset"""
    print("\n📂 Loading model and data...")
    
    # Load model
    model = ChurnModel(input_dim=36, hidden_dims=[128, 64, 32])
    model.load_state_dict(torch.load('models/churn_model.pth'))
    model.eval()
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load data
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Encode target
    y = LabelEncoder().fit_transform(df['Churn'])
    
    # Prepare features
    df_encoded = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    df_encoded.fillna(0, inplace=True)
    
    # Load and apply label encoders
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            df_encoded[col] = label_encoders[col].transform(df_encoded[col])
    
    # Add engineered features
    df_enhanced = advanced_feature_engineering(df_encoded)
    
    # Get predictions on full dataset
    scaled_data = scaler.transform(df_enhanced)
    
    with torch.no_grad():
        outputs = model(torch.tensor(scaled_data, dtype=torch.float32))
        probabilities = torch.sigmoid(outputs).numpy().flatten()
    
    print(f"✓ Loaded {len(df)} customers")
    print(f"✓ Model predictions generated")
    
    return probabilities, y, df


def calculate_metrics(y_true, y_pred, threshold):
    """Calculate all metrics for a given threshold"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Business metrics
    fp_cost_total = fp * FP_COST
    fn_cost_total = fn * FN_COST
    tp_benefit_total = tp * TP_BENEFIT
    
    total_cost = fp_cost_total
    total_benefit = tp_benefit_total
    net_profit = total_benefit - total_cost
    roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
    
    # Revenue at risk
    revenue_at_risk = fn * FN_COST
    revenue_saved = tp * FN_COST * RETENTION_SUCCESS
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'fp_cost': fp_cost_total,
        'fn_cost': fn_cost_total,
        'tp_benefit': tp_benefit_total,
        'net_profit': net_profit,
        'roi': roi,
        'revenue_at_risk': revenue_at_risk,
        'revenue_saved': revenue_saved,
        'customers_flagged': tp + fp,
        'flagged_rate': (tp + fp) / len(y_true) * 100
    }


def test_thresholds(probabilities, y_true, thresholds):
    """Test multiple thresholds and return results"""
    results = []
    
    print("\n🔄 Testing thresholds...")
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, threshold)
        results.append(metrics)
    
    return pd.DataFrame(results)


def print_comparison_table(df_results):
    """Print formatted comparison table"""
    print("\n" + "="*120)
    print("THRESHOLD COMPARISON TABLE")
    print("="*120)
    
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} "
          f"{'Flagged':<10} {'FP':<8} {'FN':<8} {'Net Profit':<15} {'ROI':<10}")
    print("-" * 120)
    
    for _, row in df_results.iterrows():
        threshold_str = f"{row['threshold']:.2f}"
        precision_str = f"{row['precision']:.1%}"
        recall_str = f"{row['recall']:.1%}"
        f1_str = f"{row['f1_score']:.1%}"
        flagged_str = f"{row['customers_flagged']}"
        fp_str = f"{row['fp']}"
        fn_str = f"{row['fn']}"
        profit_str = f"${row['net_profit']:,.0f}"
        roi_str = f"{row['roi']:.1f}%"
        
        print(f"{threshold_str:<12} {precision_str:<12} {recall_str:<12} {f1_str:<12} "
              f"{flagged_str:<10} {fp_str:<8} {fn_str:<8} {profit_str:<15} {roi_str:<10}")


def print_detailed_analysis(df_results, current_threshold=0.55):
    """Print detailed analysis and recommendations"""
    print("\n" + "="*120)
    print("DETAILED ANALYSIS")
    print("="*120)
    
    # Find best performers
    best_precision = df_results.loc[df_results['precision'].idxmax()]
    best_recall = df_results.loc[df_results['recall'].idxmax()]
    best_f1 = df_results.loc[df_results['f1_score'].idxmax()]
    best_profit = df_results.loc[df_results['net_profit'].idxmax()]
    best_roi = df_results.loc[df_results['roi'].idxmax()]
    
    # Current threshold
    current = df_results[df_results['threshold'] == current_threshold].iloc[0] if current_threshold in df_results['threshold'].values else None
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"\n1. Best Precision: {best_precision['threshold']:.2f}")
    print(f"   Precision: {best_precision['precision']:.1%}, Recall: {best_precision['recall']:.1%}, F1: {best_precision['f1_score']:.1%}")
    print(f"   False Positives: {best_precision['fp']}, Missed Churners: {best_precision['fn']}")
    print(f"   Net Profit: ${best_precision['net_profit']:,.0f}, ROI: {best_precision['roi']:.1f}%")
    
    print(f"\n2. Best Recall: {best_recall['threshold']:.2f}")
    print(f"   Precision: {best_recall['precision']:.1%}, Recall: {best_recall['recall']:.1%}, F1: {best_recall['f1_score']:.1%}")
    print(f"   False Positives: {best_recall['fp']}, Missed Churners: {best_recall['fn']}")
    print(f"   Net Profit: ${best_recall['net_profit']:,.0f}, ROI: {best_recall['roi']:.1f}%")
    
    print(f"\n3. Best F1-Score (Balanced): {best_f1['threshold']:.2f}")
    print(f"   Precision: {best_f1['precision']:.1%}, Recall: {best_f1['recall']:.1%}, F1: {best_f1['f1_score']:.1%}")
    print(f"   False Positives: {best_f1['fp']}, Missed Churners: {best_f1['fn']}")
    print(f"   Net Profit: ${best_f1['net_profit']:,.0f}, ROI: {best_f1['roi']:.1f}%")
    
    print(f"\n4. Best Net Profit: {best_profit['threshold']:.2f}")
    print(f"   Precision: {best_profit['precision']:.1%}, Recall: {best_profit['recall']:.1%}, F1: {best_profit['f1_score']:.1%}")
    print(f"   False Positives: {best_profit['fp']}, Missed Churners: {best_profit['fn']}")
    print(f"   Net Profit: ${best_profit['net_profit']:,.0f}, ROI: {best_profit['roi']:.1f}%")
    
    print(f"\n5. Best ROI: {best_roi['threshold']:.2f}")
    print(f"   Precision: {best_roi['precision']:.1%}, Recall: {best_roi['recall']:.1%}, F1: {best_roi['f1_score']:.1%}")
    print(f"   False Positives: {best_roi['fp']}, Missed Churners: {best_roi['fn']}")
    print(f"   Net Profit: ${best_roi['net_profit']:,.0f}, ROI: {best_roi['roi']:.1f}%")
    
    if current is not None:
        print(f"\n📍 CURRENT THRESHOLD: {current_threshold:.2f}")
        print(f"   Precision: {current['precision']:.1%}, Recall: {current['recall']:.1%}, F1: {current['f1_score']:.1%}")
        print(f"   False Positives: {current['fp']}, Missed Churners: {current['fn']}")
        print(f"   Net Profit: ${current['net_profit']:,.0f}, ROI: {current['roi']:.1f}%")
        print(f"   Customers Flagged: {current['customers_flagged']} ({current['flagged_rate']:.1f}% of total)")


def print_recommendations(df_results):
    """Provide threshold recommendations based on different priorities"""
    print("\n" + "="*120)
    print("RECOMMENDATIONS BY PRIORITY")
    print("="*120)
    
    # Scenario-based recommendations
    scenarios = [
        {
            'name': '🎯 Maximize Accuracy (Precision)',
            'description': 'Best when you want highly reliable predictions',
            'filter': lambda df: df[df['precision'] >= 0.65],
            'sort_by': 'precision',
            'use_case': 'Limited budget, avoid false alarms, build customer trust'
        },
        {
            'name': '🌐 Catch Most Churners (Recall)',
            'description': 'Best when you must identify as many churners as possible',
            'filter': lambda df: df[df['recall'] >= 0.70],
            'sort_by': 'recall',
            'use_case': 'Unlimited budget, churn extremely costly, cast wide net'
        },
        {
            'name': '⚖️ Balanced Approach (F1-Score)',
            'description': 'Best compromise between precision and recall',
            'filter': lambda df: df[(df['precision'] >= 0.55) & (df['recall'] >= 0.55)],
            'sort_by': 'f1_score',
            'use_case': 'Most situations, good all-around performance'
        },
        {
            'name': '💰 Maximize Profit',
            'description': 'Best financial outcome considering all costs',
            'filter': lambda df: df,
            'sort_by': 'net_profit',
            'use_case': 'Pure business optimization, revenue focus'
        },
        {
            'name': '📊 Best ROI',
            'description': 'Highest return on retention investment',
            'filter': lambda df: df[df['customers_flagged'] > 500],  # Minimum volume
            'sort_by': 'roi',
            'use_case': 'Efficiency focus, maximize value per dollar spent'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Use Case: {scenario['use_case']}")
        
        filtered = scenario['filter'](df_results)
        if not filtered.empty:
            best = filtered.nlargest(1, scenario['sort_by']).iloc[0]
            print(f"   → Recommended Threshold: {best['threshold']:.2f}")
            print(f"      Precision: {best['precision']:.1%}, Recall: {best['recall']:.1%}, F1: {best['f1_score']:.1%}")
            print(f"      Customers Flagged: {best['customers_flagged']}, False Positives: {best['fp']}, Missed: {best['fn']}")
            print(f"      Net Profit: ${best['net_profit']:,.0f}, ROI: {best['roi']:.1f}%")
        else:
            print(f"   → No threshold meets criteria")


def plot_ascii_chart(df_results):
    """Create ASCII chart of precision vs recall"""
    print("\n" + "="*120)
    print("PRECISION vs RECALL TRADE-OFF")
    print("="*120)
    print("\n" + " " * 15 + "Precision (%)")
    print(" " * 10 + "0    10   20   30   40   50   60   70   80   90   100")
    print(" " * 10 + "|----|----|----|----|----|----|----|----|----|----|")
    
    for _, row in df_results.iterrows():
        threshold = row['threshold']
        precision = row['precision'] * 100
        recall = row['recall'] * 100
        
        # Create bar
        precision_pos = int(precision / 2)  # Scale to 50 chars
        marker = f"{threshold:.2f}"
        
        # Position marker
        line = " " * 10 + "|"
        line += " " * (precision_pos - 1) + "●"
        line += " " * (50 - precision_pos)
        line += f"| {recall:5.1f}% recall"
        
        print(f"{marker:>8}: {line}")
    
    print(" " * 10 + "|----|----|----|----|----|----|----|----|----|----|")
    print(" " * 10 + "0    10   20   30   40   50   60   70   80   90   100")
    print("\n" + " " * 15 + "● = Threshold position (based on precision)")
    print(" " * 15 + "Number on right = Recall percentage")


def save_threshold(threshold, description="Custom threshold"):
    """Save selected threshold to config file"""
    config = {
        "threshold": float(threshold),
        "description": description,
        "set_date": "2025-12-05",
        "set_by": "Threshold Testing Tool"
    }
    
    with open('models/decision_threshold.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Threshold {threshold:.2f} saved to models/decision_threshold.json")


def interactive_selection(df_results):
    """Allow user to interactively select threshold"""
    print("\n" + "="*120)
    print("THRESHOLD SELECTION")
    print("="*120)
    
    print("\n🎯 Choose your priority:\n")
    print("   1. 🎯 Maximum Precision (avoid false alarms)")
    print("   2. 🌐 Maximum Recall (catch most churners)")
    print("   3. ⚖️ Balanced F1-Score (best compromise)")
    print("   4. 💰 Maximum Profit (best financial outcome)")
    print("   5. 📊 Maximum ROI (best efficiency)")
    print("   6. 🔢 Custom Threshold (enter your own)")
    print("   7. ❌ Don't change (keep current)")
    
    try:
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            best = df_results.loc[df_results['precision'].idxmax()]
            threshold = best['threshold']
            save_threshold(threshold, "Maximum Precision")
            return threshold
        elif choice == '2':
            best = df_results.loc[df_results['recall'].idxmax()]
            threshold = best['threshold']
            save_threshold(threshold, "Maximum Recall")
            return threshold
        elif choice == '3':
            best = df_results.loc[df_results['f1_score'].idxmax()]
            threshold = best['threshold']
            save_threshold(threshold, "Balanced F1-Score")
            return threshold
        elif choice == '4':
            best = df_results.loc[df_results['net_profit'].idxmax()]
            threshold = best['threshold']
            save_threshold(threshold, "Maximum Profit")
            return threshold
        elif choice == '5':
            best = df_results.loc[df_results['roi'].idxmax()]
            threshold = best['threshold']
            save_threshold(threshold, "Maximum ROI")
            return threshold
        elif choice == '6':
            threshold = float(input("Enter threshold (0.30-0.80): ").strip())
            if 0.30 <= threshold <= 0.80:
                save_threshold(threshold, "Custom Threshold")
                return threshold
            else:
                print("❌ Invalid threshold range")
                return None
        elif choice == '7':
            print("✓ No changes made")
            return None
        else:
            print("❌ Invalid choice")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Main execution"""
    print("\n" + "="*120)
    print("THRESHOLD TESTING & OPTIMIZATION TOOL")
    print("="*120)
    
    # Load data
    probabilities, y_true, df = load_model_and_data()
    
    # Test range of thresholds
    thresholds = [0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]
    df_results = test_thresholds(probabilities, y_true, thresholds)
    
    # Display results
    print_comparison_table(df_results)
    print_detailed_analysis(df_results, current_threshold=0.55)
    plot_ascii_chart(df_results)
    print_recommendations(df_results)
    
    # Interactive selection
    selected = interactive_selection(df_results)
    
    if selected:
        print(f"\n🎉 Threshold {selected:.2f} has been applied!")
        print(f"\n📊 Verify with: python3 main.py --menu (option 1)")
        print(f"🎬 Test with: python3 main.py --demo")
    
    print("\n" + "="*120)
    print("✅ THRESHOLD TESTING COMPLETE")
    print("="*120)


if __name__ == "__main__":
    main()
