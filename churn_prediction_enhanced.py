"""
Enhanced Churn Prediction Model with Advanced Features
- Improved feature engineering (interaction terms, risk scores)
- Focal loss for better class imbalance handling
- Learning rate scheduling and early stopping
- Validation-based model selection
- Comprehensive metrics tracking
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, precision_recall_curve, auc)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance - focuses on hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ChurnModel(nn.Module):
    """Enhanced MLP with batch normalization and stronger regularization"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def advanced_feature_engineering(df):
    """Create discriminative interaction and risk-based features"""
    df = df.copy()
    
    # Original engineered features
    df['tenure_to_charges_ratio'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1e-6)
    
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_count'] = df[service_cols].sum(axis=1)
    df['service_density'] = df['service_count'] / (df['MonthlyCharges'] + 1e-6)
    df['payment_reliability'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1e-6)
    
    # NEW: Contract-Tenure interactions (critical for churn)
    df['contract_tenure_interaction'] = df['Contract'] * df['tenure']
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
    df['is_established_customer'] = (df['tenure'] > 24).astype(int)
    
    # NEW: Price sensitivity indicators
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6)
    df['high_monthly_charges'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    
    # NEW: Payment method risk (electronic check has higher churn)
    df['risky_payment'] = (df['PaymentMethod'] == 2).astype(int)  # Assuming 2=Electronic check after encoding
    
    # NEW: Service adoption patterns
    df['has_streaming'] = ((df['StreamingTV'] == 1) | (df['StreamingMovies'] == 1)).astype(int)
    df['has_protection_bundle'] = ((df['OnlineSecurity'] == 1) & (df['OnlineBackup'] == 1)).astype(int)
    df['no_internet_services'] = (df['InternetService'] == 2).astype(int)  # Assuming 2='No'
    
    # NEW: Customer value segments
    df['ltv_estimate'] = df['MonthlyCharges'] * df['tenure']
    df['high_value_customer'] = (df['ltv_estimate'] > df['ltv_estimate'].quantile(0.75)).astype(int)
    
    # NEW: Engagement indicators
    df['senior_high_cost'] = (df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > 70)
    df['partner_no_dependents'] = (df['Partner'] == 1) & (df['Dependents'] == 0)
    
    return df


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif (self.mode == 'max' and score < self.best_score + self.min_delta) or \
             (self.mode == 'min' and score > self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ENHANCED CHURN PREDICTION MODEL TRAINING")
    print("="*80 + "\n")
    
    # Load dataset
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data.drop(columns=['customerID'], inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.fillna(0, inplace=True)
    
    # Dataset statistics
    churn_rate = data['Churn'].value_counts(normalize=True)
    print(f"📊 Dataset Statistics:")
    print(f"   Total customers: {len(data):,}")
    print(f"   Churn rate: {churn_rate.get('Yes', 0):.2%}")
    print(f"   Non-churn rate: {churn_rate.get('No', 0):.2%}")
    print(f"   Class imbalance ratio: {churn_rate.get('No', 0) / churn_rate.get('Yes', 1):.2f}:1\n")
    
    # Encode categorical columns
    le_churn = LabelEncoder()
    label_encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        else:
            data[col] = le_churn.fit_transform(data[col])
    
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # Advanced Feature Engineering
    print("🔧 Feature Engineering:")
    print(f"   Original features: {X.shape[1]}")
    X = advanced_feature_engineering(X)
    print(f"   Enhanced features: {X.shape[1]}")
    print(f"   New features added: {X.shape[1] - 19}\n")
    
    # Train/Val/Test split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"📈 Data Splits:")
    print(f"   Training: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)\n")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # Initialize enhanced model
    model = ChurnModel(input_dim=X_train_t.shape[1], hidden_dims=[128, 64, 32], dropout=0.4)
    
    # Use Focal Loss for better class imbalance handling
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, mode='max')
    
    # Training configuration
    num_epochs = 200
    best_val_auc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}
    
    print(f"🚀 Training Configuration:")
    print(f"   Model: Enhanced MLP [128→64→32→1]")
    print(f"   Loss: Focal Loss (α=0.25, γ=2.0)")
    print(f"   Optimizer: AdamW (lr=0.001, weight_decay=0.01)")
    print(f"   Scheduler: ReduceLROnPlateau")
    print(f"   Early Stopping: Patience=20, Monitor=val_auc")
    print(f"   Max Epochs: {num_epochs}\n")
    
    print("="*80)
    print("TRAINING PROGRESS")
    print("="*80 + "\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_probs = torch.sigmoid(val_outputs).numpy().flatten()
            val_preds = (val_probs > 0.5).astype(int)
            
            # Calculate metrics
            val_auc = roc_auc_score(y_val, val_probs)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            val_recall = recall_score(y_val, val_preds, zero_division=0)
            val_precision = precision_score(y_val, val_preds, zero_division=0)
        
        # Track history
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        # Update scheduler
        scheduler.step(val_auc)
        
        # Check early stopping
        early_stopping(val_auc, model)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss.item():.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Val Recall: {val_recall:.4f}")
        
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            model.load_state_dict(early_stopping.best_model_state)
            break
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
    
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80 + "\n")
    
    # Load best model and evaluate on test set
    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_probs = torch.sigmoid(test_outputs).numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)
        
        # Comprehensive metrics
        test_acc = accuracy_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds, zero_division=0)
        test_recall = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        test_auc = roc_auc_score(y_test, test_probs)
        
        # PR-AUC (more informative for imbalanced datasets)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_probs)
        test_pr_auc = auc(recall_curve, precision_curve)
    
    print(f"📊 Test Set Performance:")
    print(f"   Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Precision:   {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"   Recall:      {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"   F1-Score:    {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"   ROC-AUC:     {test_auc:.4f} ({test_auc*100:.2f}%)")
    print(f"   PR-AUC:      {test_pr_auc:.4f} ({test_pr_auc*100:.2f}%)")
    
    # Quality assessment
    print(f"\n🎯 Model Quality Assessment:")
    if test_auc >= 0.80:
        print(f"   ✅ EXCELLENT: ROC-AUC ≥ 0.80 - Production ready!")
    elif test_auc >= 0.75:
        print(f"   ✅ GOOD: ROC-AUC ≥ 0.75 - Acceptable for deployment")
    elif test_auc >= 0.70:
        print(f"   ⚠️  FAIR: ROC-AUC ≥ 0.70 - Consider further tuning")
    else:
        print(f"   ❌ POOR: ROC-AUC < 0.70 - Needs improvement")
    
    if test_f1 >= 0.60:
        print(f"   ✅ EXCELLENT: F1-Score ≥ 0.60 - Well balanced")
    elif test_f1 >= 0.50:
        print(f"   ✅ GOOD: F1-Score ≥ 0.50 - Acceptable balance")
    else:
        print(f"   ⚠️  F1-Score < 0.50 - Balance needs improvement")
    
    if test_recall >= 0.70:
        print(f"   ✅ EXCELLENT: Recall ≥ 0.70 - Catching most churners")
    elif test_recall >= 0.60:
        print(f"   ✅ GOOD: Recall ≥ 0.60 - Decent churn detection")
    else:
        print(f"   ⚠️  Recall < 0.60 - Missing too many churners")
    
    # Save model and artifacts
    print(f"\n💾 Saving Model and Artifacts...")
    os.makedirs('models', exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), 'models/churn_model.pth')
    print(f"   ✓ Model saved to models/churn_model.pth")
    
    # Save scaler
    import pickle
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Scaler saved to models/scaler.pkl")
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    print(f"   ✓ Feature names saved ({len(X.columns)} features)")
    
    # Save label encoders for reproducibility
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"   ✓ Label encoders saved ({len(label_encoders)} encoders)")
    
    # Save training history
    import json
    with open('models/training_history.json', 'w') as f:
        json.dump({
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
            'test_metrics': {
                'accuracy': float(test_acc),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1': float(test_f1),
                'roc_auc': float(test_auc),
                'pr_auc': float(test_pr_auc)
            },
            'config': {
                'input_dim': X_train_t.shape[1],
                'hidden_dims': [128, 64, 32],
                'dropout': 0.4,
                'epochs_trained': epoch + 1,
                'best_val_auc': float(best_val_auc)
            }
        }, f, indent=2)
    print(f"   ✓ Training history saved\n")
    
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80 + "\n")
    
    print(f"🎉 Model is ready for deployment!")
    print(f"   Next steps:")
    print(f"   1. Run calibration: python calibrate_churn_model.py")
    print(f"   2. Evaluate system: python main.py --menu")
    print(f"   3. Run demo: python main.py --demo\n")
