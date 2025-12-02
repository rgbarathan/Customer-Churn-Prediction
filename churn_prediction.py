
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define model with dropout for regularization
class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No sigmoid here, will use BCEWithLogitsLoss
        return x

# Only run training when script is executed directly
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data.drop(columns=['customerID'], inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.fillna(0, inplace=True)

    # Calculate churn rate for class imbalance handling
    churn_rate = data['Churn'].value_counts(normalize=True)
    print(f"\n--- Dataset Statistics ---")
    print(f"Total customers: {len(data)}")
    print(f"Churn rate: {churn_rate.get('Yes', 0):.2%}")
    print(f"Non-churn rate: {churn_rate.get('No', 0):.2%}")
    print(f"-------------------------\n")

    # Encode categorical columns
    le_churn = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            data[col] = LabelEncoder().fit_transform(data[col])
        else:
            data[col] = le_churn.fit_transform(data[col])

    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Feature Engineering: Add interaction features
    print("--- Feature Engineering ---")
    print(f"Original features: {X.shape[1]}")

    # 1. Tenure to charges ratio (value indicator)
    X['tenure_to_charges_ratio'] = X['TotalCharges'] / (X['tenure'] * X['MonthlyCharges'] + 1e-6)

    # 2. Service density (services per dollar)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    X['service_count'] = X[service_cols].sum(axis=1)
    X['service_density'] = X['service_count'] / (X['MonthlyCharges'] + 1e-6)

    # 3. Payment reliability (actual vs expected payments)
    X['payment_reliability'] = X['TotalCharges'] / (X['tenure'] * X['MonthlyCharges'] + 1e-6)

    print(f"Enhanced features: {X.shape[1]}")
    print(f"New features added: {X.shape[1] - 19}")
    print(f"--------------------------\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling - Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    model = ChurnModel(input_dim=X_train_t.shape[1])

    # Calculate positive weight for class imbalance (non-churn / churn)
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count])
    print(f"\n--- Class Imbalance Handling ---")
    print(f"Training samples: {len(y_train)}")
    print(f"Churn samples: {pos_count} ({pos_count/len(y_train):.2%})")
    print(f"Non-churn samples: {neg_count} ({neg_count/len(y_train):.2%})")
    print(f"Positive weight: {pos_weight.item():.2f}")
    print(f"-------------------------------\n")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 100 epochs (increased from 10)
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}')

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        preds_probs = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
        preds_cls = (preds_probs.numpy() > 0.5).astype(int)
        
        # Calculate multiple metrics
        acc = accuracy_score(y_test, preds_cls)
        precision = precision_score(y_test, preds_cls, zero_division=0)
        recall = recall_score(y_test, preds_cls, zero_division=0)
        f1 = f1_score(y_test, preds_cls, zero_division=0)
        
        print(f'\n--- Test Results ---')
        print(f'Accuracy:  {acc:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall:    {recall:.4f}')
        print(f'F1-Score:  {f1:.4f}')
        print(f'-------------------\n')

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/churn_model.pth')

    # Also save the scaler and feature names for later use in predictions
    import pickle
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    print("Model, scaler, and feature names saved successfully!")
    print(f"Model expects {X.shape[1]} features")
