
import torch
import pandas as pd
from transformers import pipeline
from churn_prediction import ChurnModel
from squad_qa_system import SQuADQASystem
import os
import pickle
import sys

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load churn model
model = ChurnModel(input_dim=19)
model.load_state_dict(torch.load('models/churn_model.pth'))
model.eval()

# Initialize enhanced SQuAD-based QA system
qa_system = SQuADQASystem()

def interactive_qa_session(customer_name, churn_probability):
    """
    Interactive Q&A session with a high-risk customer.
    Allows the customer service rep to ask multiple questions.
    """
    print(f"\n{'='*70}")
    print(f"INTERACTIVE QA SESSION - {customer_name}")
    print(f"{'='*70}")
    print(f"Churn Risk: {churn_probability:.2%}")
    print(f"\n📝 Comcast Customer Service Representative")
    print(f"You can now ask questions to help retain this customer.")
    print(f"Type 'exit' or 'quit' to end the session.\n")
    
    session_history = []
    
    while True:
        try:
            # Get user question
            user_input = input("🎤 Your question: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q', 'end']:
                print(f"\n✅ Session ended for {customer_name}")
                break
            
            # Skip empty input
            if not user_input:
                print("⚠️  Please enter a question.\n")
                continue
            
            # Determine category based on question content
            question_lower = user_input.lower()
            if any(word in question_lower for word in ['bill', 'price', 'cost', 'discount', 'reduce', 'cheaper', 'save']):
                category = "billing"
            elif any(word in question_lower for word in ['offer', 'deal', 'loyalty', 'keep', 'retain', 'stay', 'program']):
                category = "retention"
            elif any(word in question_lower for word in ['service', 'internet', 'speed', 'tv', 'phone', 'add', 'upgrade']):
                category = "services"
            elif any(word in question_lower for word in ['help', 'support', 'contact', 'call', 'assist', 'issue', 'problem']):
                category = "support"
            else:
                category = "retention"  # Default to retention for at-risk customers
            
            print(f"\n🤔 Searching knowledge base (category: {category})...")
            
            # Get answer from QA system
            qa_response = qa_system.answer_question(user_input, category)
            
            print(f"✅ Answer: {qa_response['answer']}")
            print(f"📊 Confidence: {qa_response['confidence']:.2%}")
            
            # Store in session history
            session_history.append({
                'question': user_input,
                'answer': qa_response['answer'],
                'confidence': qa_response['confidence'],
                'category': category
            })
            
            # Provide follow-up suggestions
            if qa_response['confidence'] < 0.5:
                print(f"⚠️  Low confidence answer. You may want to verify with your supervisor.")
            
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n✅ Session interrupted for {customer_name}")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue
    
    return session_history

# Example combined workflow
# Low-risk customer (loyal, long tenure)
customer_data_low = pd.DataFrame([[1, 0, 29, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 70.0, 150.0]],
                              columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])

# Scale the customer data using the same scaler
customer_data_scaled = scaler.transform(customer_data_low)
customer_tensor = torch.tensor(customer_data_scaled, dtype=torch.float32)
churn_prob_low = model(customer_tensor).item()
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

customer_data_scaled_high1 = scaler.transform(customer_data_high1)
customer_tensor_high1 = torch.tensor(customer_data_scaled_high1, dtype=torch.float32)
churn_prob_high1 = model(customer_tensor_high1).item()
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

customer_data_scaled_high2 = scaler.transform(customer_data_high2)
customer_tensor_high2 = torch.tensor(customer_data_scaled_high2, dtype=torch.float32)
churn_prob_high2 = model(customer_tensor_high2).item()
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

customer_data_scaled_high3 = scaler.transform(customer_data_high3)
customer_tensor_high3 = torch.tensor(customer_data_scaled_high3, dtype=torch.float32)
churn_prob_high3 = model(customer_tensor_high3).item()
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

customer_data_scaled_high4 = scaler.transform(customer_data_high4)
customer_tensor_high4 = torch.tensor(customer_data_scaled_high4, dtype=torch.float32)
churn_prob_high4 = model(customer_tensor_high4).item()
print("=" * 70)
print("CUSTOMER 5: HIGH-RISK #4 (Very New - Highest Charges)")
print("=" * 70)
print(f"Tenure: 1 month | Monthly Charges: $115 | Total Charges: $115")
print(f"Services: Phone + Fiber Internet (premium service)")
print(f"Fiber optic (premium but may be dissatisfied with speed/quality)")
print(f"Churn probability: {churn_prob_high4:.2%}")
print(f"Status: ⚠️  CRITICAL RISK - Brand new customer with high bill\n")

# QA System Engagement - Handle all high-risk customers
high_risk_customers = [
    {"name": "Customer 2", "prob": churn_prob_high1, "category": "billing"},
    {"name": "Customer 3", "prob": churn_prob_high2, "category": "retention"},
    {"name": "Customer 4", "prob": churn_prob_high3, "category": "services"},
    {"name": "Customer 5", "prob": churn_prob_high4, "category": "retention"}
]

print("=" * 70)
print("ENHANCED QA SYSTEM WITH SQUAD DATASET")
print("=" * 70)

# Interactive mode for critical risk customers
critical_risk_found = False
# Default to interactive mode; use --demo or --automated to override
DEMO_MODE = len(sys.argv) > 1 and sys.argv[1] == '--demo'
AUTOMATED_MODE = len(sys.argv) > 1 and sys.argv[1] == '--automated'
INTERACTIVE_MODE = not DEMO_MODE and not AUTOMATED_MODE  # Default to True

for customer in high_risk_customers:
    if customer["prob"] > 0.7:  # Critical risk threshold
        critical_risk_found = True
        print(f"\n{customer['name']}: {customer['prob']:.2%} Churn Risk")
        print("-" * 70)
        print(f"Status: 🔴 CRITICAL - This customer needs immediate attention!")
        
        if DEMO_MODE:
            # Demo mode with pre-scripted questions
            print(f"\n📋 DEMO MODE: Showing sample interactive conversation\n")
            demo_questions = [
                "How can I get a discount on my internet bill?",
                "What bundle packages do you offer?",
                "Do you have any loyalty programs for long-term customers?"
            ]
            
            demo_history = []
            for demo_q in demo_questions:
                print(f"🎤 CSR: {demo_q}")
                
                # Determine category
                question_lower = demo_q.lower()
                if any(word in question_lower for word in ['bill', 'price', 'cost', 'discount', 'reduce']):
                    category = "billing"
                else:
                    category = "retention"
                
                qa_response = qa_system.answer_question(demo_q, category)
                print(f"🤖 AI: {qa_response['answer']}")
                print(f"   Confidence: {qa_response['confidence']:.2%}\n")
                demo_history.append({'q': demo_q, 'a': qa_response['answer'], 'conf': qa_response['confidence']})
            
            print(f"📋 Session Summary:")
            print(f"   Questions asked: {len(demo_history)}")
            avg_conf = sum(q['conf'] for q in demo_history) / len(demo_history)
            print(f"   Average confidence: {avg_conf:.2%}\n")
            
        elif INTERACTIVE_MODE and sys.stdin.isatty():
            # Only use interactive mode if running in a terminal
            print(f"\nWould you like to engage in an interactive Q&A with a CSR?")
            print(f"(This simulates a customer service representative helping this customer)")
            
            try:
                user_choice = input("\nEnter 'yes' for interactive mode, or press Enter to continue: ").strip().lower()
                
                if user_choice in ['yes', 'y']:
                    session_history = interactive_qa_session(customer["name"], customer["prob"])
                    print(f"\n📋 Session Summary for {customer['name']}:")
                    print(f"   Total questions asked: {len(session_history)}")
                    if session_history:
                        avg_confidence = sum(q['confidence'] for q in session_history) / len(session_history)
                        print(f"   Average confidence: {avg_confidence:.2%}")
                else:
                    # Non-interactive mode
                    print(f"\n🤖 Automated Response:")
                    
                    question = "What loyalty programs are available?"
                    category = "retention"
                    
                    qa_response = qa_system.handle_churn_customer(question, customer["prob"])
                    
                    print(f"Q: {question}")
                    print(f"A: {qa_response['answer']}")
                    print(f"Confidence: {qa_response['confidence']:.2%}")
                    print(f"✉️  Action: Contact customer with retention offer")
            except EOFError:
                # Not in interactive terminal, show automated response
                print(f"\n🤖 Automated Response (not in interactive mode):")
                
                question = "What loyalty programs are available?"
                category = "retention"
                
                qa_response = qa_system.handle_churn_customer(question, customer["prob"])
                
                print(f"Q: {question}")
                print(f"A: {qa_response['answer']}")
                print(f"Confidence: {qa_response['confidence']:.2%}")
                print(f"✉️  Action: Contact customer with retention offer")
        else:
            # Non-interactive mode
            print(f"\n🤖 Automated Response:")
            
            question = "What loyalty programs are available?"
            category = "retention"
            
            qa_response = qa_system.handle_churn_customer(question, customer["prob"])
            
            print(f"Q: {question}")
            print(f"A: {qa_response['answer']}")
            print(f"Confidence: {qa_response['confidence']:.2%}")
            print(f"✉️  Action: Contact customer with retention offer")

# Display non-critical high-risk customers
if not critical_risk_found:
    print("\n📊 High-Risk Customers (Non-Critical):")
    
for customer in high_risk_customers:
    if 0.5 < customer["prob"] <= 0.7:  # High risk but not critical
        print(f"\n{customer['name']}: {customer['prob']:.2%} Churn Risk")
        print("-" * 70)
        
        question = "How can I reduce my bill?"
        category = "billing"
        
        qa_response = qa_system.handle_churn_customer(question, customer["prob"])
        
        print(f"Q: {question}")
        print(f"A: {qa_response['answer']}")
        print(f"Confidence: {qa_response['confidence']:.2%}")
        print(f"✉️  Action: Contact customer with retention offer")

print("\n" + "=" * 70)
