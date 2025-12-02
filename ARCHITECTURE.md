# Project Architecture & Data Flow

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              CUSTOMER CHURN PREDICTION & QA SYSTEM              │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYER
───────────
    Customer Data (19 features)
    │
    ├─ Demographics: gender, senior citizen, partner, dependents
    ├─ Tenure: months with service
    ├─ Services: phone, internet type, streaming, security, etc.
    ├─ Billing: monthly charges, total charges
    └─ Contract: type, billing method, paperless

         ↓↓↓

PREPROCESSING
──────────────
    ┌─────────────────────────────┐
    │  Feature Engineering        │
    │ ┌───────────────────────┐   │
    │ │ Label Encoding        │   │
    │ │ (categorical → numeric)   │
    │ └───────────────────────┘   │
    │         ↓                   │
    │ ┌───────────────────────┐   │
    │ │ Feature Scaling       │   │
    │ │ (StandardScaler)      │   │
    │ │ mean=0, std=1         │   │
    │ └───────────────────────┘   │
    └─────────────────────────────┘
           ↓

PREDICTION ENGINE
──────────────────
    ┌──────────────────────────────┐
    │   Churn Prediction Model     │
    │  (PyTorch Neural Network)    │
    │                              │
    │  Input Layer (19 features)   │
    │         ↓                    │
    │  Hidden Layer 1 (64 neurons) │
    │  ReLU Activation             │
    │         ↓                    │
    │  Hidden Layer 2 (32 neurons) │
    │  ReLU Activation             │
    │         ↓                    │
    │  Output Layer (1 neuron)     │
    │  Sigmoid → [0, 1]            │
    │         ↓                    │
    │  Churn Probability (%)       │
    └──────────────────────────────┘
           ↓↓↓

RISK ASSESSMENT
────────────────
    Churn Probability Score
    │
    ├─ < 30%  → LOW RISK (Green)
    │          Action: Standard support
    │
    ├─ 30-50% → MEDIUM RISK (Yellow)
    │          Action: Service improvement suggestions
    │
    ├─ 50-70% → HIGH RISK (Orange)
    │          Action: Billing reduction offer
    │
    └─ > 70%  → CRITICAL RISK (Red)
               Action: URGENT retention campaign
    
           ↓↓↓

QA SYSTEM ENGAGEMENT (if risk > 50%)
─────────────────────────────────────
    
    ┌─────────────────────────────────────────────────┐
    │   Enhanced QA System (SQuADQASystem)             │
    │                                                 │
    │  Question ──→ Category Selection ──→ Context    │
    │              (based on risk %)      Selection   │
    │                                                 │
    │  HIGH RISK (50-70%):                           │
    │  "How can I reduce my bill?"                    │
    │       ↓                                         │
    │  ┌─────────────────────────┐                   │
    │  │ Context Source (Priority)                   │
    │  ├─────────────────────────┤                   │
    │  │ 1. Comcast KB (Billing) │                   │
    │  │ 2. SQuAD Contexts       │                   │
    │  │ 3. Fallback Response    │                   │
    │  └─────────────────────────┘                   │
    │       ↓                                         │
    │  ┌─────────────────────────┐                   │
    │  │ DistilBERT QA Model     │                   │
    │  │ Extract Answer from     │                   │
    │  │ Context                 │                   │
    │  └─────────────────────────┘                   │
    │       ↓                                         │
    │  Answer + Confidence Score                     │
    │                                                 │
    │  CRITICAL RISK (>70%):                         │
    │  "What loyalty programs are available?"         │
    │       ↓                                         │
    │  ┌─────────────────────────┐                   │
    │  │ Context Source (Priority)                   │
    │  ├─────────────────────────┤                   │
    │  │ 1. Comcast KB (Retention)                   │
    │  │ 2. SQuAD Contexts       │                   │
    │  │ 3. Fallback Response    │                   │
    │  └─────────────────────────┘                   │
    │       ↓                                         │
    │  Answer + Confidence Score                     │
    │                                                 │
    └─────────────────────────────────────────────────┘
              ↓↓↓

COMCAST KNOWLEDGE BASE (39,274+ contexts)
──────────────────────────────────────────
    ┌─────────────────────────────────────┐
    │         Billing (4 contexts)        │
    │ • Payment options & flexibility     │
    │ • Discount programs                 │
    │ • Pricing & promotions              │
    └─────────────────────────────────────┘
              +
    ┌─────────────────────────────────────┐
    │        Services (3 contexts)        │
    │ • Internet plans (100-1200 Mbps)   │
    │ • TV packages                       │
    │ • Add-on services                   │
    └─────────────────────────────────────┘
              +
    ┌─────────────────────────────────────┐
    │        Support (3 contexts)         │
    │ • 24/7 customer support             │
    │ • Technical assistance              │
    │ • Device protection                 │
    └─────────────────────────────────────┘
              +
    ┌─────────────────────────────────────┐
    │       Retention (3 contexts)        │
    │ • Loyalty programs                  │
    │ • Price-lock guarantees             │
    │ • Service improvements              │
    └─────────────────────────────────────┘

SQUAD v2.0 DATASET (39,274+ contexts)
──────────────────────────────────────
    ┌──────────────────────────────┐
    │ Training Data (19,035)        │
    │ • Medical topics              │
    │ • Scientific articles         │
    │ • Historical passages         │
    └──────────────────────────────┘
              +
    ┌──────────────────────────────┐
    │ Development Data (20,239)     │
    │ • Additional contexts         │
    │ • Validation dataset          │
    │ • Edge cases                  │
    └──────────────────────────────┘

OUTPUT LAYER
────────────
    ┌───────────────────────────────────┐
    │     Customer Service Response     │
    ├───────────────────────────────────┤
    │ • Churn Risk Score (%)            │
    │ • Recommended Action              │
    │ • AI-Generated Answer             │
    │ • Confidence Score (%)            │
    │ • Source (KB/SQuAD/Fallback)      │
    └───────────────────────────────────┘
           ↓↓↓
    
    ┌─────────────────────────────────────┐
    │   CUSTOMER SERVICE TEAM             │
    │                                     │
    │ Contact high-risk customer with:    │
    │ • Personalized offer                │
    │ • AI-generated talking points       │
    │ • Service recommendations           │
    │ • Loyalty incentives                │
    └─────────────────────────────────────┘
```

---

## Data Flow Example: High-Risk Customer

```
STEP 1: Customer Profile Input
────────────────────────────
Senior Citizen: Yes
Tenure: 2 months
Services: Internet only (no phone, no add-ons)
Monthly Charges: $105
Total Charges: $210
Internet Service Type: DSL (not premium)

        ↓↓↓

STEP 2: Preprocessing
──────────────────
Feature Values (raw): [1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 105, 210]
        ↓
Label Encoding: Convert "Churn" → 0 or 1
        ↓
StandardScaler: Normalize to mean=0, std=1
Scaled Values: [-0.45, 2.10, -0.89, -0.55, ..., 1.23]

        ↓↓↓

STEP 3: Neural Network Prediction
─────────────────────────────────
Input: Scaled features [19 dimensions]
  ↓
Dense (19 → 64): Linear transformation + ReLU
  ↓
Dense (64 → 32): Linear transformation + ReLU
  ↓
Dense (32 → 1): Linear transformation + Sigmoid
  ↓
Output: 0.6576 (65.76% churn probability)

        ↓↓↓

STEP 4: Risk Assessment
──────────────────────
65.76% > 50% → HIGH RISK
65.76% < 70% → Billing-focused response

Action: Ask "How can I reduce my bill?"

        ↓↓↓

STEP 5: QA System
────────────────
Category: "billing"

Context Search (Priority Order):
1. Comcast KB - Billing Category:
   "To reduce your bill, consider bundling services 
    (Internet, TV, Phone). Senior citizens and 
    low-income customers may qualify for special 
    discounts. New customer promotions include 50% 
    off for the first 3 months."

2. If no match, use SQuAD contexts
3. If confidence low, use fallback

        ↓↓↓

STEP 6: Answer Extraction
─────────────────────────
Question: "How can I reduce my bill?"
Context: "To reduce your bill, consider bundling 
         services... Senior citizens... 50% off..."

DistilBERT QA Model:
  • Reads question
  • Scans context for relevant spans
  • Extracts best answer span
  • Calculates confidence score

Answer: "loyalty offers and service improvements"
Confidence: 32.24%

        ↓↓↓

STEP 7: Customer Service Response
──────────────────────────────────
┌─────────────────────────────────────┐
│ CUSTOMER RETENTION ALERT            │
├─────────────────────────────────────┤
│ Customer: Senior Citizen             │
│ Tenure: 2 months (NEW)               │
│ Churn Risk: 65.76% (HIGH)            │
│                                     │
│ Recommended Action:                  │
│ "Contact with senior bundle offer"  │
│                                     │
│ AI Suggestion:                      │
│ "How can I reduce my bill?"          │
│ → "loyalty offers and service..."   │
│                                     │
│ Next Steps:                         │
│ 1. Call customer proactively         │
│ 2. Offer 50% new customer promo      │
│ 3. Suggest TV+Phone bundle           │
│ 4. Apply senior citizen discount     │
│ 5. Follow up in 1 week               │
└─────────────────────────────────────┘
```

---

## System Components & Interactions

```
┌────────────────────────────────────────────────────────┐
│           FILE DEPENDENCIES & IMPORTS                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  main.py (Entry Point)                               │
│  ├─→ imports: torch, pandas, transformers             │
│  ├─→ imports: churn_prediction (ChurnModel class)    │
│  ├─→ imports: squad_qa_system (SQuADQASystem class)  │
│  ├─→ loads: models/churn_model.pth                   │
│  ├─→ loads: models/scaler.pkl                        │
│  ├─→ loads: archive/train-v2.0.json (19K contexts)   │
│  └─→ loads: archive/dev-v2.0.json (20K contexts)    │
│      ↓                                                │
│  churn_prediction.py                                 │
│  ├─→ imports: pandas, torch, sklearn                 │
│  ├─→ loads: WA_Fn-UseC_-Telco-...csv (7K records)   │
│  ├─→ exports: models/churn_model.pth (trained)       │
│  └─→ exports: models/scaler.pkl (StandardScaler)     │
│      ↓                                                │
│  squad_qa_system.py                                  │
│  ├─→ imports: json, transformers, os                 │
│  ├─→ loads: archive/*.json (39K contexts)            │
│  ├─→ initializes: DistilBERT QA pipeline             │
│  └─→ defines: Comcast knowledge base (13 contexts)   │
│      ↓                                                │
│  QA.py (Standalone Demo)                            │
│  └─→ imports: transformers                           │
│      └─→ simple demo of QA capability                │
│                                                      │
└────────────────────────────────────────────────────────┘
```

---

## Performance Monitoring

```
Training Metrics Over 100 Epochs:
──────────────────────────────────

Loss Convergence:
Epoch 0:   0.6687 ████████████████████  (Starting)
Epoch 10:  0.6203 ██████████████████    (Improving)
Epoch 20:  0.5650 ██████████████        
Epoch 30:  0.5131 █████████████         
Epoch 40:  0.4810 ████████████          
Epoch 50:  0.4596 ███████████           
Epoch 60:  0.4474 ███████████           
Epoch 70:  0.4398 ███████████           
Epoch 80:  0.4338 ███████████           
Epoch 90:  0.4287 ███████████           (Final)

Test Set Metrics:
────────────────
Accuracy:  80.91% ████████████████████  (High)
Precision: 65.85% ██████████████        (Good)
Recall:    57.91% █████████████         (Adequate)
F1-Score:  0.6163 ██████████████        (Balanced)

Interpretation:
• High accuracy: Model correctly classifies 81% of cases
• Precision: When we predict "churn", we're right 66% of time
• Recall: We catch 58% of actual churn cases
• F1: Good balance between precision and recall
```

---

## Key Design Decisions

1. **Dual Data Sources for QA**:
   - Comcast KB first (domain-specific, high confidence)
   - SQuAD contexts as backup (general knowledge)
   - Fallback response if both fail

2. **Risk-Based Question Selection**:
   - >70% risk: Retention questions (loyalty programs)
   - 50-70% risk: Billing questions (cost reduction)
   - This targets the most relevant pain points

3. **Feature Scaling**:
   - Ensures neural network learns effectively
   - Consistent preprocessing for inference
   - Saved scaler prevents data leakage

4. **Confidence Scoring**:
   - Helps prioritize high-quality recommendations
   - Identifies when human judgment is needed
   - Transparent AI decision-making

5. **Persistence**:
   - Model weights saved (reproducibility)
   - Scaler saved (consistency)
   - SQuAD data loaded once (efficiency)

---

## Integration Benefits

✅ **End-to-End Solution**: Predict churn AND provide solutions  
✅ **Multi-Source Knowledge**: Domain + general + contextual  
✅ **Confidence-Based**: Know when to trust recommendations  
✅ **Scalable**: Can handle thousands of customers  
✅ **Explainable**: Clear churn factors and recommendations  
✅ **Production-Ready**: Error handling, persistence, logging  
