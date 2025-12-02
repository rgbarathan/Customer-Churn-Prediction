# 📚 Project Files Index

## Complete File Listing for Customer Churn Prediction & Interactive Q&A System

---

## Core Machine Learning Files

### 1. **churn_prediction.py** (100 lines)
- **Purpose**: Train neural network on Telco customer data
- **Features**: 
  - 3-layer feedforward network (19→64→32→1)
  - StandardScaler for feature normalization
  - 100 epochs training with Adam optimizer
  - Accuracy: 81.19%, Precision: 66.46%, Recall: 58.45%
- **Outputs**: 
  - `models/churn_model.pth` - Trained model weights
  - `models/scaler.pkl` - Feature scaling parameters

### 2. **main.py** (141 lines)
- **Purpose**: Main entry point demonstrating full system
- **Features**:
  - Loads trained model and scaler
  - Predicts churn for 5 customer profiles
  - Integrated QA system engagement
  - Multiple modes: standard, demo, interactive
- **Modes**:
  - `python main.py` - Standard mode (automated)
  - `python main.py --demo` - Demo mode (pre-scripted)
  - `python main.py --interactive` - Interactive mode (live input)

---

## Question-Answering System Files

### 3. **squad_qa_system.py** (167 lines)
- **Purpose**: Q&A system using SQuAD dataset + Comcast KB
- **Features**:
  - Loads 39,274 SQuAD contexts (train + dev)
  - 13 curated Comcast KB contexts
  - DistilBERT neural QA model
  - 3-tier context search (priority: Comcast KB → SQuAD → fallback)
  - Category-based routing (billing, services, retention, support)
- **Dependencies**: Hugging Face Transformers, JSON parsing

### 4. **QA.py** (11 lines)
- **Purpose**: Standalone Q&A demonstration
- **Features**: Basic example of pipeline usage

---

## NEW: Logging System

### 5. **conversation_logger.py** (280 lines) ⭐ NEW
- **Purpose**: Track all conversation sessions
- **Classes**: `ConversationLogger`
- **Key Methods**:
  - `start_session()` - Begin tracking
  - `log_question()` - Record Q&A exchange
  - `end_session()` - Finalize and save
  - `generate_report()` - Aggregate statistics
  - `print_report()` - Formatted console output
  - `save_report()` - Save to JSON
  - `export_csv()` - Export to spreadsheet format
  
**Outputs**:
```
logs/
├── session_YYYYMMDD_HHMMSS.json    # Individual sessions
├── session_report.json              # Aggregate report
└── conversations_export.csv         # Spreadsheet data
```

---

## NEW: CSR Training System

### 6. **training_session.py** (470 lines) ⭐ NEW
- **Purpose**: Interactive training for customer service reps
- **Classes**: `CSRTrainingSystem`
- **Features**:
  - 4 difficulty levels (Beginner → Expert)
  - 5 training customer scenarios
  - Interactive practice sessions
  - Performance scoring (0-10)
  - Session reporting and analytics
- **Key Methods**:
  - `main_menu()` - Main training menu
  - `run_training_session()` - Interactive training loop
  - `select_difficulty()` - Choose training level
  - `view_training_report()` - Show progress
  - `_score_response()` - Rate CSR performance

**Running**:
```bash
python training_session.py
```

---

## Documentation Files

### 7. **README.md** (200+ lines)
- Project overview
- System architecture
- Feature explanations
- How to run instructions

### 8. **INTERACTIVE_GUIDE.md** (350+ lines) ⭐ NEW
- User guide for interactive mode
- 3 running modes explained
- Best practices for CSRs
- Confidence score interpretation
- Example conversations
- Troubleshooting guide

### 9. **INTEGRATION_SUMMARY.md**
- SQuAD dataset integration details
- Comcast KB setup
- Architecture explanation
- Data loading confirmation

### 10. **ARCHITECTURE.md**
- System design overview
- Component interactions
- Data flow diagrams
- Technical specifications

### 11. **PROJECT_COMPLETION.md**
- Feature checklist
- Testing status
- Deployment readiness
- Future enhancements

### 12. **ENHANCEMENT_SUMMARY.md** ⭐ NEW
- Complete summary of options 1-3
- Features implemented
- Usage instructions
- Integration details
- Benefits overview

### 13. **PROJECT_FILES_INDEX.md** (This file) ⭐ NEW
- Master file listing
- Quick reference guide

---

## Data Files

### 14. **WA_Fn-UseC_-Telco-Customer-Churn.csv**
- **Purpose**: Telco customer data for training
- **Records**: 7,043 customers
- **Features**: 19 attributes (tenure, charges, services, etc.)
- **Target**: Churn (binary: Yes/No)

### 15. **archive/train-v2.0.json**
- **Purpose**: SQuAD v2.0 training dataset
- **Size**: ~40 MB
- **Contexts**: 19,035 paragraphs for QA

### 16. **archive/dev-v2.0.json**
- **Purpose**: SQuAD v2.0 development dataset
- **Size**: ~4 MB
- **Contexts**: 20,239 paragraphs for QA

---

## Model & Output Files (Generated)

### 17. **models/churn_model.pth**
- PyTorch saved model weights
- Size: ~500 KB
- Created by: `churn_prediction.py`

### 18. **models/scaler.pkl**
- Feature scaling parameters (StandardScaler)
- Size: ~1 KB
- Created by: `churn_prediction.py`

### 19. **logs/** (Directory)
- **session_YYYYMMDD_HHMMSS.json**: Individual session records
- **session_report.json**: Aggregate statistics
- **conversations_export.csv**: Spreadsheet format export

---

## Directory Structure

```
Project Root/
├── churn_prediction.py              # Model training
├── main.py                          # Main entry point
├── squad_qa_system.py               # Q&A system
├── QA.py                            # Basic Q&A demo
├── conversation_logger.py           # Logging system (NEW)
├── training_session.py              # CSR training (NEW)
│
├── README.md                        # Project overview
├── INTERACTIVE_GUIDE.md             # Interactive mode guide (NEW)
├── INTEGRATION_SUMMARY.md           # SQuAD integration details
├── ARCHITECTURE.md                  # System architecture
├── PROJECT_COMPLETION.md            # Completion status
├── ENHANCEMENT_SUMMARY.md           # Options 1-3 summary (NEW)
├── PROJECT_FILES_INDEX.md           # This file (NEW)
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Training data
│
├── archive/
│   ├── train-v2.0.json             # SQuAD training data (19K contexts)
│   └── dev-v2.0.json               # SQuAD dev data (20K contexts)
│
├── models/                          # Generated models
│   ├── churn_model.pth             # Trained neural network
│   └── scaler.pkl                  # Feature scaler
│
└── logs/                            # Generated logs
    ├── session_*.json               # Individual sessions
    ├── session_report.json          # Report
    └── conversations_export.csv     # CSV export
```

---

## Quick Start Guide

### 1. **Train the Model**
```bash
python churn_prediction.py
```
Output: `models/churn_model.pth`, `models/scaler.pkl`

### 2. **Run Standard Prediction**
```bash
python main.py
```
Output: Customer predictions, automated QA responses

### 3. **Demo Interactive Mode**
```bash
python main.py --demo
```
Output: Pre-scripted interactive conversation example

### 4. **Live Interactive Mode**
```bash
python main.py --interactive
```
User input: Ask questions to AI in real-time

### 5. **CSR Training**
```bash
python training_session.py
```
Output: Interactive training with scoring

### 6. **View Training Report**
```bash
cat logs/session_report.json
```

### 7. **Export to Spreadsheet**
```bash
cat logs/conversations_export.csv
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | PyTorch | Neural network training |
| **NLP Library** | Hugging Face Transformers | Question answering |
| **Data Processing** | Pandas, Scikit-learn | Feature scaling, encoding |
| **Logging** | JSON, CSV | Session tracking |
| **Interface** | Terminal/CLI | User interaction |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Model Accuracy | 81.19% |
| Model Precision | 66.46% |
| Model Recall | 58.45% |
| F1-Score | 62.20% |
| SQuAD Contexts | 39,274 |
| Comcast KB | 13 curated contexts |
| Customer Scenarios | 5 profiles |
| Training Levels | 4 (Beginner → Expert) |

---

## Recent Changes Summary

### ✅ Completed Enhancements (December 1, 2025)

**Option 1: Interactive Mode Testing**
- ✅ Tested with pre-scripted questions
- ✅ Verified multi-turn conversation
- ✅ Confirmed auto-categorization works
- ✅ Tested confidence scoring

**Option 2: Conversation Logging**
- ✅ Created `conversation_logger.py` (280 lines)
- ✅ Implemented session tracking
- ✅ JSON + CSV export capabilities
- ✅ Report generation and statistics

**Option 3: CSR Training Script**
- ✅ Created `training_session.py` (470 lines)
- ✅ 4 difficulty levels implemented
- ✅ 5 training customer scenarios
- ✅ Interactive practice with scoring

---

## How to Use Each File

### For Model Training
1. Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in root
2. Run: `python churn_prediction.py`
3. Check: `models/` directory for outputs

### For Predictions
1. Ensure models are trained (see above)
2. Run: `python main.py`
3. Choose mode: standard, --demo, or --interactive

### For CSR Training
1. Run: `python training_session.py`
2. Select difficulty (1-4 or random)
3. Practice with assigned customer
4. Get scoring feedback
5. View results: select option "2" in menu

### For Logging & Analytics
1. Sessions auto-logged during training or interactive mode
2. View report: `python conversation_logger.py`
3. Check files: `logs/session_report.json`
4. Export data: `logs/conversations_export.csv`

---

## File Dependencies

```
main.py
├── churn_prediction.py      → ChurnModel class
├── squad_qa_system.py       → SQuADQASystem class
├── conversation_logger.py   → ConversationLogger (optional)
└── Data files: CSV, JSON

training_session.py
├── squad_qa_system.py       → SQuADQASystem class
├── conversation_logger.py   → ConversationLogger class
└── Data files: JSON

conversation_logger.py
└── (No ML dependencies)
```

---

## Support & Documentation

| Need | File/Resource |
|------|---------------|
| General overview | README.md |
| Interactive usage | INTERACTIVE_GUIDE.md |
| Architecture details | ARCHITECTURE.md |
| Q&A system details | INTEGRATION_SUMMARY.md |
| Training info | This file + training_session.py |
| Enhancement details | ENHANCEMENT_SUMMARY.md |
| Logging details | conversation_logger.py |

---

## Next Steps

1. **For Deployment**: Use `python main.py --interactive`
2. **For Training**: Use `python training_session.py`
3. **For Analytics**: Review `logs/session_report.json`
4. **For Improvement**: Track metrics in conversations_export.csv

---

*Last Updated: December 1, 2025*  
*Status: Complete & Tested ✅*  
*All 3 Enhancement Options Implemented*
