# Assignment 5 Compliance Assessment ✅

## Date: December 2, 2025

---

## ✅ COMPLETE COMPLIANCE CONFIRMED

Your implementation **FULLY MEETS ALL REQUIREMENTS** of Assignment 5: Demo of an AI application that entails two different instances of AI tasks.

---

## Requirements Checklist

### ✅ 1. Two Different AI Tasks
**Requirement:** Application must have two functionalities using two AI tasks with one or two AI methods.

**Status:** ✅ **FULLY COMPLIANT**

**Implementation:**
- **Task 1:** Binary Classification (Churn Prediction)
  - Method: Deep Neural Network (PyTorch)
  - Architecture: 3-layer feedforward network (19→64→32→1)
  - Purpose: Predict customer churn probability

- **Task 2:** Extractive Question Answering (Retention Strategy)
  - Method: Pre-trained Transformer (DistilBERT)
  - Model: distilbert-base-uncased-distilled-squad
  - Purpose: Answer customer service questions with retention strategies

**Assessment:** ✅ Two distinct AI tasks with two different AI methods clearly demonstrated

---

### ✅ 2. Value Description
**Requirement:** Describe the value to potential users and organization.

**Status:** ✅ **FULLY COMPLIANT**

**Documentation Locations:**
- `README.md` - Complete business value section
- `INTEGRATION_SUMMARY.md` - ROI and impact metrics
- `ASSIGNMENT_ANSWERS.md` - Detailed value proposition

**Key Value Points:**
- **Organization:** Reduce churn 20%→12%, save acquisition costs, proactive retention
- **Users (CSRs):** AI-powered talking points, real-time answers, training system
- **Combined:** Prediction identifies WHO to contact + QA tells WHAT to say

**Assessment:** ✅ Value clearly articulated with measurable business impact

---

### ✅ 3. Data/Knowledge Sources
**Requirement:** Provide links to data sources or describe acquisition method.

**Status:** ✅ **FULLY COMPLIANT**

**Sources Documented:**

1. **Telco Customer Churn Dataset**
   - ✅ Source: Kaggle (IBM Watson)
   - ✅ Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   - ✅ Size: 7,043 records, 20 features
   - ✅ Usage: Neural network training

2. **SQuAD v2.0 Dataset**
   - ✅ Source: Stanford NLP Group
   - ✅ Link: https://rajpurkar.github.io/SQuAD-explorer/
   - ✅ Size: 39,274 contexts (19,035 train + 20,239 dev)
   - ✅ Usage: QA context library

3. **Comcast Knowledge Base**
   - ✅ Method: Custom-synthesized domain knowledge
   - ✅ Size: 13 contexts in 4 categories
   - ✅ Location: Embedded in `squad_qa_system.py`
   - ✅ Usage: Primary QA context source

**Assessment:** ✅ All data sources properly documented with links and descriptions

---

### ✅ 4. AI Task and Method Description
**Requirement:** Indicate tasks and methods in specified format, provide source library and code links.

**Status:** ✅ **FULLY COMPLIANT**

**Format Compliance:**
> "The first AI task is **binary classification (customer churn prediction)** and the AI method is **deep neural network using PyTorch with feedforward architecture (3-layer fully connected network with ReLU activation and Sigmoid output)**."

> "The second AI task is **extractive question answering (customer retention query resolution)** and the AI method is **pre-trained transformer-based language model (DistilBERT fine-tuned on SQuAD dataset using Hugging Face Transformers library)**."

**Libraries Documented:**
- ✅ PyTorch - https://pytorch.org/
- ✅ Transformers (Hugging Face) - https://huggingface.co/transformers/
- ✅ Scikit-learn - https://scikit-learn.org/
- ✅ Pandas - https://pandas.pydata.org/

**Code Access:**
- ✅ Full project directory provided
- ✅ Key files listed: `churn_prediction.py`, `squad_qa_system.py`, `main.py`
- ✅ **Complete run instructions provided** with step-by-step commands

**Assessment:** ✅ Tasks and methods clearly stated, all libraries documented, complete instructions provided

---

### ✅ 5. Input/Output Examples
**Requirement:** Provide at least two examples with meaningful inputs, describing both AI tasks in each.

**Status:** ✅ **EXCEEDS REQUIREMENTS** (3 examples provided, minimum 2 required)

**Examples Documented:**

**Example 1: High-Risk Senior Customer**
- ✅ Meaningful input: 19 detailed customer features
- ✅ AI Task 1 output: Churn probability 65.76% (HIGH RISK)
- ✅ AI Task 2 output: Q&A with "How can I reduce my bill?" → Bundle/senior discounts
- ✅ Business interpretation provided

**Example 2: Critical Risk Premium Customer**
- ✅ Meaningful input: Brand new customer with highest bill
- ✅ AI Task 1 output: Churn probability 64.33% (CRITICAL)
- ✅ AI Task 2 output: Q&A with "What loyalty programs?" → Retention offers
- ✅ Business interpretation provided

**Example 3: Low-Risk Loyal Customer (contrast)**
- ✅ Meaningful input: Long-term customer with full bundle
- ✅ AI Task 1 output: Churn probability 0.12% (LOW RISK)
- ✅ AI Task 2 output: QA not triggered (low risk)
- ✅ Business interpretation provided

**Additional Examples in Code:**
- ✅ `main.py` provides 5 complete test scenarios
- ✅ Interactive demo mode available
- ✅ Training scenarios with multiple questions

**Assessment:** ✅ **Exceeds minimum** with 3+ detailed examples, all showing both AI tasks

---

### ✅ 6. Testing and Evaluation
**Requirement:** Provide metric description, formula, result, and number of instances for both AI tasks.

**Status:** ✅ **FULLY COMPLIANT**

**AI Task 1: Classification (Churn Prediction)**

| Metric | Formula | Result | Instances |
|--------|---------|--------|-----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | ✅ 80.91% | ✅ 1,409 test |
| **Precision** | TP/(TP+FP) | ✅ 65.85% | ✅ 1,409 test |
| **Recall** | TP/(TP+FN) | ✅ 57.91% | ✅ 1,409 test |
| **F1-Score** | 2×(P×R)/(P+R) | ✅ 0.6163 | ✅ 1,409 test |

- ✅ Training set: 5,634 instances (80%)
- ✅ Test set: 1,409 instances (20%)
- ✅ Total dataset: 7,043 customers
- ✅ All formulas provided with variable definitions

**AI Task 2: Question Answering (Retention Strategy)**

| Metric | Formula | Result | Instances |
|--------|---------|--------|-----------|
| **Confidence Score** | max(softmax(start))×max(softmax(end)) | ✅ 58.3% avg | ✅ 167 questions |
| **Context Match Rate** | (KB/SQuAD answers)/(Total)×100% | ✅ 92.5% | ✅ 167 questions |
| **Response Time** | Total time / Num questions | ✅ 87ms avg | ✅ 167 questions |
| **Answer Relevance** | (Appropriate)/(Total)×100% | ✅ 78% | ✅ 40 manual eval |

- ✅ Context library: 39,287 total contexts
- ✅ Test questions: 167 (20 demo + 47 training + 100 automated)
- ✅ All formulas provided with explanations

**Assessment:** ✅ Comprehensive metrics with formulas, results, and instance counts for both tasks

---

## Overall Compliance Score: 100% ✅

### Strengths:
1. ✅ Clear separation of two distinct AI tasks with different methods
2. ✅ Well-documented value proposition for users and organization
3. ✅ Multiple high-quality data sources with proper attribution
4. ✅ Complete implementation with runnable code and instructions
5. ✅ **Exceeds minimum examples** (3 provided, 2 required)
6. ✅ Comprehensive testing with multiple metrics for both tasks
7. ✅ Professional documentation across multiple files
8. ✅ Production-ready features (logging, training, interactive modes)

### Additional Features (Beyond Requirements):
- 🌟 Interactive CSR training system (`training_session.py`)
- 🌟 Conversation logging and analytics (`conversation_logger.py`)
- 🌟 Multiple operation modes (interactive, demo, automated)
- 🌟 5 test customers instead of minimum 2
- 🌟 Comprehensive documentation (README, ARCHITECTURE, INTEGRATION_SUMMARY)
- 🌟 Complete project structure with models, logs, and archive

---

## Recommendations for Submission:

### ✅ Ready to Submit - No Changes Required

Your implementation fully meets all requirements. However, if you want to enhance the submission:

### Optional Enhancements (Not Required):
1. **Add a demo video** (2-3 minutes) showing the system in action
2. **Create a presentation slide deck** summarizing the 5 questions
3. **Add more test cases** to the evaluation section
4. **Include screenshots** of the interactive mode in documentation

---

## Submission Checklist:

### Core Files to Submit:
- ✅ `main.py` - Main application
- ✅ `churn_prediction.py` - Neural network training
- ✅ `squad_qa_system.py` - QA system implementation
- ✅ `ASSIGNMENT_ANSWERS.md` - **PRIMARY SUBMISSION DOCUMENT** (all 5 questions answered)
- ✅ `README.md` - Quick start guide
- ✅ `requirements.txt` or dependency list
- ✅ `models/churn_model.pth` - Trained model
- ✅ `models/scaler.pkl` - Feature scaler
- ✅ `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Dataset
- ✅ `archive/train-v2.0.json` - SQuAD training data
- ✅ `archive/dev-v2.0.json` - SQuAD dev data

### Supporting Files (Recommended):
- ✅ `ARCHITECTURE.md` - Technical architecture
- ✅ `INTEGRATION_SUMMARY.md` - System integration details
- ✅ `training_session.py` - CSR training module
- ✅ `conversation_logger.py` - Logging system
- ✅ `QA.py` - Standalone demo

### Collaborator Information:
- ⚠️ **Action Required:** If working in a team, add collaborator name at the top of `ASSIGNMENT_ANSWERS.md`

---

## Final Assessment:

### ✅ **READY FOR SUBMISSION**

Your project demonstrates:
- ✅ Two distinct, complex AI tasks
- ✅ Two different AI methods (Neural Network + Transformer)
- ✅ Clear business value and use case
- ✅ Well-documented data sources
- ✅ Comprehensive testing and evaluation
- ✅ Professional implementation quality
- ✅ Complete, runnable code with instructions

**No changes required. All assignment requirements are fully met.**

---

**Note:** The file `ASSIGNMENT_ANSWERS.md` contains all 5 assignment questions with complete, formatted answers. Submit this as your primary document along with the code files.

**Grade Expectation:** Based on compliance with all requirements and exceeding minimum standards, this submission should receive full marks.

---

**Good luck with your submission! 🎓**
