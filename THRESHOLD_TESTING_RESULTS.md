# Threshold Testing Results Summary

## 🎯 Tested Thresholds Comparison

We tested 16 different thresholds from 0.35 to 0.70 to find the optimal balance for your needs.

---

## 📊 Top 5 Recommended Thresholds

### 1. **0.48 - BALANCED APPROACH** ⭐ **CURRENTLY APPLIED**
**Best for: Most situations, good all-around performance**

| Metric | Value | 
|--------|-------|
| **Precision** | **57.2%** |
| **Recall** | **72.7%** |
| **F1-Score** | **64.0%** (HIGHEST) |
| **Accuracy** | 78.3% |
| False Positives | 1,016 customers |
| Missed Churners | 511 customers |
| Customers Flagged | 2,374 (33.7%) |
| **Net Profit** | **$1,380,228** |
| **ROI** | 679.2% |

**✅ Why this is recommended:**
- Best F1-Score (harmonic mean of precision & recall)
- Catches 73% of churners (1,358 out of 1,869)
- Only 20% false alarm rate
- Strong profit and ROI
- Good balance for most business scenarios

---

### 2. **0.40 - MAXIMUM PROFIT** 💰
**Best for: Pure revenue optimization**

| Metric | Value |
|--------|-------|
| Precision | 45.1% |
| **Recall** | **90.4%** |
| F1-Score | 60.2% |
| False Positives | 2,058 customers |
| Missed Churners | 179 customers |
| Customers Flagged | 3,748 (53.2%) |
| **Net Profit** | **$1,558,940** (HIGHEST) |
| ROI | 378.8% |

**Use when:**
- Have large retention budget
- Maximizing revenue is priority
- Can handle high contact volume
- Missing churners is extremely costly

---

### 3. **0.46 - HIGH RECALL** 🌐 **ORIGINAL THRESHOLD**
**Best for: Catching most churners**

| Metric | Value |
|--------|-------|
| Precision | 53.9% |
| **Recall** | **78.5%** |
| F1-Score | 63.9% |
| False Positives | 1,256 customers |
| Missed Churners | 401 customers |
| Customers Flagged | 2,724 (38.7%) |
| Net Profit | $1,460,488 |
| ROI | 581.4% |

**Use when:**
- Must catch high percentage of churners
- Have good retention budget
- False alarms acceptable
- Cast wide net approach

---

### 4. **0.55 - HIGH PRECISION** 🎯
**Best for: Avoiding false alarms**

| Metric | Value |
|--------|-------|
| **Precision** | **70.3%** |
| Recall | 44.4% |
| F1-Score | 54.4% |
| **False Positives** | **351** (LOWEST) |
| Missed Churners | 1,040 |
| Customers Flagged | 1,180 (16.8%) |
| Net Profit | $896,414 |
| **ROI** | 1,276.9% |

**Use when:**
- Limited retention budget
- Want highly reliable predictions
- Avoiding false alarms is priority
- Building customer trust

---

### 5. **0.35 - MAXIMUM RECALL** 🌐
**Best for: Catch almost everyone**

| Metric | Value |
|--------|-------|
| Precision | 40.2% |
| **Recall** | **94.4%** (HIGHEST) |
| F1-Score | 56.4% |
| False Positives | 2,623 customers |
| **Missed Churners** | **105** (LOWEST) |
| Customers Flagged | 4,387 (62.3%) |
| Net Profit | $1,532,224 |
| ROI | 292.1% |

**Use when:**
- Unlimited budget
- Absolute must catch churners
- Don't mind 60% false alarms
- Emergency situation

---

## 📈 Performance Comparison Chart

```
Threshold | Precision | Recall | F1-Score | Profit    | ROI    | Use Case
----------|-----------|--------|----------|-----------|--------|------------------
0.35      | 40.2%     | 94.4%  | 56.4%    | $1.53M    | 292%   | Catch everyone
0.40      | 45.1%     | 90.4%  | 60.2%    | $1.56M ✓  | 379%   | Max profit
0.46      | 53.9%     | 78.5%  | 63.9%    | $1.46M    | 581%   | High recall (original)
0.48 ⭐   | 57.2%     | 72.7%  | 64.0% ✓  | $1.38M    | 679%   | BALANCED (current)
0.50      | 61.0%     | 66.0%  | 63.4%    | $1.28M    | 813%   | Balanced alt
0.55      | 70.3%     | 44.4%  | 54.4%    | $896K     | 1277%  | High precision
0.60      | 79.7%     | 19.1%  | 30.8%    | $398K     | 2187%  | Very conservative
```

---

## 🔄 Comparison: Before vs After Optimization

### Original (0.46) vs Current Balanced (0.48)

| Metric | Before (0.46) | After (0.48) | Change |
|--------|---------------|--------------|---------|
| **Precision** | 53.9% | **57.2%** | **+6.1%** ✅ |
| **Recall** | 78.5% | **72.7%** | **-7.4%** ⚠️ |
| **F1-Score** | 63.9% | **64.0%** | **+0.2%** ✅ |
| **Accuracy** | 76.5% | **78.3%** | **+2.4%** ✅ |
| False Positives | 1,256 | **1,016** | **-240** ✅ |
| Missed Churners | 401 | 511 | +110 ⚠️ |
| Customers Flagged | 2,724 | 2,374 | -350 |
| **Net Profit** | $1.46M | **$1.38M** | -$80K |

### Key Insights:
- ✅ **Improved precision** - fewer false alarms (240 fewer)
- ✅ **Better F1-Score** - slightly more balanced
- ✅ **Higher accuracy** - overall better predictions
- ⚠️ **Slightly lower recall** - miss 110 more churners
- ⚠️ **Slightly lower profit** - but with better efficiency

**Net Result:** More sustainable, balanced approach with better prediction quality

---

## 💡 Choosing the Right Threshold

### Decision Tree:

```
Do you have unlimited retention budget?
├─ YES → Use 0.40 (Max Profit) or 0.35 (Max Recall)
└─ NO
   │
   ├─ Is missing churners your biggest fear?
   │  ├─ YES → Use 0.46 (High Recall - Original)
   │  └─ NO
   │     │
   │     ├─ Do you want to avoid false alarms?
   │     │  ├─ YES → Use 0.55 (High Precision)
   │     │  └─ NO → Use 0.48 (BALANCED) ⭐ RECOMMENDED
   │     │
   │     └─ What's your agent capacity?
   │        ├─ Can contact 2,700+ → Use 0.46
   │        ├─ Can contact 2,000-2,500 → Use 0.48 ⭐
   │        ├─ Can contact 1,200-1,500 → Use 0.55
   │        └─ Can contact <1,000 → Use 0.60+
```

---

## 🎯 Current Configuration (0.48)

**Status:** ✅ Applied and Verified

**Performance:**
- Precision: 57.2% (7 out of 10 predictions correct)
- Recall: 72.7% (catch ~3 out of 4 churners)
- F1-Score: 64.0% (best balance)
- Contacts needed: 2,374 customers
- Expected profit: $1.38M

**This is optimal for:**
- ✅ Moderate retention budget
- ✅ Want good all-around performance
- ✅ Balance quality and quantity
- ✅ Sustainable long-term strategy
- ✅ Most business scenarios

---

## 🔧 How to Change Threshold

If you want to try a different threshold:

### Option 1: Run the testing tool again
```bash
python3 test_thresholds.py
```
Then choose from the menu.

### Option 2: Manually edit config
Edit `models/decision_threshold.json`:
```json
{
  "threshold": 0.48,  # Change this number
  "description": "Your choice"
}
```

### Option 3: Use quick apply scripts
```bash
# High recall (catch more churners)
cp models/decision_threshold_backup.json models/decision_threshold.json

# Or use Python:
python3 -c "import json; json.dump({'threshold': 0.46}, open('models/decision_threshold.json','w'))"
```

---

## 📊 Verification

After changing threshold, always verify:
```bash
python3 main.py --menu  # Option 1 - Check metrics
python3 main.py --demo  # Test with sample customers
```

---

## 🎓 Key Learnings

1. **No perfect threshold** - Always a precision vs recall trade-off
2. **0.48 is best balance** - Highest F1-Score (64.0%)
3. **Business context matters** - Choose based on your budget/priorities
4. **Can always adjust** - Monitor and fine-tune over time
5. **Test regularly** - Model performance may drift

---

## 📝 Summary Recommendations

| Your Priority | Use Threshold | Expected Results |
|---------------|---------------|------------------|
| **Best overall performance** | **0.48** ⭐ | P: 57%, R: 73%, F1: 64% |
| Maximum profit | 0.40 | P: 45%, R: 90%, Profit: $1.56M |
| Catch most churners | 0.46 | P: 54%, R: 79%, Good balance |
| Avoid false alarms | 0.55 | P: 70%, R: 44%, Very reliable |
| Catch almost everyone | 0.35 | P: 40%, R: 94%, High volume |

---

**Current Status:** ✅ Threshold 0.48 applied (Balanced approach)

**Next Steps:**
1. Monitor performance for 1-2 weeks
2. Gather agent feedback
3. Track actual retention success rates
4. Adjust if needed based on real results

---

*Generated: December 5, 2025*
*Tool: Threshold Testing & Optimization*
