# Q2 Experiment Results: Refusal Direction Analysis

## Model: mistral_7b

### Hypothesis 1 (Harm Concept Detector): **DISCONFIRMED**
- Criterion: d(A,B)<0.2 AND d(B,F)>0.8
- Observed d(A,B): 7.1058
- Observed d(B,F): 1.576

### Hypothesis 2 (Assistance Intent Detector): **DISCONFIRMED**
- Criterion: d(A,B)>0.8 AND d(B,F)<0.2
- Observed d(A,B): 7.1058
- Observed d(B,F): 1.576

**Interpretation**: The results are mixed and do not clearly confirm either primary hypothesis. The refusal direction's behavior may be more nuanced or context-dependent.

## Model: mistral_nemo_12b

### Hypothesis 1 (Harm Concept Detector): **DISCONFIRMED**
- Criterion: d(A,B)<0.2 AND d(B,F)>0.8
- Observed d(A,B): 3.2921
- Observed d(B,F): 2.6401

### Hypothesis 2 (Assistance Intent Detector): **DISCONFIRMED**
- Criterion: d(A,B)>0.8 AND d(B,F)<0.2
- Observed d(A,B): 3.2921
- Observed d(B,F): 2.6401

**Interpretation**: The results are mixed and do not clearly confirm either primary hypothesis. The refusal direction's behavior may be more nuanced or context-dependent.
