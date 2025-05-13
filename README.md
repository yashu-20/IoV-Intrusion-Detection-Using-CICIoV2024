# Robust Intrusion Detection in IoV Using PU Learning and Supervised Ensembles  
*Capstone Project | Mississippi State University | Jan 2025 – May 2025*

## The Problem: Connected Vehicles, Real Threats

Modern vehicles are no longer just machines — they're intelligent, connected systems forming the **Internet of Vehicles (IoV)**. But with connectivity comes vulnerability. 

In 2015, security researchers remotely hacked a Jeep Cherokee, manipulating its **steering, brakes, and infotainment system** via CAN Bus messages — the heart of vehicle communication. This high-profile incident made one thing clear:

> **We need smarter, scalable, and adaptive cybersecurity solutions for vehicles.**

## The Research Question

Can we **detect attacks** like **Denial-of-Service (DoS)** and **Spoofing** in CAN Bus networks **with very limited labeled attack data**?

Traditional IDS models fail here because:
- Labeling vehicular attack data is **expensive and limited**
- Class imbalance leads to **biased models**
- Real-world environments demand **semi-supervised learning**

This project tackles the challenge using a **hybrid ML framework** with:
1. **Supervised Learning**
2. **Positive-Unlabeled (PU) Learning**
3. **Synthetic Data Augmentation**
4. **A/B Testing for Statistical Validation**

---

## The Dataset: CICIoV2024

- Provided by the **Canadian Institute for Cybersecurity (CIC)**
- Simulated **DoS** and **Spoofing** attacks on a **2019 Ford CAN Bus**
- Logged traffic via **OBD-II port** under real driving and attack scenarios
- Final cleaned and deduplicated samples used:
  - **Benign**: 3,547
  - **Attack**: 41  
  - **Total**: 3,588

To overcome this data scarcity, we later generated **synthetic samples** using **Gaussian distributions** for both classes.

---

## The Models: From Supervision to Adaptation

### Supervised Models
- **Random Forest**, **XGBoost**, **SVM**, **Naive Bayes**, **Voting Classifier**
- Trained using traditional 80-20 split + 5-fold cross-validation

### PU Learning Models
- **ElkanotoPU**
- **Bagging PU**
- **PU-SVM**
- **Two-Step PU Learning**  
> Used only **positive (attack)** and **unlabeled (benign)** samples to mimic real-world constraints

### Synthetic Data Generation
- Gaussian-based sampling for every numeric feature
- 20% extra attack + benign samples to address imbalance

---

## Evaluation: Metrics & A/B Testing

We didn’t stop at F1-scores. We applied **paired t-tests** to validate improvements in:
- **F1-Score**
- **Number of Positives Predicted**

| Model                | F1 Score (Original) | F1 Score (Synthetic) |  Positives Detected |
|---------------------|---------------------|-----------------------|-----------------------|
| Voting Classifier   | 0.9937              | 0.9978                | +3.1%                |
| Two-Step PU         | 0.9833              | 0.9918                | +5.2%                |
| PU Naive Bayes      | 0.2105              | 0.9940                | **+76.3%**           |

---

## Codebase Walkthrough

```bash
 original_data.py          → Supervised + PU Learning on original deduplicated dataset
 synthetic_data.py         → Same pipeline applied on Gaussian-augmented dataset
 ABtest.py                 → Paired t-tests comparing model improvements
 comparison.py             → Metric comparison plots (F1, precision, recall)
