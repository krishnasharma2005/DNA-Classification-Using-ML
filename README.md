# 🧬 DNA Classification Using Machine Learning

## Project Overview

This project demonstrates the use of **machine learning** to classify DNA promoter sequences. Promoter recognition is critical in genetics for understanding gene regulation and predicting diseases.
With the growing volume of genetic data, manual classification is no longer feasible. Our approach applies supervised learning algorithms to automate the classification of DNA sequences.

## Key Objectives

1. Preprocess raw DNA promoter sequences into a machine-readable format.  
2. Train and compare multiple ML models for classification.  
3. Evaluate model performance using accuracy and visualization.  
4. Provide a lightweight, reproducible pipeline for bioinformatics tasks.

---

## Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Promoter+Gene+Sequences)
- **Type**: DNA promoter sequences  
- **Classes**: Promoter (+1) / Non-promoter (-1)  
- **Encoding**: One-hot encoding of nucleotides (A, T, G, C) → 228 features per sequence

---

## Preprocessing Pipeline

1. Load DNA sequence dataset (.data format)
2. Convert sequences to individual nucleotides
3. One-hot encode each nucleotide:
   - A → [1, 0, 0, 0]  
   - T → [0, 1, 0, 0]  
   - G → [0, 0, 1, 0]  
   - C → [0, 0, 0, 1]
4. Label encoding: +1 → 1, -1 → 0

---

## Machine Learning Models Used

| Model               | Accuracy (%) |
|--------------------|--------------|
| SVM (Linear)       | **96.29**    |
| Neural Network (MLP)| 92.59       |
| Naive Bayes        | 92.59        |
| AdaBoost           | 85.19        |
| K-Nearest Neighbors| 77.77        |
| Decision Tree      | 77.77        |
| Random Forest      | 51.80        |

---

## Results and Analysis

- **Best Model**: SVM (Linear Kernel) with **96.29%** accuracy  
- **Importance of Encoding**: One-hot encoding proved essential for transforming DNA into numerical input  
- **Discussion**:
  - Naive Bayes and Neural Networks also showed strong performance
  - Random Forest struggled due to feature dependencies
  - Class imbalance and feature redundancy influenced model performance

---

## Literature Survey Highlights

- Deep learning models can outperform classical ML but require high computation
- Alignment-free methods (e.g., ML-DSP) offer faster classification
- DNA classification is enhanced with preprocessing and feature engineering

---

## Tools and Libraries

- Python  
- Scikit-learn  
- NumPy, Pandas  
- Matplotlib, Seaborn  

---

## Conclusion

This project presents a simple yet effective ML pipeline for classifying DNA sequences, demonstrating how traditional machine learning models can support genetic research. It lays the groundwork for future work in mutation detection, cancer prediction, and large-scale genomics.
