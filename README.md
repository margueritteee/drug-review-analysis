# Drug Review Sentiment Analysis & Recommender System 💊

A machine learning project that performs sentiment analysis on drug reviews and recommends medications based on patient conditions.

## 📊 Project Overview

- **Dataset**: 362,763 WebMD drug reviews
- **Task 1**: Sentiment Analysis (Positive/Negative/Neutral classification)
- **Task 2**: Drug Recommendation System based on patient conditions

## 🎯 Results

### Sentiment Analysis Performance
- **Best Model**: Ensemble Voting (Logistic Regression + LinearSVC + Random Forest)
- **Final Accuracy**: 70.64%
- **Improvement**: +3.73% from baseline (66.91%)

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Baseline (Naive Bayes) | 66.91% | - |
| Bigrams + Logistic Regression | 70.54% | +3.63% |
| **Ensemble Voting** | **70.64%** | **+3.73%** ✓ |

### Drug Recommender System
- **Conditions Covered**: 671 medical conditions
- **Drugs Available**: 2,191 unique medications

## 🛠️ Technologies Used

- Python 3.x
- pandas, numpy, scikit-learn
- nltk, matplotlib, seaborn
- imbalanced-learn

## 📁 Project Structure

├── notebooks/
│ └── Drug_Review_Analysis.ipynb
├── models/
│ ├── sentiment_model_ensemble.pkl
│ └── vectorizer_ensemble.pkl
├── data/
│ ├── drug_recommendations.csv
│ └── webmd_drug_reviews.csv
├── visualizations/
│ ├── model_comparison.png
│ └── confusion_matrix.png
└── README.md

## 🚀 Getting Started

### Installation
pip install -r requirements.txt

### Download NLTK Data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

## 📈 Key Findings

1. Bigrams improved accuracy by 3.63%
2. Logistic Regression outperformed Naive Bayes
3. Ensemble methods provided consistent improvements
4. Neutral sentiment is challenging (only 14% of dataset)

## 📊 Visualizations

![Model Comparison](model_comparison.png)
![Confusion Matrix](confusion_matrix.png)

## 📝 Dataset

**Source**: [WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)

## 👨‍💻 Author

Margueritte - Master's 2 Student in Artificial Intelligence

## 📄 License

MIT License
