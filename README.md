# 📰 Fake News Detection.

## 📌 Overview
This project aims to detect fake news articles using Natural Language Processing (NLP) and machine learning techniques. By analyzing text patterns and metadata, the model classifies news as either **real** or **fake**, helping combat misinformation in digital media.

## 🎯 Objectives
- Preprocess and clean textual data
- Extract meaningful features using TF-IDF
- Train and evaluate classification models
- Deploy a simple web interface for predictions 

## 🧠 Technologies Used
- **Python** (Scikit-learn)
- **NLP** (SpaCy)
- **Vectorization** (TF-IDF)
- **Models**: Logistic Regression, PassiveAggressiveClassifier
- **Deployment**: Streamlit 

## 📁 Dataset
- **Source**: [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
- Contains labeled news articles with title, text, and label (`FAKE` or `REAL`)

## ⚙️ Project Structure
```
fake-news-detection/
│
├── data/                  # Raw and cleaned datasets
├── notebooks/             # EDA and model training
├── src/                   # Python scripts for preprocessing and modeling
├── app/                   # Streamlit or Flask app 
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

## 🔍 Workflow
1. **Data Cleaning**: Remove punctuation, stopwords, and apply lemmatization.
2. **Feature Extraction**: Convert text to numerical vectors using TF-IDF.
3. **Model Training**: Train classifiers and evaluate accuracy, precision, recall.
4. **Deployment**: Build a user-friendly interface for real-time predictions.

## 📊 Results
- Achieved **accuracy > 90%** using Logistic Regression
- PassiveAggressiveClassifier showed strong performance on imbalanced data
- Model interprets key words contributing to classification

---

