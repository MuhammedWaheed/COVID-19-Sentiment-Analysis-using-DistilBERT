# 🦠 COVID-19 Sentiment Analysis using DistilBERT

## 📌 Project Overview

This project focuses on sentiment analysis of COVID-19 related tweets using a fine-tuned DistilBERT transformer model.

The goal is to classify tweets into three sentiment classes:
- **Negative**
- **Neutral**
- **Positive**

The project demonstrates a full NLP pipeline, starting from text preprocessing and classical baselines to fine-tuning a modern transformer model using Hugging Face Trainer API.

---

## 📂 Dataset

The dataset is taken from Kaggle:  
**[COVID-19 NLP Text Classification](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)**

Each tweet contains:
- `OriginalTweet` – raw tweet text
- `Sentiment` – sentiment label

Original labels include 5 classes, which were mapped into 3 classes for this project:

| Original Label     | Mapped Label |
| ------------------ | ------------ |
| Extremely Negative | Negative     |
| Negative           | Negative     |
| Neutral            | Neutral      |
| Positive           | Positive     |
| Extremely Positive | Positive     |

---

## 🔧 Data Preprocessing

The following preprocessing steps were applied:
- Convert text to lowercase
- Remove URLs
- Remove user mentions (@username)
- Remove hashtag symbols while keeping words
- Remove special characters and numbers
- Normalize whitespace

Cleaned text was stored in a new column: `clean_text`

---

## 🤖 Models Used

### 1️⃣ Baseline Model
- **TF-IDF + Logistic Regression**
- Used as a classical baseline for comparison

### 2️⃣ Final Model (Main)
- **DistilBERT** (`distilbert-base-uncased`)
- Fine-tuned for 3-class sentiment classification
- Implemented using Hugging Face Trainer API

---

## 🔍 Why DistilBERT?

- 40% fewer parameters than BERT
- ~60% faster inference
- Retains over 95% of BERT's performance
- Ideal for real-world and production use

---

## 📁 Project Structure

```
├── data/
│   ├── Corona_NLP_train.csv
│   └── Corona_NLP_test.csv
├── notebooks/
│   └── sentiment_analysis.ipynb
└── README.md
```

---

## 🧠 Key Learnings

- Transformer models significantly outperform classical NLP approaches
- DistilBERT provides an excellent balance between speed and performance
- Proper preprocessing and label mapping are critical for fair evaluation
- Hugging Face Trainer simplifies large-scale NLP experimentation

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Early stopping
- Class-weighted loss for Neutral class
- Deployment using FastAPI or Streamlit
- Multilingual sentiment analysis

---

## 👨‍💻 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Pandas
- NumPy

---

## ✅ Conclusion

This project demonstrates an end-to-end sentiment analysis pipeline using modern NLP techniques.

By leveraging DistilBERT, the model achieves strong performance while remaining computationally efficient and suitable for real-world deployment.
