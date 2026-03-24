# 📈 Stock Trend Prediction Using News Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-NLTK-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-yellow?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Language-Bangla%20%2F%20Bengali-red?style=for-the-badge" />
</p>

> **A machine learning project that predicts stock market trends by performing sentiment analysis on Bangla (Bengali) financial news articles using Naive Bayes classifiers.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Results](#-results)
- [File Structure](#-file-structure)
- [Limitations & Future Work](#-limitations--future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project bridges the gap between **natural language processing** and **financial market prediction** for the **Bangladeshi stock market**. It analyzes Bangla-language financial news — including article titles and body text — and classifies each news item as having a **positive**, **negative**, or **neutral** sentiment label. This sentiment is then used to infer the likely direction of related stock trends.

Three flavors of Naive Bayes are evaluated and compared:
- **GaussianNB**
- **MultinomialNB**
- **BernoulliNB**

---

## 💡 Motivation

Financial markets in Bangladesh are heavily influenced by Bangla news from platforms like Prothom Alo, The Daily Star (Bangla), and DSE announcements. Most existing stock prediction tools focus on English-language news or purely numerical data. This project fills that niche by:

1. Processing **Bangla text** with custom preprocessing pipelines
2. Removing **Bangla-specific stop words**
3. Applying **Bag-of-Words vectorization** to news content
4. Classifying the sentiment impact on stock price movements

---

## ✨ Features

- ✅ **Bangla text preprocessing** — punctuation removal, special character handling, English word filtering
- ✅ **Custom Bangla stop word removal** using an external stop word CSV
- ✅ **Title + body text fusion** — combines news title and article for richer features
- ✅ **Three Naive Bayes models** trained and compared simultaneously
- ✅ **Real-time prediction** — pass any Bangla news title + body to get instant sentiment predictions from all three models
- ✅ **Google Colab ready** — designed to run seamlessly on Colab with file upload support

---

## 🏗 Project Architecture

```
Raw News (Title + Body)
         │
         ▼
  Text Preprocessing
  ┌─────────────────────────────────────┐
  │ • Combine Title + News text         │
  │ • Clean Bangla symbols (।,—,…)      │
  │ • Remove English characters         │
  │ • Remove Bangla stop words          │
  │ • Lowercase & strip punctuation     │
  └─────────────────────────────────────┘
         │
         ▼
  CountVectorizer (Bag of Words)
         │
         ▼
  Train / Test Split (80/20)
         │
    ┌────┴──────────────────────────────────┐
    ▼                   ▼                   ▼
GaussianNB         MultinomialNB       BernoulliNB
    │                   │                   │
    └───────────────────┴───────────────────┘
                        │
                        ▼
              Sentiment Prediction
           (Positive / Negative / Neutral)
```

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `Neews.csv` | Main dataset containing financial news articles in Bangla with sentiment labels |
| `stopword.csv` *(required, upload manually)* | List of Bangla stop words for preprocessing |

### Dataset Columns

| Column | Description |
|--------|-------------|
| `Title` | Headline of the financial news article |
| `News` | Full body of the news article (in Bangla) |
| `Labeling` | Sentiment label — used as the prediction target |

> **Note:** The dataset focuses on Bangladeshi financial news, particularly from the Dhaka Stock Exchange (DSE) domain.

---

## 🛠 Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| `Python 3.8+` | Core programming language |
| `pandas` | Data loading and manipulation |
| `nltk` | Natural language processing utilities |
| `scikit-learn` | ML models, vectorizer, train/test split, accuracy metrics |
| `re` | Regex-based text cleaning |
| `string` | Punctuation removal |
| `Google Colab` | Development & execution environment |
| `Jupyter Notebook` | Interactive development and experimentation |

---

## ⚙️ Installation

### Option 1: Run on Google Colab (Recommended)

1. Open the notebook file `stock_trend_prediction_using_news_analysis.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `Neews.csv` and `stopword.csv` when prompted.
3. Run all cells in order.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/nakib33/Stock-trend-prediction-using-news-analysis.git
cd Stock-trend-prediction-using-news-analysis

# Install required dependencies
pip install nltk pandas scikit-learn

# Download NLTK resources
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

Then open `stock_trend_prediction_using_news_analysis.ipynb` in Jupyter Notebook or JupyterLab.

---

## 🚀 Usage

### 1. Load and Preprocess Data

```python
import pandas as pd
df = pd.read_csv('News.csv', encoding='utf-8')
df['combined'] = df['Title'] + ' ' + df['News']
```

### 2. Train the Models

After preprocessing and vectorization, three models are trained simultaneously:

```python
gaussian_model.fit(X_train.toarray(), y_train)
multinomial_model.fit(X_train, y_train)
bernoulli_model.fit(X_train, y_train)
```

### 3. Predict on New News

```python
new_title = "your bangla news title here"
new_news  = "your bangla news body here"

sentiment_gaussian, sentiment_multinomial, sentiment_bernoulli = predict_sentiment(new_title, new_news)

print(f"GaussianNB Sentiment:    {sentiment_gaussian}")
print(f"MultinomialNB Sentiment: {sentiment_multinomial}")
print(f"BernoulliNB Sentiment:   {sentiment_bernoulli}")
```

**Example Output:**
```
GaussianNB Sentiment:    Positive
MultinomialNB Sentiment: Positive
BernoulliNB Sentiment:   Positive
```

---

## 🤖 Model Details

### Preprocessing Pipeline

| Step | Function | Description |
|------|----------|-------------|
| Symbol Cleaning | `clean_sym()` | Removes Bangla punctuation (`।`, `—`, `?`, `!`, etc.) and numbers |
| English Removal | `remove_eng()` | Strips English characters and digits from Bangla text |
| Stop Word Removal | `preprocess()` | Removes Bangla stop words loaded from `stopword.csv` |
| Lowercasing | Built-in | Converts text to lowercase for uniformity |

### Vectorization

- **Method:** `CountVectorizer` (Bag-of-Words)
- Each preprocessed document is converted into a sparse word frequency matrix.

### Classifiers

| Model | Input Format | Best For |
|-------|-------------|----------|
| `GaussianNB` | Dense array | Normally distributed continuous features |
| `MultinomialNB` | Sparse matrix | Word count / frequency features (standard NLP choice) |
| `BernoulliNB` | Sparse matrix | Binary word presence/absence features |

---

## 📊 Results

The three classifiers are evaluated using **accuracy score** on a held-out 20% test set. Results may vary depending on dataset size and quality.

```python
from sklearn.metrics import accuracy_score

print("GaussianNB Accuracy:    ", accuracy_score(y_test, y_pred_gaussian))
print("MultinomialNB Accuracy: ", accuracy_score(y_test, y_pred_multinomial))
print("BernoulliNB Accuracy:   ", accuracy_score(y_test, y_pred_bernoulli))
```

> 📝 **Tip:** MultinomialNB typically performs best on text classification tasks with word-count features.

---

## 📁 File Structure

```
Stock-trend-prediction-using-news-analysis/
│
├── Neews.csv                                          # Main labeled news dataset
├── stock_trend_prediction_using_news_analysis.ipynb  # Jupyter Notebook (interactive)
├── stock_trend_prediction_using_news_analysis (1).py # Python script version
└── README.md                                          # Project documentation
```

> ⚠️ `stopword.csv` must be uploaded manually — it is not included in the repository.

---

## ⚠️ Limitations & Future Work

### Current Limitations

- The model relies on a relatively small labeled dataset — accuracy may be limited.
- `stopword.csv` is an external dependency not versioned in the repo.
- Only Bag-of-Words features are used; word order and context are ignored.
- No deep learning or transformer-based models (e.g., BanglaBERT) are used.

### Planned Improvements

- [ ] **Add BanglaBERT / mBERT** for contextual Bangla NLP
- [ ] **Expand the dataset** with more labeled Bangla financial news
- [ ] **Include TF-IDF vectorization** alongside Bag-of-Words
- [ ] **Add evaluation metrics** — Precision, Recall, F1-score, Confusion Matrix
- [ ] **Integrate DSE stock price data** to correlate news sentiment with actual price movement
- [ ] **Build a web interface** for real-time news input and prediction
- [ ] **Version and include `stopword.csv`** in the repository directly

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** this repository
2. **Create** a new branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open a Pull Request**

Please open an issue first to discuss major changes.

---

## 📄 License

This project is open-source. Feel free to use and build upon it with attribution.

---

## 🙏 Acknowledgements

- [NLTK](https://www.nltk.org/) for natural language processing utilities
- [Scikit-learn](https://scikit-learn.org/) for ML model implementations
- [Google Colab](https://colab.research.google.com/) for the cloud-based execution environment
- Bangla financial news sources for dataset inspiration

---

<p align="center">
  Made with ❤️ for Bangla NLP & Financial Analytics
</p>
