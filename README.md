📰 Fake News Detection using Machine Learning

A complete end-to-end Machine Learning project that classifies news articles as Real or Fake using NLP, TF-IDF vectorization, Logistic Regression, and a fully cleaned text pipeline.

This project includes data preprocessing, EDA, model training, evaluation, and real-time predictions using a random test sample.

📌 Project Overview

Fake news has become a major issue due to misinformation spreading across the internet.
This ML project detects whether a news article is real or fake using:

Text Cleaning (regex-based)

TF-IDF Vectorization

Logistic Regression Classifier

Evaluation Metrics (Accuracy, Classification Report, Confusion Matrix)

Random Test Sample Prediction

📂 Dataset

Dataset used: Fake & Real News Dataset (Kaggle / ISOT dataset)
It contains two files:

Fake.csv — Fake news articles

True.csv — Real news articles

We combine, shuffle, preprocess, and train the model on them.

🧹 Text Preprocessing

Custom wordopt() function performs:

Lowercasing

Removing links

Removing HTML tags

Removing punctuation

Removing numbers

Removing non-word characters

Removing newline characters

Removing words inside brackets

Optional stopword removal

Cleaned text is used for TF-IDF vectorization.

🧠 Model Used
TF-IDF Vectorizer

Converts text into numerical form based on word importance.

Logistic Regression

Chosen because:

Works extremely well for text classification

Fast & lightweight

Achieves 95%–98% accuracy

📊 Evaluation Metrics

The model is evaluated on:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

🎯 Random Test Sample Prediction

The notebook includes a feature to:

Pick a random news article from the test set

Display the news text

Predict whether it's Real / Fake

Show the actual label

This mimics real-world testing.

🛠️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn

Regex (re)

Jupyter Notebook

▶️ How to Run the Project

Clone the repository

Install dependencies

Open the Jupyter Notebook

Run all cells

Observe model evaluation

Test on random samples

📌 Folder Structure
📁 Fake-News-Detection/
│── fake_news_detection.ipynb
│── Fake.csv
│── True.csv
│── README.md
│── model.pkl (optional)
│── vectorizer.pkl (optional)

📈 Results

Achieved 95%+ accuracy

Logistic Regression outperformed Naive Bayes

Clean TF-IDF features helped the model capture writing patterns

Excellent performance on unseen test samples
