# Sentiment Analysis on Twitter Data using Machine Learning

Sentiment Analysis is a Natural Language Processing (NLP) technique used to determine whether a piece of text expresses a positive, negative, or neutral sentiment. It is widely applied in analyzing opinions, reviews, feedback, and social media content to understand public perception of a product, service, brand, or event.

This project implements **binary sentiment analysis** on Twitter data using **Natural Language Processing (NLP)** and a **Logistic Regression** model. It classifies tweets as either **positive** or **negative** based on their content. The project is built and executed in **Google Colab**, with the **Sentiment140 dataset** sourced directly via the **Kaggle API**.

## 📁 Dataset

[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

- 📄 Contains **1.6 million labeled tweets**
  
- 🟡 Labels:
  
  - `0` → Negative sentiment
    
  - `4` → Positive sentiment (converted to `1` in this project)


## 🧠 Features

- 🔽 Kaggle API integration to download dataset
  
- 🧹 Text preprocessing (cleaning, stopword removal, stemming)
 
- 🔡 TF-IDF vectorization to convert text into numerical format
 
- 🔁 Model training using **Logistic Regression**
 
- 📊 Accuracy evaluation on both training and test sets
  
- 💾 Save & reload model using `pickle`
 
- 🔮 Make predictions on unseen tweet samples
  

## 🧪 Technologies Used

Python 3, Google Colab

NLP Libraries: NLTK, Regex

ML Libraries: Scikit-learn

Data: Sentiment140 dataset

Model Persistence: pickle

API Access: Kaggle API


## 🚀 How It Works:

Load Kaggle API key and download the Sentiment140 dataset.

Preprocess the data:

Clean text

Apply stemming

Remove stopwords

Convert text to TF-IDF vectors

Train a logistic regression model using Scikit-learn.

Evaluate model accuracy on train and test data.

Save and load the model for future predictions.


## 📂 Project Files

File	Description:

sentiment_analysis.py	Main script containing data processing, model training, etc.

kaggle.json	Kaggle API key (user-provided, do not upload to GitHub)

trained_model.sav	Pickled model file for future use

training.1600000...csv	Raw tweet dataset (from Kaggle Sentiment140)


📈 Results: 

✅ Training Accuracy: ~79.6%

✅ Testing Accuracy: ~77.6%

💬 The model performs reliably on binary sentiment classification tasks.

🗃 Sample Output:

Tweet: "I love this!" → Prediction: Positive Tweet  

Tweet: "Worst day ever..." → Prediction: Negative Tweet

🙋‍♀️ Author

Yashaswini S

B.Tech – Artificial Intelligence & Machine Learning

M S Engineering College
