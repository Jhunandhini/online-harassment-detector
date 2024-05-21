from flask import Flask, request, render_template, url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
import nltk
import re, string
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download NLTK data
nltk.download("stopwords")
nltk.download("punkt")

app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
model_file = "best_rfc_model.pkl"
vectorizer_file = "tfidf_vectorizer.pkl"
rfc = joblib.load(model_file)
tfidf_vectorizer = joblib.load(vectorizer_file)

# Preprocessing functions
def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', text)

def remove_stopwords(text):
    stop = stopwords.words("english")
    return " ".join([word for word in text.split() if word not in stop])

porter_stemmer = PorterStemmer()

# Function to preprocess and predict new input
def preprocess_and_predict(input_text):
    input_text = remove_punctuation(input_text)
    input_text = remove_stopwords(input_text)
    input_text = " ".join([porter_stemmer.stem(word) for word in input_text.split()])
    input_tfidf = tfidf_vectorizer.transform([input_text])
    prediction = rfc.predict(input_tfidf.toarray())
    return prediction[0]

# Load model evaluation metrics
model_metrics = joblib.load("model_metrics.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    prediction = preprocess_and_predict(comment)
    result = "That's an Abusive Statement" if prediction == 1 else "Not Abusive"
    return render_template('result.html', comment=comment, prediction=result)

@app.route('/performance')
def performance():
    return render_template('performance.html', accuracy=model_metrics['accuracy'], confusion_matrix=model_metrics['confusion_matrix'])

if __name__ == '__main__':
    app.run(debug=False)
