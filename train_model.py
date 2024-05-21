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
import joblib

# Download NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# Load data
file_path = r"C:\Users\JHUNANDHINI\OneDrive\Documents\Suspicious Communication on Social Platforms.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', text)

def remove_stopwords(text):
    stop = stopwords.words("english")
    return " ".join([word for word in text.split() if word not in stop])

df['cleaned_comments'] = df['comments'].apply(remove_punctuation).apply(remove_stopwords)

# Stemming
porter_stemmer = PorterStemmer()
df['cleaned_comments'] = df['cleaned_comments'].apply(lambda x: " ".join([porter_stemmer.stem(word) for word in x.split()]))

# Digit removal
df['cleaned_comments'] = df['cleaned_comments'].apply(lambda x: " ".join([word for word in x.split() if not word.isdigit()]))

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(use_idf=True, sublinear_tf=True)
tfidf = tfidf_vectorizer.fit_transform(df['cleaned_comments'].tolist())

# Splitting data into training and testing sets
X = tfidf.toarray()
y = np.array(df['tagging'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Addressing class imbalance with Random Over-Sampling
oversample = RandomOverSampler(sampling_strategy='not majority')
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

# Train the Random Forest Classifier
rfc = RandomForestClassifier()
print("Training the Random Forest Classifier...")
rfc.fit(X_train_over, y_train_over)
print("Training complete.")

# Evaluate the model
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and TF-IDF vectorizer
joblib.dump(rfc, "best_rfc_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Save model evaluation metrics
model_metrics = {
    'accuracy': accuracy,
    'confusion_matrix': conf_matrix
}
joblib.dump(model_metrics, "model_metrics.pkl")
