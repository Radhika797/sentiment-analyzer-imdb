import pandas as pd
import re
import nltk
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Configuration ---
DATA_DIR = "data"
MODEL_DIR = "models"
DATASET_FILE = os.path.join(DATA_DIR, "IMDB Dataset.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# --- Ensure directories exist ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- NLTK Resource Download ---
def download_nltk_resources():
    """Downloads necessary NLTK resources if not already present."""
    resources = [('corpora/stopwords', 'stopwords'),
                 ('corpora/wordnet', 'wordnet'),
                 ('corpora/omw-1.4', 'omw-1.4')]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}...")
            nltk.download(name)

download_nltk_resources()
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# --- Text Preprocessing ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(tokens)

# --- Training Function ---
def train_and_save_model():
    print("--- Training Model (this might take a few minutes) ---")
    if not os.path.exists(DATASET_FILE):
        print(f"FATAL ERROR: Dataset file not found at {DATASET_FILE}")
        print("Please download 'IMDB Dataset.csv' and place it in the 'data' folder.")
        return None, None

    print(f"Loading dataset from {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)

    print("Preprocessing text data...")
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    print("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment_label']

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', random_state=42, C=0.5)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    print(f"Saving model to {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)
    print(f"Saving vectorizer to {VECTORIZER_FILE}")
    joblib.dump(tfidf_vectorizer, VECTORIZER_FILE)

    print("--- Model Training and Saving Complete ---")
    return model, tfidf_vectorizer

# --- Prediction Function ---
def predict_sentiment(text, model, vectorizer):
    if not model or not vectorizer:
        print("Model or vectorizer not loaded. Cannot predict.")
        return "Error", 0.0

    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    probability = model.predict_proba(vectorized_text)

    sentiment_label = "Positive" if prediction[0] == 1 else "Negative"
    confidence = probability[0][prediction[0]]
    return sentiment_label, confidence

# --- Main Execution ---
if __name__ == "__main__":
    model = None
    vectorizer = None

    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        print(f"Loading pre-trained model from {MODEL_FILE}...")
        model = joblib.load(MODEL_FILE)
        print(f"Loading pre-trained vectorizer from {VECTORIZER_FILE}...")
        vectorizer = joblib.load(VECTORIZER_FILE)
        print("Model and vectorizer loaded successfully.")
    else:
        print("Pre-trained model/vectorizer not found in 'models/' folder.")
        print("Attempting to train and save a new model...")
        if not os.path.exists(DATASET_FILE):
            print(f"CRITICAL: Dataset file {DATASET_FILE} not found. Please place 'IMDB Dataset.csv' in the '{DATA_DIR}' folder and re-run.")
            exit()
        model, vectorizer = train_and_save_model()

    if model and vectorizer:
        print("\n--- Testing Prediction on Sample Reviews ---")
        test_review_positive = "This is an amazing movie, full of joy and wonderful acting. I loved it!"
        test_review_negative = "Absolutely terrible film. Boring, dull, and a complete waste of my time."

        sentiment_pos, confidence_pos = predict_sentiment(test_review_positive, model, vectorizer)
        print(f"Review: '{test_review_positive}'\nSentiment: {sentiment_pos} (Confidence: {confidence_pos:.2f})\n")

        sentiment_neg, confidence_neg = predict_sentiment(test_review_negative, model, vectorizer)
        print(f"Review: '{test_review_negative}'\nSentiment: {sentiment_neg} (Confidence: {confidence_neg:.2f})\n")

        print("Script execution finished. If models were trained, they are now saved in the 'models' folder.")
    else:
        print("Failed to load or train the model. Cannot perform predictions.")
        if not os.path.exists(DATASET_FILE):
            print(f"Ensure 'IMDB Dataset.csv' is in the '{DATA_DIR}' directory and run the script again.")
