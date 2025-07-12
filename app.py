import streamlit as st  # MUST be imported before set_page_config
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")  # MUST be first Streamlit command

import joblib
import re
import nltk
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(tokens)

# Load Model and Vectorizer
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

@st.cache_resource
def load_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    else:
        return None, None

model, vectorizer = load_model_and_vectorizer()

# --- Streamlit UI ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below and click **Analyze** to see if it's positive or negative.")

user_input = st.text_area("✍️ Your Review", height=150, placeholder="Type or paste your movie review here...")

if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model or vectorizer not found. Please ensure files exist in 'models/' directory.")
    else:
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0][prediction]

        sentiment_label = "😊 Positive" if prediction == 1 else "😠 Negative"
        st.success(f"**Sentiment:** {sentiment_label}")
        st.info(f"**Confidence:** {prob:.2f}")
