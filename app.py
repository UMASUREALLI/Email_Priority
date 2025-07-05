
import streamlit as st
import pickle
import nltk
import string
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained model and preprocessors
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Clean and preprocess the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}0-9]", " ", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Email Priority Classifier", page_icon="📬")
st.title("📧 Email Priority Predictor")
st.markdown("Paste the **Subject** and **Body** of your email, and I'll predict its priority!")

subject = st.text_input("📌 Subject")
body = st.text_area("📝 Email Body")

if st.button("🚀 Predict Priority"):
    full_text = subject + " " + body
    cleaned = clean_text(full_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = rf_model.predict(vectorized)
    label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🟢 **Predicted Priority:** `{label}`")
