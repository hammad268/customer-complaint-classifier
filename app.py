import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

model = joblib.load("complaint_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

st.set_page_config(
    page_title="Customer Complaint Classifier",
    page_icon="📋",
    layout="centered"
)

st.title("📋 Customer Complaint Classifier")
st.write("Enter a customer complaint and the model will predict its category.")

complaint = st.text_area(
    "Enter your complaint",
    placeholder="Example: My credit card was charged twice and I want a refund.",
    height=150
)

if st.button("Classify Complaint", use_container_width=True):
    if complaint.strip():

        cleaned = clean_text(complaint)

        vectorized = tfidf.transform([cleaned])

        prediction = model.predict(vectorized)

        category = prediction[0]

        st.success(f"Predicted Category: {category}")

    else:
        st.warning("Please enter a complaint.")