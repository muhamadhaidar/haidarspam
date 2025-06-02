import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📨 SMS Spam Detection")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df = pd.read_csv("SMSSpamCollection.txt", sep="\t", header=None, names=["label", "message"])
    return df

# Fungsi untuk melatih dan menyimpan pipeline
@st.cache_resource
def train_and_save_model(df):
    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, "spam_pipeline.pkl")
    return pipeline

# Cek apakah model sudah ada
if os.path.exists("spam_pipeline.pkl"):
    model = joblib.load("spam_pipeline.pkl")
else:
    df = load_data()
    model = train_and_save_model(df)

# Input pengguna
st.subheader("Cek apakah SMS termasuk spam:")
user_input = st.text_area("Masukkan pesan SMS:")

if st.button("Deteksi"):
    if user_input.strip() == "":
        st.warning("Tolong masukkan teks.")
    else:
        pred = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]

        label = "🚫 Spam" if pred == 1 else "✅ Bukan Spam"
        st.success(f"Hasil Deteksi: {label}")
        st.write(f"Probabilitas Spam: {proba[1]:.2f}")
