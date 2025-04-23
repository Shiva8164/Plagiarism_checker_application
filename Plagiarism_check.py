import streamlit as st
import joblib
import numpy as np
import re
import nltk
import os
import zipfile

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# === Step 1: Unzip NLTK data if not already extracted ===
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    with zipfile.ZipFile('nltk_data.zip', 'r') as zip_ref:
        zip_ref.extractall(nltk_data_dir)

# === Step 2: Set NLTK data path ===
nltk.data.path.append(nltk_data_dir)

# === Step 3: Load stopwords and lemmatizer ===
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"❌ Error loading NLTK corpora: {e}")

# === Text cleaning function ===
def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# === Load models ===
st.write("✅ App is running... Loading models...")

try:
    lgbm_model = joblib.load("lightgbm_plagiarism_model.pkl")
    pca = joblib.load("pca_model.pkl")
    st.write("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")

# === Load SBERT ===
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("✅ SentenceTransformer model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading SentenceTransformer model: {e}")

# === Main Streamlit app ===
def main():
    st.title("🔍 Plagiarism Checker")
    st.write("✅ UI loaded successfully!")

    # User input
    original_text = st.text_area("Enter Original Text:")
    new_text = st.text_area("Enter New Text to Check:")

    if st.button("Check Plagiarism"):
        if original_text and new_text:
            st.write("✅ Processing input...")

            # Clean text
            original_cleaned = cleaning_data(original_text)
            new_cleaned = cleaning_data(new_text)

            try:
                # SBERT encoding
                embedded_original = sbert_model.encode([original_cleaned])[0]
                embedded_new = sbert_model.encode([new_cleaned])[0]
                st.write("✅ Text encoded successfully!")
            except Exception as e:
                st.error(f"❌ Error in text encoding: {e}")
                return

            try:
                # LightGBM prediction
                X_input = np.abs(embedded_original - embedded_new).reshape(1, -1)
                X_input_pca = pca.transform(X_input)
                prediction = lgbm_model.predict(X_input_pca)
                result = "Plagiarized" if prediction[0] == 1 else "Not Plagiarized"

                if result == "Plagiarized":
                    st.error(f"⚠️ Plagiarized!")
                else:
                    st.success(f"✅ Not Plagiarized!")
            except Exception as e:
                st.error(f"❌ Error in model prediction: {e}")

if __name__ == "__main__":
    main()
