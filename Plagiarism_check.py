import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Add path to local NLTK data directory (make sure this folder is present in your repo)
nltk.data.path.append('./nltk_data')

# Loading stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Loading trained models
st.write("✅ App is running... Loading models...")

try:
    lgbm_model = joblib.load("lightgbm_plagiarism_model.pkl")
    pca = joblib.load("pca_model.pkl")
    st.write("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")

# Loading SentenceTransformer model
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("✅ SentenceTransformer model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading SentenceTransformer model: {e}")

# Streamlit UI
def main():
    st.title("🔍 Plagiarism Checker")

    # Debugging: Show that the app UI has loaded
    st.write("✅ UI loaded successfully!")

    # Taking Input 
    original_text = st.text_area("Enter Original Text:")
    new_text = st.text_area("Enter New Text to Check:")

    if st.button("Check Plagiarism"):
        if original_text and new_text:
            st.write("✅ Processing input...")

            # Apply text cleaning
            original_cleaned = cleaning_data(original_text)
            new_cleaned = cleaning_data(new_text)

            try:
                # Converting text to embeddings
                embedded_original = sbert_model.encode([original_cleaned])[0]
                embedded_new = sbert_model.encode([new_cleaned])[0]
                st.write("✅ Text encoded successfully!")
            except Exception as e:
                st.error(f"❌ Error in text encoding: {e}")
                return

            # Using LightGBM for final check
            try:
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
