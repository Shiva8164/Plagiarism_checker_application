A **Plagiarism Checker** using **Machine Learning (LightGBM + PCA)** and **SBERT sentence embeddings** to detect similarity between two texts.

![Plagiarism Checker](https://raw.githubusercontent.com/Shiva8164/Plagiarism_Checker_ai/main/Plagerised_image.png)

---

## 🚀 Features
✔ **TF-IDF & SBERT Approaches** to text similarity detection  
✔ **LightGBM Model** for high-speed plagiarism detection  
✔ **PCA (768 → 100 components)** for dimensionality reduction & faster processing  
✔ **Streamlit Web UI** for easy text input and checking  
✔ **Stopword Removal & Lemmatization** for better text preprocessing  
## 📊 TF-IDF vs. SBERT Approach

| **Method** | **Vectorization** | **Components** | **Logistic Regression Accuracy** | **LightGBM Accuracy** |
|------------|------------------|---------------|-------------------------|-----------------|
| **TF-IDF** | TF-IDF Vectorizer | 5000 (fixed vocab) | **62%** | **69%** |
| **SBERT + PCA** | `all-MiniLM-L6-v2` | **Reduced from 768 → 100** | **79%** | **81%** |

🔹 **Why SBERT + PCA?**  
- TF-IDF treats text as **bag-of-words**, ignoring word meaning.  
- SBERT captures **semantic similarity**, leading to **better accuracy**.  
- PCA **reduces computation time** by lowering dimensions from **768 → 100**.  

🔹 **Why LightGBM?**  
- Dataset is **large**; traditional boosting methods like XGBoost were **too slow**.  
- LightGBM handles **high-dimensional data efficiently**.  
- Faster **training and inference** than other ensemble models.

---
