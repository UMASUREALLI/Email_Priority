#  Email Priority Predictor

This is a Streamlit web application that automatically classifies the priority level of emails (High / Medium / Low) using machine learning and natural language processing (NLP). Users can paste the subject and body of any email, and the app will predict its urgency based on trained data.


---

##  Project Overview

Email overload is a common issue in both professional and personal communication. Prioritizing emails manually can be time-consuming and error-prone. This project solves that by:

- Using NLP techniques to clean and analyze email text
- Extracting important features using TF-IDF vectorization
- Training a Random Forest classifier to predict email importance
- Building an intuitive web interface using Streamlit

---

##  Features

- Clean and intuitive UI (built with Streamlit)
- Paste email subject + body → Get instant priority prediction
- Supports three classes: `High`, `Medium`, `Low`
- Fully trained offline — no internet or API calls needed
- Includes preprocessing pipeline with:
  - Tokenization
  - Stopword removal
  - Lemmatization

---

##  Model Pipeline

1. Text Preprocessing:
   - Lowercasing, punctuation/digit removal
   - Tokenization using NLTK
   - Stopword filtering
   - Lemmatization with WordNet

2. Feature Extraction:
   - TF-IDF Vectorization with unigrams + bigrams

3. Model:
   - Trained with `RandomForestClassifier` (100 trees)
   - Labels encoded as:
     - High → 0
     - Medium → 1
     - Low → 2

---

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/email-priority-app.git
cd email-priority-app
#Install dependencies
pip install -r requirements.txt
#Run the app
streamlit run app.py


