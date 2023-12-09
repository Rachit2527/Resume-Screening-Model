import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.porter import PorterStemmer
import re

# Load the pre-trained model and vectorizer
rf = RandomForestClassifier()
vect = TfidfVectorizer(stop_words='english')

# Load the dataset
data = pd.read_csv('resume_dataset.csv')

# Preprocess text
ps = PorterStemmer()
def preprocess_text(text):
    text = " ".join([ps.stem(word) for word in text.split()])
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

data['Resume'] = data['Resume'].apply(preprocess_text)

# Encode labels
from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
data['Category'] = en.fit_transform(data['Category'])

# Fit the vectorizer
x = data['Resume']
y = data['Category']
x = vect.fit_transform(x)

# Fit the model
rf.fit(x, y)

# Streamlit app
st.title("Resume Category Prediction App")

# Input text
resume_input = st.text_area("Enter Resume:", "")

# Button to trigger prediction
if st.button("Predict Category"):
    # Preprocess input text
    input_feature = vect.transform([preprocess_text(resume_input)])

    # Predict category
    prediction = en.inverse_transform(rf.predict(input_feature))[0]

    # Display predicted category
    st.success(f"Predicted Category: {prediction}")
