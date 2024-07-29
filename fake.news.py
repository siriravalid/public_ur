import pickle
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the model and vectorizer
model = pickle.load(open('fake_news_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

# Initialize stemmer
port_stem = PorterStemmer()

def preprocess_text(text):
    """Preprocess the input text."""
    # Replace non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and stem
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text

# Streamlit app
st.title('Fake News Detection')

# Input field for news article
news_text = st.text_area("Enter the news article text here:")

# Predict button
if st.button('Predict'):
    if news_text:
        # Preprocess the input text
        news_text_processed = preprocess_text(news_text)
        
        # Transform the input text using the same vectorizer
        news_text_vectorized = vectorizer.transform([news_text_processed])
        
        # Predict
        prediction = model.predict(news_text_vectorized)
        
        if prediction[0] == 1:
            st.error("The news is Fake")
        else:
            st.success("The news is Real")
    else:
        st.warning("Please enter some text")
