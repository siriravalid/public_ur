import pickle
import streamlit as st

# Load the model
model = pickle.load(open('fake_news.model.sav', 'rb'))

# Streamlit page title
st.title('Fake News Detection')

# Input for news text
news_text = st.text_area('Enter the news text')

# Prediction logic
prediction = ''

if st.button('Check News'):
    # Predict
    prediction = model.predict([news_text])
    
    # Display the result
    if prediction[0] == 1:
        prediction = 'The news is Fake'
    else:
        prediction = 'The news is Real'

st.success(prediction)
