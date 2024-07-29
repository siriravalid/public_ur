import pickle
import streamlit as st

# Load the saved model
model = pickle.load(open('fake_news_model.sav', 'rb'))

# Page title
st.title('Fake News Detection')

# Input field for news article
news_text = st.text_area("Enter the news article text here:")

# Predict button
if st.button('Predict'):
    # Transform the input text (Assuming preprocessing was done in the model training)
    # Placeholder for transformation function
    # Note: Actual transformation might be needed here depending on how the model was trained

    # Predict
    # You may need to preprocess the text if required, similar to how it was done during training
    # Assuming the model is able to handle raw input
    prediction = model.predict([news_text])
    
    if prediction[0] == 1:
        st.error("The news is Fake")
    else:
        st.success("The news is Real")
