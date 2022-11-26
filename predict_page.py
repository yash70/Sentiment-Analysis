import streamlit as st
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

df = pd.read_csv('IMDB Dataset.csv')
X = df['review']
new_data = tf.fit_transform(X)

def load_model():
    with open('final_model.pkl', 'rb') as file:
        model = joblib.load(file)
    return model


model = load_model()


def show_predict_page():
    st.title("Sentiment Analysis of Movie Reviews")

    name = st.text_input('Hey, what should we call you?')

    st.write("### Enter Your Review Here:")
    review = st.text_input('')

    button = st.button('Classify My Review!')

    if button is True:
        review = [review]
        review = tf.transform(review)
        ans = model.predict(review)

        if ans == 0:
            st.subheader(f'Hey {name}, the review you have entered is: Negative')

        else:
            st.subheader(f'Hey {name}, the review you have entered is: Positive')


