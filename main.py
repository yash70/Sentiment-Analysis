from flask import Flask, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

tf = TfidfVectorizer()
df = pd.read_csv('IMDB Dataset.csv')
X = df['review']
new_data = tf.fit_transform(X)


def load_model():
    with open('final_model.pkl', 'rb') as file:
        model = joblib.load(file)
    return model


model = load_model()

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')
    review = [review]
    review = tf.transform(review)
    ans = model.predict(review)

    if ans == 0:
        return 'It is a Negative Review'
    else:
        return 'It is a Positive Review'


if __name__ == '__main__':
    app.run(debug=True)
