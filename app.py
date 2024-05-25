import streamlit as st
import pickle
import re
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')
vector_form = pickle.load(open('vector.pkl', 'rb'))
loaded_model = pickle.load(open('news_model.pkl', 'rb'))

# Initialize the PorterStemmer
ps = PorterStemmer()
vectorization = TfidfVectorizer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content

def predict_fake_news(news):
    news = stemming(news)
    input_data = [news]
    transformed_input = vector_form.transform(input_data)
    prediction = loaded_model.predict(transformed_input)
    return prediction

# Streamlit UI
st.title("Fake News Classification App")

title = st.text_input("Enter the news title:")

if st.button("Predict"):
    if title:
        prediction = predict_fake_news(title)

        if prediction[0] == 0:
            st.error("Fake News Detected!")
        else:
            st.success("Real News Detected!")
    else:
        st.warning("Please enter a title for prediction.")
