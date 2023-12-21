import streamlit as st
import pickle
import tensorflow as tf
from transformers import AutoTokenizer, TFAlbertModel
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

# Function to load the model
def load_model(model_name):
    with open(model_name, 'rb') as file:
        return pickle.load(file)
    
# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_and_lemmatize(text):
    stop_words = set(stopwords.words('english'))
    additional_stop_words = ['subject', 're', 'edu', 'use', 'http', 'https', 'www', 'html', 'index', 'com', 'net', 'org', 'ect', 'hou', 'cc', 'recipient', 'na', 'pm', 'am', 'et', 'enron']
    stop_words = stop_words.union(additional_stop_words)

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 3 ]
    return ' '.join(filtered_words)

def remove_html_tags(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
    
def preprocess_input(body, sender, subject):
    subject = subject.lower()
    body = body.lower()
    sender = sender.lower()

    subject = re.sub('^re:', '', subject, flags=re.IGNORECASE)
    subject = re.sub('^fwd:', '', subject, flags=re.IGNORECASE)

    sender = sender if '<' not in sender.strip() else sender.split('<')[0].strip()
    subject = clean_text(subject)
    body = clean_text(body)

    body = tokenize_and_lemmatize(body)

    body = remove_html_tags(body)

    return sender + ' ' + subject + ' ' + body


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Streamlit page configuration
st.set_page_config(page_title="Email Phishing Detection", layout="wide")

# Title
st.title("Email Phishing Detection")

# Model Selection
model_option = st.sidebar.selectbox(
    "Choose a Model",
    ("Albert Model", "TF-IDF", "CNN")  # List other models if you have
)

# Load the selected model
if model_option == "Albert Model":
    model = tf.keras.models.load_model('albert_model_weights.h5', custom_objects={"TFAlbertModel": TFAlbertModel.from_pretrained("albert-base-v2")})
elif model_option == "CNN":
    model = tf.keras.models.load_model('cnn_model_weights.h5')
elif model_option == "TF-IDF":
    model = load_model('tf_idf.pkl')

# Email input fields
st.subheader("Email Details")
sender = st.text_input("Sender", value="", placeholder="Unknown")
subject = st.text_input("Subject", value="", placeholder="Unknown")
email_body = st.text_area("Email Body", height=250)

# Predict button
if st.button("Predict"):
    processed_input = preprocess_input(email_body, sender, subject)

    if (model_option == "Albert Model"):
        albert_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

        text = albert_tokenizer.encode_plus(processed_input, add_special_tokens=True, max_length=256, pad_to_max_length=True, return_attention_mask=True)
        text_input_ids = np.array([text['input_ids']])
        text_attention_masks = np.array([text['attention_mask']])
        prediction = model.predict([text_input_ids, text_attention_masks])
    elif (model_option == "CNN"):
        tokenizer = pickle.load(open('tokenizer_cnn.pkl', 'rb'))
        test_sequences = tokenizer.texts_to_sequences([processed_input])
        test_sequences = pad_sequences(test_sequences, maxlen=100)
        prediction = model.predict(test_sequences)
    else:
        prediction = model.predict([processed_input])

    # Display the prediction
    if prediction[0] > 0.5:
        st.error("This email is likely a phishing attempt.")
    else:
        st.success("This email seems safe.")