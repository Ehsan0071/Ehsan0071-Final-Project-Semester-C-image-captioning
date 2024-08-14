import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib

# Load the pre-trained model, tokenizer, and configurations
@st.cache(allow_output_mutation=True)
def load_resources():
    model = load_model('image_captioning_model.keras')
    feature_extractor = load_model('feature_extractor.keras')
    tokenizer = joblib.load('tokenizer.pkl')
    config = joblib.load('config.pkl')
    return model, feature_extractor, tokenizer, config

model, feature_extractor, tokenizer, config = load_resources()
max_length = config['max_length']
vocab_size = config['vocab_size']

# Function to preprocess the image for the VGG16 model
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image /= 255.0
    return image

# Function to generate a caption for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return ' '.join(in_text.split()[1:-1])

# Streamlit app
st.title("Image Captioning Web App")
st.write("Upload an image, and the model will generate a caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Generating caption...")
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Extract features using the pre-trained VGG16 model
    features = feature_extractor.predict(preprocessed_image)
    
    # Generate a caption for the image
    caption = generate_desc(model, tokenizer, features, max_length)
    
    st.write("Caption: ", caption)

st.write("This app uses a CNN-LSTM model trained on the Flickr8k dataset to generate captions for images. The model is based on the VGG16 architecture for feature extraction and an LSTM network for sequence generation.")
