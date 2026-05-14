import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64

# Configure the page
st.set_page_config(
    page_title="Word Predictor",
    page_icon="🔮",
    layout="centered"
)

# Custom CSS for Glassmorphism
def local_css():
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
    }
    
    /* Target the main container to apply glassmorphism */
    [data-testid="stVerticalBlock"] {
        background: rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem !important;
        margin-top: 1rem;
    }

    /* Target inner blocks to remove double padding/background */
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] {
        background: transparent;
        box-shadow: none;
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
        border: none;
        padding: 0 !important;
    }
    
    /* Hide top header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Text Input Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px;
        color: #333 !important;
        font-size: 16px;
        padding: 10px 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6a11cb !important;
        box-shadow: 0 0 0 2px rgba(106, 17, 203, 0.2) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
    }
    
    /* Typography */
    h1 {
        text-align: center;
        color: #1e293b !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    p {
        text-align: center;
        color: #475569 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Success Message */
    [data-testid="stSuccess"] {
        background-color: rgba(76, 175, 80, 0.2) !important;
        color: #1b5e20 !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(4px);
    }
    
    </style>
    """, unsafe_allow_html=True)

local_css()

# Load models and data (cache bypassed)
@st.cache_resource
def load_assets():
    try:
        model = load_model('lstm_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('max_len.pkl', 'rb') as f:
            max_len = pickle.load(f)
        return model, tokenizer, max_len
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

model, tokenizer, max_len = load_assets()

# Predict function
def predict_next_word(model, tokenizer, max_sequence_len, text):
    try:
        # Tokenize the text
        sequence = tokenizer.texts_to_sequences([text])[0]
        # Pad sequence
        padded_sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')
        # Predict
        predicted_probs = model.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Find word from index
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                return word
        return ""
    except Exception as e:
        return ""

# UI Content
st.markdown("<h1>✨ Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p>Type a phrase and let AI predict the next word.</p>", unsafe_allow_html=True)

input_text = st.text_input("Enter your text:", placeholder="Type something here...", label_visibility="collapsed")

if st.button("Predict Next Word"):
    if input_text:
        with st.spinner("Predicting..."):
            if model and tokenizer and max_len:
                next_word = predict_next_word(model, tokenizer, max_len, input_text)
                if next_word:
                    st.success(f"**Predicted Word:** {next_word}")
                else:
                    st.warning("Could not predict the next word. Try a different phrase.")
            else:
                st.error("Models not loaded properly. Please ensure the files exist.")
    else:
        st.warning("Please enter some text first.")
