import os
import datetime
import requests
import streamlit as st
from transformers import pipeline
from gtts import gTTS
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from PIL import Image
from sklearn.cluster import KMeans
import cv2
import numpy as np
import plotly.graph_objects as go

# Hugging Face API Token
HUGGINGFACE_API_TOKEN = "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"

# Function to perform image captioning
def generate_image_caption(image_path):
    try:
        st.write("Generating detailed caption for the uploaded image...")
        caption_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        caption = caption_model(image_path)[0]["generated_text"]
        st.success("Image caption generated successfully!")
        return caption
    except Exception as e:
        st.error(f"Error during image caption generation: {e}")
        return None

# Function to perform text-to-speech using Hugging Face TTS API
def text2speech_huggingface(text, folder_name="items"):
    try:
        st.write("Converting caption to audio using Hugging Face TTS...")
        API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        payload = {"inputs": text}

        # Make API request
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            audio_file = f"{folder_name}/audio_{now}.flac"
            os.makedirs(folder_name, exist_ok=True)
            with open(audio_file, "wb") as file:
                file.write(response.content)
            st.success("Audio file generated successfully!")
            return audio_file
        else:
            st.error(f"Hugging Face TTS API Error: {response.json()}")
            return None
    except Exception as e:
        st.error(f"Error during Hugging Face TTS: {e}")
        return None

# Fallback function to perform text-to-speech using gTTS
def text2speech_gtts(text, folder_name="items"):
    try:
        st.write("Converting caption to audio using gTTS as fallback...")
        tts = gTTS(text)
        now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        audio_file = f"{folder_name}/audio_{now}.mp3"
        os.makedirs(folder_name, exist_ok=True)
        tts.save(audio_file)
        st.success("Audio file generated successfully using gTTS!")
        return audio_file
    except Exception as e:
        st.error(f"Error during gTTS TTS: {e}")
        return None

# Visualization functions
def display_image_metadata(image_path):
    img = Image.open(image_path)
    st.write("### Image Metadata")
    st.write(f"**Format:** {img.format}")
    st.write(f"**Size (Width x Height):** {img.size}")
    st.write(f"**Mode:** {img.mode}")

def plot_word_frequency(caption):
    words = caption.split()
    word_counts = Counter(words)
    df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(df["Word"], df["Frequency"], color="skyblue")
    plt.xticks(rotation=45)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Word Frequency in Generated Caption")
    st.pyplot(plt)

def plot_word_cloud(caption):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(caption)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def plot_dominant_colors(image_path, n_colors=5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    plt.figure(figsize=(10, 2))
    plt.title("Dominant Colors")
    plt.axis("off")
    plt.imshow([colors])
    st.pyplot(plt)

def analyze_sentiment(caption):
    sentiment_model = pipeline("sentiment-analysis")
    sentiment = sentiment_model(caption)[0]
    st.write("### Sentiment Analysis")
    st.write(f"**Label:** {sentiment['label']}")
    st.write(f"**Confidence:** {sentiment['score']:.2f}")
    labels = ["Positive", "Negative"] if sentiment['label'] == "POSITIVE" else ["Negative", "Positive"]
    sizes = [sentiment['score'], 1 - sentiment['score']]
    colors = ["green", "red"]
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")
    st.pyplot(plt)

def analyze_caption_complexity(caption):
    words = caption.split()
    word_count = len(words)
    char_count = sum(len(word) for word in words)
    avg_word_length = char_count / word_count if word_count > 0 else 0
    st.write("### Caption Complexity Analysis")
    st.write(f"**Word Count:** {word_count}")
    st.write(f"**Character Count:** {char_count}")
    st.write(f"**Average Word Length:** {avg_word_length:.2f}")
    metrics = ["Word Count", "Character Count", "Avg Word Length"]
    values = [word_count, char_count, avg_word_length]
    plt.figure(figsize=(7, 4))
    plt.bar(metrics, values, color="skyblue")
    plt.title("Caption Complexity Metrics")
    st.pyplot(plt)

def compare_image_caption(image_path, caption):
    img = Image.open(image_path)
    img_width, img_height = img.size
    caption_length = len(caption.split())
    fig = go.Figure(data=[
        go.Bar(name='Image Width', x=['Image'], y=[img_width]),
        go.Bar(name='Image Height', x=['Image'], y=[img_height]),
        go.Bar(name='Caption Word Count', x=['Caption'], y=[caption_length])
    ])
    fig.update_layout(title="Image Dimensions vs Caption Word Count",
                      barmode='group', xaxis_title="Metric", yaxis_title="Value")
    st.plotly_chart(fig)

# Streamlit App
st.title("Enhanced Image to Audio Caption with Visualizations")

# Sidebar with Hamburger Menu
menu = st.sidebar.radio("Choose an option", ["Upload Image", "Visualizations"])

if menu == "Upload Image":
    uploaded_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.read())
        
        st.image("temp_image.jpg", caption="Uploaded Image", use_column_width=True)
        display_image_metadata("temp_image.jpg")
        
        caption = generate_image_caption("temp_image.jpg")
        if caption:
            st.write(f"Generated Caption: {caption}")
            audio_file = text2speech_huggingface(caption)
            if not audio_file:
                st.warning("Hugging Face TTS failed. Falling back to gTTS...")
                audio_file = text2speech_gtts(caption)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")

elif menu == "Visualizations":
    uploaded_image = st.file_uploader("Choose an image for visualization:", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.read())

        caption = generate_image_caption("temp_image.jpg")
        if caption:
            plot_word_frequency(caption)
            plot_word_cloud(caption)
            plot_dominant_colors("temp_image.jpg")
            analyze_sentiment(caption)
            analyze_caption_complexity(caption)
            compare_image_caption("temp_image.jpg", caption)
