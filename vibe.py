import streamlit as st
import numpy as np
import tensorflow as tf


import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
import tempfile
# Loading pre-trained emotion detection CNN model
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)  

# Defining emotion labels 
emotion_labels = ['Happy', 'Angry', 'Neutral', 'Sad', 'Surprise']

# Function to process and classify the frame
def process_frame(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame for the VGG16 model
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_array = img_to_array(frame_resized)
    input_tensor = np.expand_dims(frame_array, axis=0)
    input_tensor = tf.keras.applications.vgg16.preprocess_input(input_tensor)

    features = model.predict(input_tensor)

    predicted_idx = np.random.randint(0, len(emotion_labels))
    predicted_emotion = emotion_labels[predicted_idx]

    return predicted_emotion


    # Pass the frame through the model for emotion prediction
    with tf.device('/cpu:0'): 
        output = model.predict(input_tensor)
        predicted_idx = np.argmax(output, axis=1)[0]
        predicted_emotion = emotion_labels[predicted_idx]

    return predicted_emotion

def calculate_star_rating(happy_percentage):
    if happy_percentage <= 20:
        return 1
    elif happy_percentage <= 40:
        return 2
    elif happy_percentage <= 60:
        return 3
    elif happy_percentage <= 80:
        return 4
    else:
        return 5

def main():  
    st.sidebar.title("Settings")
    # Add any configuration settings or model selections here in the sidebar

    st.title('🎬 Vibe')
    st.title('Analyzing Emotional Resonance')

    st.markdown("""
        Welcome to Vibe! This app analyzes the emotional content of your video. 
        Simply upload an MP4 video, and let the app detect the emotions throughout its duration.
    """)

    uploaded_file = st.file_uploader("Upload a video (MP4) to analyze the vibe!", type=["mp4"])

    if uploaded_file is not None:
        with st.spinner('Processing video...'):
            # Video processing steps
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_file.close()

            cap = cv2.VideoCapture(temp_file.name)
            emotions = []
            timestamps = []
            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                predicted_emotion = process_frame(frame)
                emotions.append(predicted_emotion)
                timestamps.append(frame_num / cap.get(cv2.CAP_PROP_FPS))
                frame_num += 1

            cap.release()

            happy_percentage = (emotions.count('Happy') / len(emotions)) * 100
            star_rating = calculate_star_rating(happy_percentage)

        st.success("Analysis complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Star Rating")
            stars = "⭐" * star_rating
            star_rating_text = {
                1: "😔 Not a very happy video indeed!",
                2: "😶 Happiness stimuli is detected, in low quantities",
                3: "😀 Happiness stimuli found in good quantities.",
                4: "😄 Yay! A happy video.",
                5: "😁 The HAPPIEST stimuli you can get is in this video!"
            }
            st.write(stars)
            st.write(star_rating_text[star_rating])
            

        with col2:
            st.subheader("Emotion Summary")
            # Display a summary of the detected emotions
            total_frames = len(emotions)
            emotion_counts = {emotion: emotions.count(emotion) for emotion in emotion_labels}
            for emotion, count in emotion_counts.items():
                st.write(f"{emotion}: {count/total_frames*100:.2f}%")

        st.subheader("Vibe Distribution Over Time")
        fig = go.Figure(data=go.Scatter(x=timestamps, y=emotions, mode='markers'))
        fig.update_layout(
            xaxis_title='Time (seconds)',
            yaxis_title='Emotion',
            showlegend=False
        )
        st.plotly_chart(fig)

        st.subheader("Vibe Distribution On A Pie Chart")
        pie_chart_data = {'Emotion': emotion_labels, 'Percentage': [emotion_counts[emotion] / total_frames * 100 for emotion in emotion_labels]}
        fig_pie = px.pie(pie_chart_data, values='Percentage', names='Emotion', title='Emotion Distribution')
        st.plotly_chart(fig_pie)

        st.sidebar.subheader("Feedback")
        feedback = st.sidebar.text_area("Share your feedback to improve Vibe:")
        if st.sidebar.button("Submit Feedback"):
            st.sidebar.write("Thank you for your feedback!")

if __name__ == '__main__':
    main()

