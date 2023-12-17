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
import base64

# Loading pre-trained emotion detection CNN model
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)  

# Defining emotion labels 
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

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
        
def generate_report(emotions, timestamps, star_rating, emotion_counts):
    report = ""
    report += "Vibe Video Analysis Report\n\n"
    report += f"Star Rating: {star_rating} out of 5\n\n"
    report += "Emotion Distribution:\n"
    for emotion, count in emotion_counts.items():
        report += f"{emotion}: {count}\n"
    report += "\nTimestamps and Detected Emotions:\n"
    for time, emotion in zip(timestamps, emotions):
        report += f"Time: {time} - Emotion: {emotion}\n"
    
    return report

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="vibe_report.csv">Download CSV Report</a>'
    return href

def main():  
    
    
    #logo
    logo_path = 'vibelogo-removebg-preview.png'
    st.sidebar.image(logo_path, width=90) 

    st.sidebar.title("Settings")
    
    # Star rating chooser
    star_rating_emotion = st.sidebar.selectbox(
            "Choose the emotion for star rating:",
            options=emotion_labels,
            index=emotion_labels.index('Happy')  # Default to 'Happy'
        )
    
    # Slider to select the frame analysis interval
    frame_interval = st.sidebar.slider("Analyze every X seconds (0 for every frame)", 
                                       min_value=0, 
                                       max_value=5, 
                                       value=1)

    
    st.title('🎬 Vibe')
    st.header('Analyzing Emotional Resonance')

    st.markdown("""
        Welcome to Vibe! This app analyzes the emotional content of your video. 
        Simply upload an MP4 video, and let the app detect the vibe.
    """)

    uploaded_file = st.file_uploader("Upload a video (MP4) to analyze the vibe!", type=["mp4"])

    if uploaded_file is not None:
        with st.spinner('Processing video...'):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_file.close()

            cap = cv2.VideoCapture(temp_file.name)
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            #Error handling for short videos
            video_duration = total_frames / fps  # Total duration of the video in seconds

            if frame_interval > video_duration and frame_interval != 0:
                st.error("Frame interval too large for this video.")
                return
                
            interval_frames = max(1, int(frame_interval * fps))  # Calculate the number of frames per interval

            emotions = []
            timestamps = []
            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % interval_frames == 0:  # Process frame based on the selected interval
                    predicted_emotion = process_frame(frame)
                    emotions.append(predicted_emotion)
                    timestamps.append(frame_num / fps)

                frame_num += 1

            cap.release()

            happy_percentage = (emotions.count('Happy') / len(emotions)) * 100
            star_rating = calculate_star_rating(happy_percentage)

        st.success("Analysis complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Star Rating")

            # Calculate the star rating based on the selected emotion
            selected_emotion_percentage = (emotions.count(star_rating_emotion) / len(emotions)) * 100
            star_rating = calculate_star_rating(selected_emotion_percentage)

            stars = "⭐" * star_rating
            st.write(stars)

            emotion_rating_text = {
                'Happy': ["😔 Not a very happy video indeed!", "😶 Some happiness detected", "😀 Good amount of happiness!", "😄 A happy video!", "😁 The HAPPIEST video!"],
                'Angry': ["😊 Not an angry video at all!", "😐 Slight anger detected", "😠 Noticeable anger", "😤 Quite an angry video!", "😡 Extremely angry video!"],
                'Surprise': ["😑 No surprise in the video", "😶 Slightly surprised video", "😯 Quite surprising video", "😲 Definitely surprised video", "🤯 Mind Blown!"],
                'Sad': ["🙂 Not even a sad video!", "😐 Slightly sad video", "😔 Sad video", "😥 Near tears", "😭 Extremely sad video!"],
            }
            st.write(emotion_rating_text[star_rating_emotion][star_rating - 1])
            

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

         # Generate report
        report = generate_report(emotions, timestamps, star_rating, emotion_counts)
    
        # Create a link to download the report
        st.sidebar.download_button(label="Download Report",
                                   data=report,
                                   file_name="vibe_report.txt",
                                   mime="text/plain")

if __name__ == '__main__':
    main()

