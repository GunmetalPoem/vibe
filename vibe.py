import streamlit as st
import numpy as np
import tensorflow as tf

# Loading pre-trained emotion detection CNN model
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)  

# Defining emotion labels 
emotion_labels = ['Happy', 'Angry', 'Neutral', 'Sad', 'Surprise']

# Function to process and classify the frame
def process_frame(frame):
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the frame for the model
    frame_resized = cv2.resize(frame_gray, (48, 48))
    frame_array = img_to_array(frame_resized)
    input_tensor = np.expand_dims(frame_array, axis=0)

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
    st.title('Emotion Detection from MP4 Video')

    # File uploader to upload the MP4 video
    uploaded_file = st.file_uploader("Upload a video (MP4)", type=["mp4"])

    if uploaded_file is not None:
        # Create a temporary file and write the uploaded video content to it
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # Read the video file using OpenCV
        cap = cv2.VideoCapture(temp_file.name)

        # Perform emotion detection and store emotions and timestamps in lists
        emotions = []
        timestamps = []
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            predicted_emotion = process_frame(frame)
            emotions.append(predicted_emotion)
            timestamps.append(frame_num / cap.get(cv2.CAP_PROP_FPS))
            frame_num += 1

        # Release video capture
        cap.release()

        # Calculate the star rating based on the percentage of "Happy"
        happy_percentage = (emotions.count('Happy') / len(emotions)) * 100
        star_rating = calculate_star_rating(happy_percentage)

        # Display the star rating
        st.subheader("Star Rating")

        if star_rating == 1:
          st.write("Not a very happy video indeed!")
        elif star_rating == 2:
          st.write("Happiness stimuli is detected, in low quantities")
        elif star_rating == 3:
          st.write("Happiness stimuli found in good quantities.")
        elif star_rating == 4:
          st.write("Yay! A happy video.")
        else:
          st.write("The HAPPIEST stimuli you can get is in this video!")


        stars = "⭐" * star_rating
        st.write(stars)

        # Plot emotions over time as a scatter plot using plotly
        st.subheader("Emotion Distribution Over Time")
        fig = go.Figure(data=go.Scatter(x=timestamps, y=emotions, mode='markers'))
        fig.update_layout(
            xaxis_title='Time (seconds)',
            yaxis_title='Emotion',
            showlegend=False
        )
        st.plotly_chart(fig)

        # Create pie chart for emotion percentages using plotly
        st.subheader("Emotion Distribution On A Pie Chart")
        emotion_counts = {emotion: emotions.count(emotion) for emotion in emotion_labels}
        total_frames = len(emotions)
        percentages = [emotion_counts[emotion] / total_frames * 100 for emotion in emotion_labels]
        pie_chart_data = {'Emotion': emotion_labels, 'Percentage': percentages}
        fig_pie = px.pie(pie_chart_data, values='Percentage', names='Emotion', title='Emotion Distribution')
        st.plotly_chart(fig_pie)

if __name__ == '__main__':
    main()
