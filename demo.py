import streamlit as st
import cv2
from anomaly_detection import discriminator  # Import the required function
import time  # Import time module

# Initialize the fire detection model (if required)
fire_detector_state = True

# Streamlit UI
st.title("Video Anomaly Detection")

# Upload video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

if video_file:
    # Save the video to a temporary location
    video_path = f"temp_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    # Load the video
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total Frames: {frame_count}")
    
    frame_id = 0
    progress_bar = st.progress(0)

    # Create placeholders for displaying frames, anomaly status, and processing time
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()
    processing_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Measure start time for processing
        start_time = time.time()

        # Process every 5th frame for anomaly detection
        if frame_id % 5 == 0:
            prediction =  discriminator(frame, fire_detector_state)
            if prediction == 1:
                alert_placeholder.error(f"Frame {frame_id}: ALERT: Anomaly Detected!")
            else:
                alert_placeholder.success(f"Frame {frame_id}: No Anomaly Detected")

        # Measure end time for processing
        end_time = time.time()
        processing_time = end_time - start_time

        # Display processing time in the processing placeholder
        processing_placeholder.info(f"Processing time for Frame {frame_id}: {processing_time:.2f} seconds")

        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update the frame placeholder with the current frame
        frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Frame {frame_id}")

        # Update progress bar
        progress = int((frame_id / frame_count) * 100)
        progress_bar.progress(progress)

        frame_id += 1

    cap.release()
    st.success("Video processing completed!")

