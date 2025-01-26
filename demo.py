import streamlit as st
import cv2
import os
import time
from anomaly_detection import discriminator  # Import the required function
from alerting import send_email_alert_with_latest_image  # Import the alerting function

# Ensure the temporary folder exists
TEMP_FOLDER = "temp_frames"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Streamlit UI
st.title("Video Anomaly Detection")

# File uploader
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

if video_file:
    # Save the video to a temporary file
    video_path = f"temp_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error loading video. Please try another file.")
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Total Frames: {frame_count}")

        # Initialize progress and placeholders
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        processing_placeholder = st.empty()

        frame_id = 0
        frame_interval = 5  # Hardcoded frame interval
        previous_frame_path = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            if frame_id % frame_interval == 0:
                # Save the current frame to the temporary folder
                current_frame_path = os.path.join(TEMP_FOLDER, f"frame_{frame_id}.jpg")
                cv2.imwrite(current_frame_path, frame)

                # Process the frame
                prediction = discriminator(frame, fire_detector_state=True)
                if prediction == 1:
                    alert_placeholder.error(f"Frame {frame_id}: ALERT: Anomaly Detected!")
                    
                    # Send email with the latest frame
                    send_email_alert_with_latest_image(
                        subject="Anomaly Detected!",
                        body="An anomaly was detected. Please see the attached image for details.",
                        to_email="rohanrj389@gmail.com",  # Replace with your email
                        folder_path=TEMP_FOLDER  # Folder where images are stored
                    )
                else:
                    alert_placeholder.success(f"Frame {frame_id}: No Anomaly Detected")

                # Delete the previous frame, if it exists
                if previous_frame_path and os.path.exists(previous_frame_path):
                    os.remove(previous_frame_path)
                previous_frame_path = current_frame_path

            end_time = time.time()
            processing_time = end_time - start_time
            processing_placeholder.info(f"Processing time for Frame {frame_id}: {processing_time:.2f} seconds")

            # Convert frame to RGB and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Frame {frame_id}")

            # Update progress bar
            progress = int((frame_id / frame_count) * 100)
            progress_bar.progress(progress)

            frame_id += 1

        cap.release()
        st.success("Video processing completed!")

        # Clean up temporary files and folder
        if previous_frame_path and os.path.exists(previous_frame_path):
            os.remove(previous_frame_path)
        os.rmdir(TEMP_FOLDER)

        os.remove(video_path)
