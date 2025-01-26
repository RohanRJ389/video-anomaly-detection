from anomaly import load_model, predict_single_input  # Import LSTM utilities
import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import random
import torch
from feature_extraction import preprocess_image
# Import necessary libraries
from ultralytics import YOLO  # type: ignore # YOLOv8 framework
from anomaly_detection import discriminator
import time


# Load annotation file
def load_annotations(file_path):
    annotations = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            video_name = parts[0]
            event = parts[1]
            frames = list(map(int, parts[2:]))
            annotations[video_name] = {
                'event': event,
                'frames': frames
            }
    return annotations

# Generate labels for frames based on annotations
def generate_frame_labels(total_frames, annotations):
    labels = np.zeros(total_frames, dtype=int)  # Default to 0 (normal)
    for i in range(0, len(annotations['frames']), 2):
        start_frame = annotations['frames'][i]
        end_frame = annotations['frames'][i+1]
        if start_frame != -1 and end_frame != -1:
            labels[start_frame:end_frame+1] = 1  # Mark anomalous frames
    return labels


import cv2


# Define thresholds for fire and smoke confidence
FIRE_THRESHOLD = 0.6  # Confidence threshold for fire detection
SMOKE_THRESHOLD = 0.5  # Confidence threshold for smoke detection







device = "cuda" if torch.cuda.is_available() else "cpu"


# yolo_model


# Sliding window size
WINDOW_SIZE = 6
sliding_window = []  # Initialize the sliding window as a global or persistent list

import random
import torch

# Define constants
WINDOW_SIZE = 6  # Number of frames in the sliding window
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the LSTM model
model_path = "best_for_now.pth"  # Update with your model's file path
lstm_model = load_model(model_path, device=DEVICE)

# Process videos and calculate metrics
def process_videos(video_folder, results_folder, annotation_file, fire_detector_state):
    """
    Process videos and calculate metrics.
    
    :param video_folder: Folder containing videos
    :param annotation_file: Path to the annotation file
    :param fire_detector_state: State of the fire detector
    """
    annotations = load_annotations(annotation_file)

    total_processing_time = 0  # To accumulate time
    total_frames_processed = 0  # To count frames
    failed_videos = []  # List to store names and reasons of videos that failed processing

    for video_name, anno in annotations.items():
        video_path = os.path.join(video_folder, video_name)
        if not os.path.exists(video_path):
           # print(f"Video {video_name} not found!")
            failed_videos.append((video_name, "Video not found"))
            continue
        
        print(f"Processing video: {video_name}")
        
        try:
            y_true, y_pred, y_prob = [], [], []  # True labels, predictions, and probabilities for this video
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Generate ground truth labels
            frame_labels = generate_frame_labels(total_frames, anno)
            
            # Process every 5th frame
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 5 == 0:  # Process every 5th frame
                    # Run the discriminator
                    start_time = time.time()  # Timing starts before each call
                    prediction, probabilities = discriminator(frame, fire_detector_state)
                    end_time = time.time()
                    
                    # Calculate processing time for this frame
                    processing_time = end_time - start_time
                    total_processing_time += processing_time
                    total_frames_processed += 1

                    # Append prediction, label, and probability
                    y_pred.append(prediction)
                    y_true.append(frame_labels[frame_count])
                    y_prob.append(probabilities[1]) 
                
                frame_count += 1
            
            cap.release()
            
            # Ensure the results folder exists
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            # Save y_true, y_pred, and probabilities for this video
            npy_save_path = os.path.join(results_folder, f"{video_name}_results.npy")
            np.save(npy_save_path, {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
            
            print(f"Saved results for {video_name} to {npy_save_path}")

        except Exception as e:
            error_message = str(e)
            print(f"Error processing video {video_name}: {error_message}")
            failed_videos.append((video_name, error_message))

    # Compute average processing time at the end
    if total_frames_processed > 0:
        average_time_per_frame = total_processing_time / total_frames_processed
    else:
        average_time_per_frame = 0

    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total processing time: {total_processing_time:.4f} seconds")
    print(f"Average time per frame: {average_time_per_frame:.4f} seconds")

    # Save failed video names and reasons to a text file
    if failed_videos:
        failed_videos_path = os.path.join(results_folder, "failed_videos.txt")
        with open(failed_videos_path, 'w') as f:
            for video, reason in failed_videos:
                f.write(f"{video}: {reason}\n")
        print(f"Failed video names and reasons saved to {failed_videos_path}")



if __name__ == "__main__":
    # Set paths
    VIDEO_FOLDER = r"videos"
    RESULTS_FOLDER = r"results-with-probabilities"
    ANNOTATION_FILE = r"Temporal_Anomaly_Annotation.txt"
    fire_detector_state = True
    
    # Run processing
    process_videos(VIDEO_FOLDER,RESULTS_FOLDER, ANNOTATION_FILE, fire_detector_state)


