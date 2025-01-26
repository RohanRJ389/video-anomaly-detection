import cv2
import torch
from ultralytics import YOLO

device =  'cuda' if torch.cuda.is_available() else 'cpu'
def preprocess_image(frame):
    """
    Preprocess the image frame for YOLO model input.
    
    Args:
        frame (ndarray): Input video frame.
    
    Returns:
        torch.Tensor: Preprocessed image ready for YOLO input.
    """
    # Resize the frame or normalize as per YOLO model's expected input
    frame_resized = cv2.resize(frame, (640, 640))  # Assuming YOLO model expects 640x640 input
    frame_normalized = frame_resized / 255.0  # Normalize pixel values between 0 and 1
    frame_tensor = torch.from_numpy(frame_normalized).float().permute(2, 0, 1).unsqueeze(0)  # Change shape to (1, 3, 640, 640)
    return frame_tensor

def extract_top_yolo_features(frame, model_path="models/best.pt", device=device):
    """
    Extract YOLO features from a single frame and return them as a numpy array.

    Args:
        frame (ndarray): Input video frame.
        model (YOLO): Preloaded YOLO model instance.
        device (str): Device to process features ('cuda' or 'cpu').

    Returns:
        np.ndarray: Extracted YOLO features for the single frame.
    """
    model=YOLO(model_path)
    # Preprocess the single frame
    model.to(device)
    input_tensor = preprocess_image(frame).to(device)

    # Forward pass through the YOLO model
    with torch.no_grad():
        raw_output = model.model(input_tensor)

    # Extract the top 10 predictions (adjust indices as per YOLO's output)
    confidence_scores = raw_output[0][:, 4, :]  # Extract confidence scores
    confidence_scores_flat = confidence_scores.flatten()
    top_10_indices = torch.topk(confidence_scores_flat, 10).indices
    top_10_predictions = raw_output[0][0, :, top_10_indices]

    # Convert extracted features into numpy
    features_array = top_10_predictions.cpu().numpy()

    return features_array
