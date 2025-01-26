
import torch
from feature_extraction import extract_top_yolo_features
from ultralytics import YOLO



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # LSTM output
        out, _ = self.lstm(x)
        # Fully connected layer on last time-step output
        out = self.fc(out[:, -1, :])
        return out

# Load the trained model
def load_model(model_path, input_size=10, hidden_size=264, num_layers=2, num_classes=2, device=device):
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_single_input(model, input_features, threshold=0.9, device='cuda'):
    """
    Predicts the output for a single input set of feature vectors using thresholding logic.

    Args:
    - model: Trained LSTM model.
    - input_features: Tensor of shape [6, 9, 10].
    - threshold: Threshold for classification (default: 0.5).
    - device: Device to run the model ('cpu' or 'cuda').

    Returns:
    - Predicted class (int).
    - Softmax probabilities (list of floats).
    """
    # Ensure the input is in the correct format
    input_features = torch.stack(input_features).unsqueeze(0)  # Shape: [1, 6, 9, 10]
    input_features = input_features.view(1, -1, 10).to(device)  # Reshape: [1, 54, 10]

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_features)
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()  # Softmax scores
        print(f"\nprobabilities: {probabilities}\n")
        # Use thresholding logic for binary classification
        print(type(probabilities[0]),type(threshold))
        predicted_class = 1 if probabilities[1] >= threshold else 0  # Binary classification threshold logic

    return predicted_class, probabilities

lstm_model_path = "best_for_now.pth" #th your model file path
lstm_model = load_model(lstm_model_path, device=device)

def anomaly_detector(frame, window=[], WINDOW_SIZE=6):
    """
    Detect anomalies in a video frame using a sliding window of features.

    Args:
        frame (ndarray): Input video frame.
        window (list): The sliding window to store frame features.

    Returns:
        int: Anomaly detection result (0 for normal, 1 for anomaly).
        list: Softmax probabilities of the prediction.
    """
    # Extract features for the current frame
    features = extract_top_yolo_features(frame)

    # Update the sliding window
    if len(window) >= WINDOW_SIZE:
        window.pop(0)  # Remove the oldest frame's features
    window.append(torch.tensor(features))  # Add the new frame's features

    # Check if the sliding window is ready for prediction
    if len(window) < WINDOW_SIZE:
        # print("Sliding window not full. Waiting for more frames.")
        return 0, [1.0, 0.0]  # Default to normal until the window is full

    # Make predictions using the LSTM model
    predicted_class, probabilities = predict_single_input(lstm_model, window)

    # Debugging output for sliding window state
    print(f"Sliding window size: {len(window)}")
    print(f"Current window content shapes: {[f.shape for f in window]}")

    return predicted_class, probabilities



def detect_fire_smoke(frame, model, FIRE_THRESHOLD=0.5, SMOKE_THRESHOLD=0.5):
    """
    Detects fire or smoke in a video frame using YOLOv8 model.

    Args:
        frame (numpy.ndarray): The video frame (image) to analyze.
        model: YOLOv8 model loaded using Ultralytics framework.

    Returns:
        int: 1 if fire or smoke confidence exceeds thresholds, else 0.
    """
    if frame is None:
        print("Error: Invalid frame provided.")
        return 0  # Return normal if no valid frame
    
    # Run YOLOv8 inference
    results = model(frame)  # Run the model on the frame
    
    # Process results
    for result in results:
        boxes = result.boxes  # Extract bounding boxes
        for box in boxes:
            confidence = box.conf.item()  # Confidence score
            cls = int(box.cls.item())  # Class index
            label = model.names[cls]  # Class label
            
            # Check if detection is fire or smoke and meets thresholds
            if label.lower() == "fire" and confidence >= FIRE_THRESHOLD:
               # print(f"Fire detected with confidence: {confidence:.2f}")
                return 1  # Return 1 if fire is detected
            elif label.lower() == "smoke" and confidence >= SMOKE_THRESHOLD:
                print(f"Smoke detected with confidence: {confidence:.2f}")
                return 1  # Return 1 if smoke is detected

    # If no fire or smoke detected with sufficient confidence
    return 0


# Example usage of detect_fire_smoke in the discriminator
import time  # Import for timing the processing

fire_model_path = r"fire_smoke.pt"  # Replace with the path to your .pt file
fire_model = YOLO(fire_model_path)


# Example usage of detect_fire_smoke in the discriminator with timing
def discriminator(frame, fire_detector_state):
    """
    Runs the fire detection or anomaly detection and records processing time.
    
    Args:
        frame (ndarray): Video frame input.
        fire_detector_state (bool): State of the fire detector
    
    Returns:
        int: The final prediction after processing.
    """
    start_time = time.time()  # Record start time
    # Assuming 'model' is the YOLOv8 model already loaded
    if fire_detector_state:
        prediction = detect_fire_smoke(frame, fire_model)
        if prediction:
            end_time = time.time()  # Record end time
            processing_time = end_time - start_time
            print(f"Time taken for fire detection: {processing_time:.4f} seconds")
            return prediction

    # Call the anomaly detection logic
    prediction, prob = anomaly_detector(frame)
    end_time = time.time()  # Record end time
    processing_time = end_time - start_time
    print(f"Time taken for anomaly detection: {processing_time:.4f} seconds")
    print(f"\n\n{prob}\n")
    return prediction,prob
