import torch
import torch.nn as nn
import concurrent.futures
from feature_extraction import extract_top_yolo_features
from ultralytics import YOLO
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Fully connected layer on last time-step output
        return out

def load_model(model_path, input_size=10, hidden_size=264, num_layers=2, num_classes=2, device=device):
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_single_input(model, input_features, threshold=0.9, device='cuda'):
    input_features = torch.stack(input_features).unsqueeze(0).view(1, -1, 10).to(device)
    with torch.no_grad():
        output = model(input_features)
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        predicted_class = 1 if probabilities[1] >= threshold else 0
    return predicted_class, probabilities

lstm_model_path = "models/best_for_now.pth"
lstm_model = load_model(lstm_model_path, device=device)

def detect_fire_smoke(frame, model, FIRE_THRESHOLD=0.5, SMOKE_THRESHOLD=0.5):
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf.item()
            cls = int(box.cls.item())
            label = model.names[cls]
            if (label.lower() == "fire" and confidence >= FIRE_THRESHOLD) or \
               (label.lower() == "smoke" and confidence >= SMOKE_THRESHOLD):
                return 1
    return 0

def anomaly_detector(frame, window=[], WINDOW_SIZE=6):
    features = extract_top_yolo_features(frame)
    if features is None or len(features) == 0:
        return 0, [1.0, 0.0]

    if len(window) >= WINDOW_SIZE:
        window.pop(0)
    window.append(torch.tensor(features))

    if len(window) < WINDOW_SIZE:
        return 0, [1.0, 0.0]

    predicted_class, probabilities = predict_single_input(lstm_model, window)
    return predicted_class, probabilities

def discriminator(frame, fire_detector_state):
    start_time = time.time()

    # Parallelize the fire detection and anomaly detection tasks using multiprocessing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_fire = executor.submit(detect_fire_smoke, frame, fire_model)
        future_anomaly = executor.submit(anomaly_detector, frame)

        fire_result = future_fire.result()
        anomaly_result, prob = future_anomaly.result()

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Time taken for frame processing: {processing_time:.4f} seconds")

    if fire_result == 1:
        print(f"Anomaly detected: Fire or Smoke detected!")
        return fire_result

    print(f"Anomaly detection result: {anomaly_result}, Probabilities: {prob}")
    return anomaly_result, prob
#ignore