import os
import torch
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch.nn.functional as F
from scipy.interpolate import interp1d

from model import STGCN 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe hand tracking setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Model parameters (ensure these match the trained model)
NUM_KEYPOINTS = 21
TARGET_T = 114  # Target number of frames
IN_FEATURES = 6
NUM_CLASSES = 4  # UPDRS score classes

# Load pre-trained model
def load_model(model_path):
    """
    Load pre-trained model with robust state dict handling.
    
    Args:
    - model_path: Path to the model checkpoint
    
    Returns:
    - Loaded and initialized model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with parameters from checkpoint or defaults
    model_params = checkpoint.get('model_params', {
        'gcn_hidden': 64,
        'lstm_hidden': 64,
        'dropout_rate': 0.4
    })
    
    # Create model with current architecture
    model = STGCN(
        in_features=IN_FEATURES,
        gcn_hidden=model_params['gcn_hidden'],
        lstm_hidden=model_params['lstm_hidden'],
        num_classes=NUM_CLASSES,
        num_nodes=NUM_KEYPOINTS,
        dropout_rate=model_params.get('dropout_rate', 0.3)
    ).to(device)
    
    # Prepare state dict for loading
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Create a new state dict to handle potential mismatches
    new_state_dict = {}
    for k, v in state_dict.items():
        # Handle potential renaming or shape changes
        if k in model.state_dict():
            # Check if shape matches
            if v.shape == model.state_dict()[k].shape:
                new_state_dict[k] = v
            else:
                # Special handling for batch norm layers
                if 'bn' in k:
                    # If it's a batch norm layer with mismatched shape
                    print(f"Adapting {k} to new shape")
                    
                    # For batch norm layers, we might need to expand or truncate
                    if len(v.shape) == 1:
                        # Assuming we need to repeat the values to match the new shape
                        new_shape = model.state_dict()[k].shape[0]
                        new_v = v.repeat(new_shape // v.shape[0] + 1)[:new_shape]
                        new_state_dict[k] = new_v
                else:
                    print(f"Skipping {k} due to shape mismatch")
    
    # Load the state dict with some flexibility
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except Exception as e:
        print(f"Partial state dict loading: {e}")
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    return model

def normalize_keypoints(data):
    """
    Robust keypoint normalization with handling for different input shapes.
    
    Args:
    - data: NumPy array of keypoints
    
    Returns:
    - Normalized keypoints
    """
    # Ensure data is at least 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Handle potential zero variance
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    
    # Avoid division by zero
    scale = data_max - data_min
    scale[scale == 0] = 1
    
    return (data - data_min) / scale

def interpolate_missing_frames(data, target_frames=114):
    """
    Interpolate missing frames with advanced handling.
    
    Args:
    - data: NumPy array of keypoints
    - target_frames: Desired number of frames
    
    Returns:
    - Interpolated keypoints
    """
    from scipy.interpolate import interp1d
    
    # Ensure data is 3D: [frames, num_keypoints, coordinates]
    if data.ndim == 2:
        data = data.reshape(data.shape[0], -1, 3)
    
    frames, num_keypoints, coords = data.shape
    
    # Initialize interpolated data
    interpolated_data = np.zeros((target_frames, num_keypoints, coords))
    
    # Interpolate each keypoint and coordinate
    for k in range(num_keypoints):
        for c in range(coords):
            # Find valid (non-zero) frames
            valid_indices = np.where(data[:, k, c] != 0)[0]
            
            if len(valid_indices) > 1:
                # Create interpolation function
                interp_func = interp1d(
                    valid_indices, 
                    data[valid_indices, k, c], 
                    kind='linear', 
                    fill_value='extrapolate'
                )
                
                # Generate interpolated values
                interpolated_data[:, k, c] = interp_func(np.linspace(0, frames-1, target_frames))
            
            elif len(valid_indices) == 1:
                # If only one valid point, repeat that point
                interpolated_data[:, k, c] = data[valid_indices[0], k, c]
    
    return interpolated_data

def extract_hand_keypoints(video_path, hand_side='right'):
    """Extract hand keypoints from video."""
    keypoints_list = []
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # MediaPipe hand tracking setup
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
        
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = hands.process(rgb_frame)
            
            # Extract keypoints if hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract keypoint coordinates
                    frame_keypoints = []
                    for landmark in hand_landmarks.landmark:
                        frame_keypoints.append([landmark.x, landmark.y, landmark.z])
                    
                    keypoints_list.append(frame_keypoints)
            
            frame_count += 1
    
    cap.release()
    
    # Convert to numpy array
    keypoints = np.array(keypoints_list)
    
    return keypoints

def compute_additional_features(keypoints):
    """
    Compute velocity and acceleration features.
    
    Args:
    - keypoints: Interpolated keypoints
    
    Returns:
    - Enhanced feature tensor
    """
    # Compute velocity
    velocity = np.diff(keypoints, axis=0, prepend=keypoints[0:1])
    
    # Compute acceleration
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    
    # Concatenate spatial, velocity, and acceleration features
    enhanced_features = np.concatenate([
        keypoints,      # Spatial coordinates
        velocity,       # Velocity 
        acceleration    # Acceleration
    ], axis=-1)
    
    return enhanced_features

def preprocess_keypoints(keypoints, target_frames=114, num_keypoints=21):
    """
    Comprehensive keypoint preprocessing pipeline.
    
    Args:
    - keypoints: Raw keypoint data
    - target_frames: Desired number of frames
    - num_keypoints: Expected number of keypoints
    
    Returns:
    - Preprocessed keypoint tensor
    """
    # Validate input
    if len(keypoints) == 0:
        raise ValueError("No keypoints detected")
    
    # Interpolate missing frames
    interpolated_keypoints = interpolate_missing_frames(keypoints, target_frames)
    
    # Normalize keypoints
    normalized_keypoints = normalize_keypoints(interpolated_keypoints)
    
    # Compute additional features
    enhanced_features = compute_additional_features(normalized_keypoints)
    
    # Reshape to match model input: [num_nodes, timesteps, features]
    processed_keypoints = enhanced_features.transpose(1, 0, 2)
    
    return processed_keypoints

def build_adjacency(num_nodes=21):
    """
    Create a more comprehensive adjacency matrix.
    
    Args:
    - num_nodes: Number of keypoints
    
    Returns:
    - Adjacency matrix
    """
    # Initialize adjacency matrix
    adjacency = np.zeros((num_nodes, num_nodes))
    
    # Define comprehensive hand keypoint connections
    connections = [
        # Thumb connections
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Inter-finger connections
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Add connections to adjacency matrix
    for (i, j) in connections:
        adjacency[i, j] = adjacency[j, i] = 1
    
    # Add self-connections
    np.fill_diagonal(adjacency, 1)
    
    return adjacency

@app.route('/')
def upload_page():
    """Render the upload page."""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict_updrs():
    """Predict UPDRS score from uploaded video."""
    # Check if video file is present
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    hand_side = request.form.get('hand_side', 'right')
    
    # Validate file
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded video
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)
    
    try:
        # Load pre-trained model (update path as needed)
        model = load_model('final_model.pth')
        
        # Convert inputs to tensors on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract keypoints
        keypoints = extract_hand_keypoints(video_path, hand_side)
        
        # Preprocess keypoints
        processed_keypoints = preprocess_keypoints(keypoints)
        
        # Build adjacency matrix
        adj = build_adjacency(num_nodes=21)
        
        # Convert to PyTorch tensors
        keypoints_tensor = torch.tensor(processed_keypoints, dtype=torch.float32).to(device)
        adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
        
        # Ensure correct tensor shape: [batch, nodes, timesteps, features]
        if keypoints_tensor.dim() == 3:
            keypoints_tensor = keypoints_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(keypoints_tensor, adj_tensor)
            probabilities = F.softmax(predictions, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Map predicted class to UPDRS score
        updrs_mapping = {
            0: 'Minimal or No Motor Symptoms',
            1: 'Mild Motor Symptoms',
            2: 'Moderate Motor Symptoms',
            3: 'Severe Motor Symptoms'
        }
        
        return jsonify({
            'updrs_score': predicted_class,
            'updrs_description': updrs_mapping[predicted_class],
            'confidence': probabilities.max().item()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Close all open file handles and attempt to remove the file
        import gc
        gc.collect()
        
        try:
            # Try to close the file handle
            if 'video_file' in locals():
                video_file.close()
            
            # Wait a moment and then try to remove
            import time
            time.sleep(0.1)
            os.remove(video_path)
        except PermissionError:
            print(f"Could not remove {video_path}. File may be in use.")
        except Exception as e:
            print(f"Error removing file: {e}")
            
# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True)