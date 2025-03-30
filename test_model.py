import cv2
import numpy as np
import tensorflow as tf
from string import ascii_uppercase
import os

def preprocess_image(image):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blur, 
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Resize to model input size (128x128)
    resized = cv2.resize(binary, (128, 128))
    
    # Normalize to [0,1] range
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    input_data = np.expand_dims(normalized, axis=[0, -1])
    
    return input_data, binary, blur

def main():
    # Load the model
    model_path = os.path.join('Models', 'isl_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Print model summary
    model.summary()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Predictions', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Debug', cv2.WINDOW_NORMAL)
    
    # Arrange windows
    cv2.moveWindow('Original', 0, 0)
    cv2.moveWindow('Gaussian Blur', 640, 0)
    cv2.moveWindow('Processed', 0, 480)
    cv2.moveWindow('Predictions', 640, 480)
    cv2.moveWindow('Debug', 1280, 0)
    
    # List of characters (A-Z + blank)
    characters = list(ascii_uppercase) + ['blank']
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Get ROI
        h, w = frame.shape[:2]
        x1 = int(0.5 * w)
        y1 = 10
        x2 = w - 10
        y2 = int(0.5 * w)
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0), 2)
        
        # Get hand region
        roi = frame[y1:y2, x1:x2]
        
        # Preprocess image
        input_data, binary, blur = preprocess_image(roi)
        
        # Create debug visualization
        debug_viz = np.zeros((600, 400, 3), dtype=np.uint8)
        
        # Add debug info
        debug_info = [
            f"Input shape: {input_data.shape}",
            f"Input min: {input_data.min():.3f}",
            f"Input max: {input_data.max():.3f}",
            f"Input mean: {input_data.mean():.3f}",
            f"Input std: {input_data.std():.3f}"
        ]
        
        for i, info in enumerate(debug_info):
            cv2.putText(debug_viz, info, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        
        # Show input image in debug window
        input_display = cv2.resize(input_data[0, :, :, 0], (300, 300))
        input_display = (input_display * 255).astype(np.uint8)
        debug_viz[200:500, 50:350] = cv2.cvtColor(input_display, cv2.COLOR_GRAY2BGR)
        
        # Get predictions
        predictions = model.predict(input_data, verbose=0)[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        
        # Create prediction visualization
        pred_viz = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Draw top 3 predictions with confidence bars
        for i, idx in enumerate(top_3_idx):
            char = characters[idx]
            conf = predictions[idx] * 100
            
            # Draw character and confidence
            text = f"{char}: {conf:.1f}%"
            cv2.putText(pred_viz, text, (20, 50 + i*100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # Draw confidence bar
            bar_width = int(conf * 4)  # Scale factor for visualization
            cv2.rectangle(pred_viz, (200, 30 + i*100), 
                         (200 + bar_width, 60 + i*100), (0,255,0), -1)
            
            # Add to debug info
            y_pos = 350 + i*30
            cv2.putText(debug_viz, f"Prob {char}: {predictions[idx]:.6f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        
        # Add titles to the windows
        title_frame = frame.copy()
        cv2.putText(title_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Convert blur to 3 channels for adding text
        blur_display = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        cv2.putText(blur_display, "Gaussian Blur", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Convert binary to 3 channels for adding text
        binary_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.putText(binary_display, "Binary Threshold", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Show the frames
        cv2.imshow('Original', title_frame)
        cv2.imshow('Gaussian Blur', blur_display)
        cv2.imshow('Processed', binary_display)
        cv2.imshow('Predictions', pred_viz)
        cv2.imshow('Debug', debug_viz)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 