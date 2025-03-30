import tensorflow as tf
import numpy as np
import os

def main():
    # Load the model
    model_path = os.path.join('Models', 'isl_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("\nModel Summary:")
    model.summary()
    
    # Create a test input
    test_input = np.random.uniform(0, 1, (1, 128, 128, 1)).astype(np.float32)
    
    # Get predictions
    predictions = model.predict(test_input, verbose=1)
    
    print("\nPrediction shape:", predictions.shape)
    print("Prediction sum:", np.sum(predictions))  # Should be close to 1.0
    print("Max probability:", np.max(predictions))
    print("Min probability:", np.min(predictions))
    
    # Check if probabilities sum to 1 (softmax output)
    print("\nProbabilities sum to 1?", np.allclose(np.sum(predictions), 1.0))
    
    # Print all class probabilities
    print("\nAll class probabilities:")
    for i, prob in enumerate(predictions[0]):
        print(f"Class {i}: {prob:.6f}")

if __name__ == "__main__":
    main() 