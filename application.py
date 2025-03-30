import sys
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QHBoxLayout, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from string import ascii_uppercase
import enchant

# Suppress MediaPipe logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)

class ISLRecognizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ISL Gesture Recognition")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create top section with video and processed image
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Left side - Video display
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Main video feed
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)
        
        # Processed image preview
        self.preview_label = QLabel("Processed Image")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(200, 200)
        left_layout.addWidget(self.preview_label)
        
        top_layout.addWidget(left_widget)
        
        # Right side - Recognition results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Current character
        self.char_label = QLabel("Current Character:")
        self.char_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.char_label)
        
        self.current_char = QLabel("-")
        self.current_char.setStyleSheet("font-size: 48px; font-weight: bold;")
        right_layout.addWidget(self.current_char)
        
        # Current word
        self.word_label = QLabel("Current Word:")
        self.word_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.word_label)
        
        self.current_word = QLabel("")
        self.current_word.setStyleSheet("font-size: 32px;")
        right_layout.addWidget(self.current_word)
        
        # Sentence
        self.sentence_label = QLabel("Sentence:")
        self.sentence_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.sentence_label)
        
        self.current_sentence = QLabel("")
        self.current_sentence.setStyleSheet("font-size: 24px;")
        self.current_sentence.setWordWrap(True)
        right_layout.addWidget(self.current_sentence)
        
        # Suggestions
        self.suggestions_label = QLabel("Suggestions:")
        self.suggestions_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.suggestions_label)
        
        # Suggestion buttons
        suggestions_layout = QGridLayout()
        self.suggestion_buttons = []
        for i in range(3):
            btn = QPushButton("")
            btn.setStyleSheet("font-size: 20px; padding: 10px;")
            btn.clicked.connect(lambda checked, x=i: self.apply_suggestion(x))
            suggestions_layout.addWidget(btn, 0, i)
            self.suggestion_buttons.append(btn)
        
        right_layout.addLayout(suggestions_layout)
        top_layout.addWidget(right_widget)
        main_layout.addWidget(top_widget)
        
        # Initialize video capture and processing
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load the models
        self.model = None
        try:
            # Get absolute paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, 'Models')  # Changed from '../Models' to 'Models'
            keras_path = os.path.join(models_dir, 'isl_model.keras')
            h5_path = os.path.join(models_dir, 'isl_model.h5')
            
            print(f"Attempting to load model from:\n.keras path: {keras_path}\n.h5 path: {h5_path}")
            
            # Create Models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            # Verify file existence
            if not os.path.exists(keras_path) and not os.path.exists(h5_path):
                raise FileNotFoundError("Neither .keras nor .h5 model files exist at the specified paths")
            
            # Try loading .keras format first
            if os.path.exists(keras_path):
                print("Found .keras file, attempting to load...")
                self.model = tf.keras.models.load_model(keras_path)
                print("Main model loaded successfully from .keras format")
            elif os.path.exists(h5_path):
                print("Found .h5 file, attempting to load...")
                self.model = tf.keras.models.load_model(h5_path)
                print("Main model loaded successfully from .h5 format")
            
            if self.model is None:
                raise Exception("Model loading failed for unknown reason")
                
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}\n\n"
            error_msg += f"Attempted paths:\n"
            error_msg += f".keras: {keras_path}\n"
            error_msg += f".h5: {h5_path}\n\n"
            error_msg += "Please ensure:\n"
            error_msg += "1. The model files exist in the Models directory\n"
            error_msg += "2. The files have proper read permissions\n"
            error_msg += "3. The model files are not corrupted"
            
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
        
        # Initialize spell checker
        self.spell = enchant.Dict("en_US")
        
        # Initialize character tracking
        self.char_count = {char: 0 for char in ascii_uppercase}
        self.char_count['blank'] = 0
        self.blank_flag = 0
        
        # Initialize text tracking
        self.current_symbol = ""
        self.word = ""
        self.sentence = ""
        
        # Start camera
        self.start_camera()
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            print("Error: Could not open camera")
    
    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.video_label.clear()
        self.preview_label.clear()
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Get the region of interest
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            x1 = int(0.5 * w)
            y1 = 10
            x2 = w - 10
            y2 = int(0.5 * w)
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0), 1)
            
            # Process hand region
            roi = frame[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
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
            
            # Predict gesture
            self.predict_gesture(binary)
            
            # Update main video display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Update processed image preview
            h, w = binary.shape
            qt_image = QImage(binary.data, w, h, w, QImage.Format_Grayscale8)
            self.preview_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def predict_gesture(self, image):
        if self.model is None:
            return
            
        # Preprocess image
        resized = cv2.resize(image, (128, 128))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=[0, -1])
        
        # Get prediction
        prediction = self.model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        # Update character counts
        if predicted_class == 26:  # blank
            self.char_count['blank'] += 1
            symbol = 'blank'
        else:
            symbol = ascii_uppercase[predicted_class]
            self.char_count[symbol] += 1
        
        # Process the predicted symbol
        self.process_symbol(symbol)
        
    def process_symbol(self, symbol):
        # Threshold for character detection
        DETECTION_THRESHOLD = 30  # Lowered from 60 to 30 for more responsiveness
        
        # Get prediction confidence
        prediction = self.model.predict(input_data, verbose=0)[0]
        confidence = prediction[predicted_class] * 100
        
        # Update the current symbol if count exceeds threshold and confidence is high enough
        if self.char_count[symbol] > DETECTION_THRESHOLD and confidence > 15:  # Added confidence threshold
            # Reset all counts
            self.char_count = {char: 0 for char in self.char_count}
            
            # Process the symbol
            if symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if self.word:
                        self.add_word_to_sentence()
            else:
                self.blank_flag = 0
                self.word += symbol
            
            # Update UI with confidence
            self.current_char.setText(f"{symbol if symbol != 'blank' else 'SPACE'} ({confidence:.1f}%)")
            self.current_word.setText(self.word)
            self.current_sentence.setText(self.sentence)
            
            # Update suggestions
            self.update_suggestions()
    
    def add_word_to_sentence(self):
        if self.word:
            self.sentence += " " + self.word
            self.word = ""
            self.current_word.setText("")
            self.current_sentence.setText(self.sentence)
    
    def update_suggestions(self):
        if not self.word:
            for btn in self.suggestion_buttons:
                btn.setText("")
            return
            
        suggestions = self.spell.suggest(self.word)
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                btn.setText(suggestions[i])
            else:
                btn.setText("")
    
    def apply_suggestion(self, index):
        suggestion = self.suggestion_buttons[index].text()
        if suggestion:
            self.sentence += " " + suggestion
            self.word = ""
            self.current_word.setText("")
            self.current_sentence.setText(self.sentence)
            self.update_suggestions()
    
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ISLRecognizer()
    window.show()
    sys.exit(app.exec_())
