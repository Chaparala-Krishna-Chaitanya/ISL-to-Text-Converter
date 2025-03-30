import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class VideoCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Capture App")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Create control buttons
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        # Initialize video capture
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)  # Use default camera (0)
        if self.cap.isOpened():
            self.timer.start(30)  # Update every 30ms (approximately 33 fps)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            print("Error: Could not open camera")
    
    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.video_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert frame to QImage
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Display image
            self.video_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoCaptureApp()
    window.show()
    sys.exit(app.exec_())
