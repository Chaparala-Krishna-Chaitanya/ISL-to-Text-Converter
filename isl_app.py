import numpy as np
import cv2
import os
import sys
import time
import operator
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
import enchant
import tensorflow as tf
from tensorflow import keras
import json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

def create_main_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(96, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(27, activation='softmax')
    ])
    return model

def create_specialized_model(output_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(96, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_units, activation='softmax')
    ])
    return model

class Application:
    def __init__(self):
        self.spell = enchant.Dict("en_US")
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        # Load main model
        try:
            self.loaded_model = create_main_model()
            self.loaded_model.load_weights("Models/model_new.h5")
            print("Main model loaded successfully")
        except Exception as e:
            print(f"Error loading main model: {str(e)}")
            self.loaded_model = None
        
        # Load specialized models
        try:
            self.loaded_model_dru = create_specialized_model(3)  # D, R, U
            self.loaded_model_dru.load_weights("Models/model-bw_dru.h5")
            print("D/R/U model loaded successfully")
        except Exception as e:
            print(f"Error loading D/R/U model: {str(e)}")
            self.loaded_model_dru = None
        
        try:
            self.loaded_model_tkdi = create_specialized_model(4)  # D, I, K, T
            self.loaded_model_tkdi.load_weights("Models/model-bw_tkdi.h5")
            print("D/I/K/T model loaded successfully")
        except Exception as e:
            print(f"Error loading D/I/K/T model: {str(e)}")
            self.loaded_model_tkdi = None
        
        try:
            self.loaded_model_smn = create_specialized_model(3)  # M, N, S
            self.loaded_model_smn.load_weights("Models/model-bw_smn.h5")
            print("M/N/S model loaded successfully")
        except Exception as e:
            print(f"Error loading M/N/S model: {str(e)}")
            self.loaded_model_smn = None
        
        # Initialize character tracking
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        
        # Time tracking for predictions
        self.prediction_start_time = None
        self.current_prediction = None
        self.last_added_symbol = None
        
        print("Models loaded from disk")
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")
        
        # Main video panel
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)
        
        # Processed image panel
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)
        
        # Title
        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))
        
        # Current Symbol
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)
        
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))
        
        # Word
        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)
        
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))
        
        # Sentence
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)
        
        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))
        
        # Suggestions
        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))
        
        # Suggestion buttons
        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)
        
        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)
        
        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)
        
        # Initialize text variables
        self.str = ""
        self.word = " "  # Initialize with a space instead of empty string
        self.current_symbol = "Empty"
        self.photo = "Empty"
        
        # Start video loop
        self.video_loop()
    
    def video_loop(self):
        ok, frame = self.vs.read()
        
        if ok:
            cv2image = cv2.flip(frame, 1)
            
            # Define ROI
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            
            # Update main video panel
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
            # Process ROI
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Predict gesture
            self.predict(res)
            
            # Update processed image panel
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            
            # Update text displays
            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))
            
            # Update suggestions - check for empty word first
            if self.word.strip():  # Only get suggestions if word is not empty or just spaces
                try:
                    predicts = self.spell.suggest(self.word.strip())
                    
                    if len(predicts) > 0:
                        self.bt1.config(text=predicts[0], font=("Courier", 20))
                    else:
                        self.bt1.config(text="")
                        
                    if len(predicts) > 1:
                        self.bt2.config(text=predicts[1], font=("Courier", 20))
                    else:
                        self.bt2.config(text="")
                        
                    if len(predicts) > 2:
                        self.bt3.config(text=predicts[2], font=("Courier", 20))
                    else:
                        self.bt3.config(text="")
                except Exception as e:
                    print(f"Error getting suggestions: {str(e)}")
                    self.bt1.config(text="")
                    self.bt2.config(text="")
                    self.bt3.config(text="")
            else:
                # Clear suggestion buttons if word is empty
                self.bt1.config(text="")
                self.bt2.config(text="")
                self.bt3.config(text="")
        
        self.root.after(5, self.video_loop)
    
    def predict(self, test_image):
        if self.loaded_model is None:
            return
            
        # Resize image
        test_image = cv2.resize(test_image, (128, 128))
        
        # Get predictions from all models
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1), verbose=0) if self.loaded_model_dru is not None else None
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1), verbose=0) if self.loaded_model_tkdi is not None else None
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1), verbose=0) if self.loaded_model_smn is not None else None
        
        # Layer 1: Main model prediction
        prediction = {}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        
        # Layer 2: Specialized model predictions
        if result_dru is not None and self.current_symbol in ['D', 'R', 'U']:
            prediction = {}
            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]
        
        if result_tkdi is not None and self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {}
            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]
        
        if result_smn is not None and self.current_symbol in ['M', 'N', 'S']:
            prediction1 = {}
            prediction1['M'] = result_smn[0][0]
            prediction1['N'] = result_smn[0][1]
            prediction1['S'] = result_smn[0][2]
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            
            if prediction1[0][0] == 'S':
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]
        
        # Time-based prediction tracking
        current_time = time.time()
        
        # If this is a new prediction or a different prediction than the last one
        if self.current_prediction != self.current_symbol:
            self.current_prediction = self.current_symbol
            self.prediction_start_time = current_time
            # Reset the counter for character repetition
            self.character_ready = False
        else:
            # Same prediction continues
            elapsed_time = current_time - self.prediction_start_time
            
            # If 'blank' (space) is predicted for more than 3 seconds
            if self.current_symbol == 'blank' and elapsed_time > 3.0 and self.word.strip():
                if self.str:
                    self.str += " "
                self.str += self.word.strip()
                self.word = " "  # Reset to a single space instead of empty string
                self.prediction_start_time = current_time  # Reset timer
                self.last_added_symbol = None
                self.character_ready = False
            
            # If any letter is predicted for more than 2 seconds
            elif self.current_symbol != 'blank' and elapsed_time > 2.0:
                # If we've waited long enough, add the character
                if not self.character_ready:
                    self.word += self.current_symbol
                    self.character_ready = True  # Mark that we've added this instance
                    # Don't update last_added_symbol - that would prevent repetition
                
                # If we continue showing the same symbol for another 2 seconds after adding it once
                elif elapsed_time > 4.0:  # Original 2 seconds + 2 more seconds
                    self.word += self.current_symbol  # Add the character again
                    self.prediction_start_time = current_time  # Reset timer for next potential repeat
                    self.character_ready = False  # Reset ready state for next cycle
    
    def action1(self):
        if self.word.strip():  # Check if word is not empty
            try:
                predicts = self.spell.suggest(self.word.strip())
                if len(predicts) > 0:
                    if self.str:
                        self.str += " "
                    self.str += predicts[0]
                    self.word = " "  # Reset to space instead of empty string
            except Exception as e:
                print(f"Error in action1: {str(e)}")
    
    def action2(self):
        if self.word.strip():  # Check if word is not empty
            try:
                predicts = self.spell.suggest(self.word.strip())
                if len(predicts) > 1:
                    if self.str:
                        self.str += " "
                    self.str += predicts[1]
                    self.word = " "  # Reset to space instead of empty string
            except Exception as e:
                print(f"Error in action2: {str(e)}")
    
    def action3(self):
        if self.word.strip():  # Check if word is not empty
            try:
                predicts = self.spell.suggest(self.word.strip())
                if len(predicts) > 2:
                    if self.str:
                        self.str += " "
                    self.str += predicts[2]
                    self.word = " "  # Reset to space instead of empty string
            except Exception as e:
                print(f"Error in action3: {str(e)}")
    
    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Application...")
    (Application()).root.mainloop()