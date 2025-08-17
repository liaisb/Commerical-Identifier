import cv2
import numpy as np
import time
import subprocess
import os
from datetime import datetime
import json

class CommercialDetector:
    def __init__(self):
        # Load configuration
        self.load_config()
        
        # Initialize model
        self.model = cv2.dnn.readNetFromONNX('commercial_detector.onnx')
        print("Model loaded successfully")
        
        # Initialize video capture
        self.setup_camera()
        
        # Initialize audio
        self.setup_audio()
        
        # State tracking
        self.current_state = 'game'
        self.last_switch_time = time.time()
        self.min_switch_interval = 3  # seconds
    
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            'game_volume': 80,
            'commercial_volume': 30,
            'confidence_threshold': 0.7,
            'camera_index': 0,
            'debug_mode': True
        }
        
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = default_config
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=4)
    
    def setup_camera(self):
        """Initialize video capture"""
        self.cap = cv2.VideoCapture(self.config['camera_index'])
        if not self.cap.isOpened():
            raise Exception("Failed to open camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def setup_audio(self):
        """Initialize audio devices"""
        try:
            # Set up TV audio (HDMI/Line-in)
            subprocess.run(['amixer', 'sset', 'Master', f"{self.config['game_volume']}%"])
            # Set up music audio (separate input)
            subprocess.run(['amixer', 'sset', 'Line', '0%'])
        except Exception as e:
            print(f"Audio setup error: {e}")
    
    def switch_audio(self, to_state):
        """Switch audio between game and commercial"""
        if to_state != self.current_state:
            if to_state == 'commercial':
                # Fade out game audio, fade in music
                subprocess.run(['amixer', 'sset', 'Master', f"{self.config['commercial_volume']}%"])
                subprocess.run(['amixer', 'sset', 'Line', '100%'])
            else:
                # Fade out music, fade in game audio
                subprocess.run(['amixer', 'sset', 'Master', f"{self.config['game_volume']}%"])
                subprocess.run(['amixer', 'sset', 'Line', '0%'])
            
            self.current_state = to_state
            self.log_switch(to_state)
    
    def log_switch(self, switch_type):
        """Log audio switches"""
        with open('switches.log', 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: Switched to {switch_type}\n")
    
    def preprocess_frame(self, frame):
        """Prepare frame for model inference"""
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (224, 224), 
            swapRB=True, 
            crop=True
        )
        return blob
    
    def predict(self, frame):
        """Run inference on a frame"""
        blob = self.preprocess_frame(frame)
        self.model.setInput(blob)
        output = self.model.forward()
        
        # Get probability of commercial
        commercial_prob = output[0][1]
        is_commercial = commercial_prob > self.config['confidence_threshold']
        
        return is_commercial, commercial_prob
    
    def run(self):
        """Main detection loop"""
        print("Commercial Detector running...")
        print("Press 'Q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading from camera")
                    break
                
                # Run detection
                is_commercial, confidence = self.predict(frame)
                
                # Check if enough time has passed since last switch
                time_since_switch = time.time() - self.last_switch_time
                
                if time_since_switch > self.min_switch_interval:
                    # Switch audio if needed
                    if is_commercial:
                        self.switch_audio('commercial')
                    else:
                        self.switch_audio('game')
                    
                    if is_commercial != (self.current_state == 'commercial'):
                        self.last_switch_time = time.time()
                
                # Display debug info if enabled
                if self.config['debug_mode']:
                    status = "COMMERCIAL" if is_commercial else "GAME"
                    cv2.putText(frame, f"{status} ({confidence:.2f})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Audio: {self.current_state.upper()}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
                    cv2.imshow('Commercial Detector', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Add small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CommercialDetector()
    detector.run()
