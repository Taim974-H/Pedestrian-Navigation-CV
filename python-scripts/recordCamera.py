import pyrealsense2 as rs
import numpy as np
import cv2
import time
from datetime import datetime

class RealSenseCamera:
    def __init__(self):
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable RGB stream
        self.width, self.height = 640, 480
        self.fps = 30
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # Start the pipeline
        self.pipeline.start(self.config)
        
        # Wait for camera to warm up
        time.sleep(2)
        
        # Video writer initialization
        self.video_writer = None
        self.is_recording = False
        
    def start_recording(self):
        """Start recording video"""
        if not self.is_recording:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'recording_{timestamp}.mp4'
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            self.is_recording = True
            print(f"Started recording: {filename}")
            
    def stop_recording(self):
        """Stop recording video"""
        if self.is_recording:
            self.video_writer.release()
            self.is_recording = False
            print("Recording stopped")
    
    def process_frame(self, frame):
        # Implement image processing algorithms here
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        return {
            'original': frame,
            'gray': gray,
            'blurred': blurred,
            'edges': edges
        }
    
    def run(self):
        try:
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert frame to numpy array
                frame = np.asanyarray(color_frame.get_data())
                
                # Process frame
                processed_frames = self.process_frame(frame)
                
                # Record frame if recording is active
                if self.is_recording:
                    self.video_writer.write(frame)
                
                # Add recording indicator
                if self.is_recording:
                    cv2.putText(
                        processed_frames['original'],
                        "REC",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
                
                # Display results
                cv2.imshow('Original', processed_frames['original'])
                cv2.imshow('Grayscale', processed_frames['gray'])
                cv2.imshow('Blurred', processed_frames['blurred'])
                cv2.imshow('Edges', processed_frames['edges'])
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('r'):  # Start/stop recording
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                    
        finally:
            # Clean up
            if self.is_recording:
                self.stop_recording()
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    camera = RealSenseCamera()
    camera.run()