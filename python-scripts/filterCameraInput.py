import pyrealsense2 as rs
import numpy as np
import cv2
import time

class RealSenseCamera:
    def __init__(self):
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable RGB stream
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start the pipeline
        self.pipeline.start(self.config)
        
        # Wait for camera to warm up
        time.sleep(2)
        
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
                
                # Display results
                cv2.imshow('Original', processed_frames['original'])
                cv2.imshow('Grayscale', processed_frames['gray'])
                cv2.imshow('Blurred', processed_frames['blurred'])
                cv2.imshow('Edges', processed_frames['edges'])
                
                # Break loop with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Clean up
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    camera = RealSenseCamera()
    camera.run()