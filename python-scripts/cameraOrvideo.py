import pyrealsense2 as rs
import numpy as np
import cv2
import time
from datetime import datetime

def process_video_frame(frame):
        """
        Process a single frame by applying various transformations such as:
        1. Grayscale conversion
        2. Gaussian blur
        3. Edge detection (Canny)

        Args:
            frame (numpy.ndarray): The input frame from the video.

        Returns:
            dict: A dictionary containing processed frames.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 100, 200)
        
        # You can add more transformations here (e.g., resizing, filtering, etc.)

        # Return all processed frames in a dictionary
        return {
            'original': frame,
            'gray': gray,
            'blurred': blurred,
            'edges': edges
        }

class VideoProcessor:
    def __init__(self, input_source="camera", fps=30, frame_type="edges"):
        """
        Initialize video processor with either camera or video file.
        input_source: "camera" for RealSense camera or path to video file
        fps: Optional FPS argument to override video file FPS
        """
        self.input_source = input_source
        self.width, self.height = 640, 480
        self.fps = fps
        self.frame_type = frame_type
        
        if input_source == "camera":
            # Configure RealSense pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.pipeline.start(self.config)
            time.sleep(2)  # Wait for camera to warm up
            self.video_capture = None
        else:
            # Open video file
            self.pipeline = None
            self.video_capture = cv2.VideoCapture(input_source)
            if not self.video_capture.isOpened():
                raise ValueError(f"Could not open video file: {input_source}")
            # Get video properties
            self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Override FPS from the file if needed
            if fps == 30:  # Default FPS is 30 unless otherwise specified
                self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        
        # Video writer initialization
        self.video_writer = None
        self.is_recording = False

    def start_recording(self):
        """Start recording video"""
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'recording_{self.frame_type}_{timestamp}.mp4'
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

    def get_frame(self):
        """Get next frame from either camera or video file"""
        if self.input_source == "camera":
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())
        else:
            ret, frame = self.video_capture.read()
            if not ret:
                return None
            return frame
    
    def run(self):
        try:
            while True:
                # Get frame from either camera or video
                frame = self.get_frame()

                if frame is None:
                    if self.input_source != "camera":
                        print("End of video file reached")
                    break

                # Process the frame
                processed_frames = process_video_frame(frame)

                # Determine what to display and record based on the frame_type
                if self.is_recording:
                    if self.frame_type == "original":
                        self.video_writer.write(processed_frames['original'])
                        cv2.putText(
                            processed_frames['original'],
                            "REC",
                            (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2
                        )
                    elif self.frame_type == "gray":
                        self.video_writer.write(processed_frames['gray'])
                        cv2.putText(
                            processed_frames['gray'],
                            "REC",
                            (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2
                        )
                    elif self.frame_type == "blurred":
                        self.video_writer.write(processed_frames['blurred'])
                        cv2.putText(
                            processed_frames['blurred'],
                            "REC",
                            (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2
                        )
                    elif self.frame_type == "edges":
                        edges_bgr = cv2.cvtColor(processed_frames['edges'], cv2.COLOR_GRAY2BGR)  # Convert edges to BGR format
                        self.video_writer.write(edges_bgr)  # Save the 3-channel edges frame
                        cv2.putText(
                            edges_bgr,
                            "REC",
                            (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2
                        )

                # Display the appropriate frame based on the frame_type
                if self.frame_type == "original":
                    cv2.imshow('Original', processed_frames['original'])
                elif self.frame_type == "gray":
                    cv2.imshow('Grayscale', processed_frames['gray'])
                elif self.frame_type == "blurred":
                    cv2.imshow('Blurred', processed_frames['blurred'])
                elif self.frame_type == "edges":
                    cv2.imshow('Edges', processed_frames['edges'])

                # Handle keyboard input
                key = cv2.waitKey(int(1000 / self.fps)) & 0xFF  # Control frame display time based on FPS
                if key == ord('q'):  # Quit
                    break
                elif key == ord('r'):  # Start/stop recording
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('p'):  # Pause (only for video file)
                    if self.input_source != "camera":
                        cv2.waitKey(0)  # Wait until any key is pressed

        finally:
            # Clean up
            if self.is_recording:
                self.stop_recording()
            if self.pipeline:
                self.pipeline.stop()
            if self.video_capture:
                self.video_capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # For camera input:
    # processor = VideoProcessor("camera")
    # import os
    # print(os.path.exists(r"C:\Users\taimo\Desktop\PedestrianNavigationCV-FAU\python-scripts\erlangen-pavement-1.mp4"))
    # For video file input:
    processor = VideoProcessor(r"C:\Users\taimo\Desktop\PedestrianNavigationCV-FAU\python-scripts\output_video_fixed.mp4",fps=30, frame_type="edges")  # Replace with your video file name
    processor.run()