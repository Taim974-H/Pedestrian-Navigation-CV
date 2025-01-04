import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(frame, radius=1, n_points=8):
    """Extract Local Binary Pattern (LBP) features from a grayscale image."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_frame, n_points, radius, method='uniform')
    return lbp

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame for consistency
        frame = cv2.resize(frame, (640, 480))
        
        # Extract LBP features
        lbp_features = extract_lbp_features(frame)
        
        # Convert frame to YCrCb color space and extract luminance
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        gray = ycrcb[:, :, 0]
        
        # Apply Sobel operator for gradient detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        # Normalize gradient magnitude for display
        magnitude_display = magnitude / np.max(magnitude + 1e-6)
        
        # Apply adaptive thresholding to magnitude
        magnitude_uint8 = cv2.convertScaleAbs(magnitude)
        adaptive_thresh = cv2.adaptiveThreshold(
            magnitude_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)
        
        # Find contours on the thresholded image
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw detected contours on the frame
        output_frame = frame.copy()
        cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)
        
        # Overlay LBP visualization
        lbp_vis = (lbp_features / lbp_features.max() * 255).astype(np.uint8)
        lbp_color = cv2.applyColorMap(lbp_vis, cv2.COLORMAP_JET)
        
        # Display results
        cv2.imshow('Original Frame', frame)
        cv2.imshow('LBP Features', lbp_color)
        cv2.imshow('Gradient Magnitude', magnitude_display)
        cv2.imshow('Contours', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = r'C:\Users\taimo\Desktop\PedestrianNavigationCV-FAU\research-papers-implementation\vision-based-pavement-marking-detection-and-condition-assessment-a-case-study\processed_video.mp4'
extract_features(video_path)
