import pyrealsense2 as rs
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class HueRange:
    """Defines the hue range for road and pavement detection"""
    PAVEMENT_LOWER = 10
    PAVEMENT_UPPER = 40

    ROAD_LOWER_1 = 0
    ROAD_LOWER_2 = 40

    ROAD_UPPER_1 = 10
    ROAD_UPPER_2 = 180

class RealSenseRoadDetector:
    def __init__(self):
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable RGB stream
        self.width, self.height = 640, 480
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        
        # Start the pipeline
        self.pipeline.start(self.config)
        
        # Initialize processing parameters
        self.kernel = np.ones((8, 8), np.uint8)
        
    def create_roi_mask(self, height_fraction: float = 0.3) -> np.ndarray:
        roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        polygon = np.array([
            [(0, self.height),
             (0, int(self.height * height_fraction)),
             (self.width, int(self.height * height_fraction)),
             (self.width, self.height)]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, polygon, 255)
        return roi_mask

    def create_road_mask(self, hue: np.ndarray) -> np.ndarray:
        road_mask_1 = cv2.inRange(hue, HueRange.ROAD_LOWER_1, HueRange.ROAD_UPPER_1)
        road_mask_2 = cv2.inRange(hue, HueRange.ROAD_LOWER_2, HueRange.ROAD_UPPER_2)
        return cv2.bitwise_or(road_mask_1, road_mask_2)

    def create_pavement_mask(self, hue: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        pavement_mask = cv2.inRange(hue, HueRange.PAVEMENT_LOWER, HueRange.PAVEMENT_UPPER)
        pavement_mask = cv2.bitwise_and(pavement_mask, cv2.bitwise_not(road_mask))
        return pavement_mask

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return cv2.morphologyEx(refined, cv2.MORPH_CLOSE, self.kernel)

    def check_center_region(self, road_mask: np.ndarray, pavement_mask: np.ndarray, 
                          roi_height: int = 50, roi_width: int = 100) -> str:
        x_center = self.width // 2
        y_bottom = self.height - roi_height
        roi = pavement_mask[y_bottom:self.height, x_center - roi_width // 2:x_center + roi_width // 2]

        pavement_pixels = cv2.countNonZero(roi)
        total_pixels = roi.size
        pavement_ratio = pavement_pixels / total_pixels

        return 'pavement' if pavement_ratio > 0.5 else 'road'

    def draw_detection_box(self, image: np.ndarray, result: str, 
                         roi_height: int = 50, roi_width: int = 100) -> np.ndarray:
        overlay_image = image.copy()
        x_center = self.width // 2
        y_bottom = self.height - roi_height
        
        # Set color based on detection result
        box_color = (0, 255, 0) if result == 'pavement' else (0, 0, 255)
        
        # Draw detection box
        cv2.rectangle(overlay_image,
                     (x_center - roi_width // 2, y_bottom),
                     (x_center + roi_width // 2, self.height),
                     box_color, 2)
        
        # Add label
        label = f'Detected: {result.capitalize()}'
        cv2.putText(overlay_image, label, 
                   (x_center - roi_width // 2, y_bottom - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        return overlay_image

    def process_frame(self, frame: np.ndarray) -> tuple:
        # Create ROI mask
        roi_mask = self.create_roi_mask()
        
        # Extract hue channel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        
        # Apply ROI mask to hue channel
        masked_hue = cv2.bitwise_and(hue, hue, mask=roi_mask)
        
        # Create and refine road mask
        road_mask = self.create_road_mask(masked_hue)
        road_mask = self.refine_mask(road_mask)
        
        # Create and refine pavement mask
        pavement_mask = self.create_pavement_mask(masked_hue, road_mask)
        pavement_mask = self.refine_mask(pavement_mask)
        
        # Check center region
        result = self.check_center_region(road_mask, pavement_mask)
        
        # Draw detection box
        result_image = self.draw_detection_box(frame, result)
        
        return result_image, result, road_mask, pavement_mask

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
                result_image, detection_result, road_mask, pavement_mask = self.process_frame(frame)
                
                # Display results
                cv2.imshow('Road Detection', result_image)
                # cv2.imshow('Road Mask', road_mask)
                # cv2.imshow('Pavement Mask', pavement_mask)
                
                # Print detection result
                print(f"Current detection: {detection_result}")
                
                # Break loop with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Clean up
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealSenseRoadDetector()
    detector.run()