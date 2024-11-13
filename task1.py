import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class HueRange:
    """Defines the hue range for road and pavement detection"""
    PAVEMENT_LOWER = 10
    PAVEMENT_UPPER = 40
    ROAD_LOWER_1 = 0
    ROAD_UPPER_1 = 10
    ROAD_LOWER_2 = 40
    ROAD_UPPER_2 = 180

class RoadSegmentation:
    def __init__(self, image_path: str, resize_dims: tuple = (640, 480)):
        """
        Initialize the road segmentation processor.
        
        Args:
            image_path (str): Path to the input image
            resize_dims (tuple): Dimensions for resizing the image (width, height)
        """
        self.image = cv2.imread(image_path)
        if resize_dims:
            self.image = cv2.resize(self.image, resize_dims)
        self.height, self.width = self.image.shape[:2]
        self.kernel = np.ones((8, 8), np.uint8)

    def create_roi_mask(self, height_fraction: float = 0.3) -> np.ndarray:
        """
        Create a Region of Interest (ROI) mask for the lower part of the image.
        
        Args:
            height_fraction (float): Fraction of image height to include in ROI
        
        Returns:
            np.ndarray: Binary mask for ROI
        """
        roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        polygon = np.array([
            [(0, self.height),
             (0, int(self.height * height_fraction)),
             (self.width, int(self.height * height_fraction)),
             (self.width, self.height)]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, polygon, 255)
        return roi_mask

    def extract_hue_channel(self) -> np.ndarray:
        """
        Extract the hue channel from the image.
        
        Returns:
            np.ndarray: Hue channel
        """
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 0]

    def plot_hue_histogram(self, hue: np.ndarray) -> None:
        """
        Plot histogram of hue values.
        
        Args:
            hue (np.ndarray): Hue channel data
        """
        hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
        plt.figure(figsize=(10, 5))
        plt.plot(hist)
        plt.title('Hue Histogram')
        plt.xlabel('Hue Value')
        plt.ylabel('Frequency')
        plt.show()

    def create_road_mask(self, hue: np.ndarray) -> np.ndarray:
        """
        Create a binary mask for road areas.
        
        Args:
            hue (np.ndarray): Hue channel data
        
        Returns:
            np.ndarray: Binary mask for road areas
        """
        road_mask_1 = cv2.inRange(hue, HueRange.ROAD_LOWER_1, HueRange.ROAD_UPPER_1)
        road_mask_2 = cv2.inRange(hue, HueRange.ROAD_LOWER_2, HueRange.ROAD_UPPER_2)
        return cv2.bitwise_or(road_mask_1, road_mask_2)

    def create_pavement_mask(self, hue: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        pavement_mask = cv2.inRange(hue, HueRange.PAVEMENT_LOWER, HueRange.PAVEMENT_UPPER)
        pavement_mask = cv2.bitwise_and(pavement_mask, cv2.bitwise_not(road_mask))
        return pavement_mask

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to refine a mask.
        
        Args:
            mask (np.ndarray): Input binary mask
        
        Returns:
            np.ndarray: Refined binary mask
        """
        refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return cv2.morphologyEx(refined, cv2.MORPH_CLOSE, self.kernel)

    def segment_image(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to segment the original image.
        
        Args:
            mask (np.ndarray): Binary mask to apply
        
        Returns:
            np.ndarray: Segmented image
        """
        return cv2.bitwise_and(self.image, self.image, mask=mask)
    
    def check_center_region(self, road_mask: np.ndarray, pavement_mask: np.ndarray, roi_height: int = 50, roi_width: int = 100) -> str:
        """
        Check whether the center-bottom part of the image is within the pavement or road region.

        Args:
            road_mask (np.ndarray): Binary mask for the road region.
            pavement_mask (np.ndarray): Binary mask for the pavement region.
            roi_height (int): Height of the ROI rectangle.
            roi_width (int): Width of the ROI rectangle.

        Returns:
            str: 'pavement' if the region is mostly pavement, 'road' otherwise.
        """
        # Define the center-bottom ROI
        x_center = self.width // 2
        y_bottom = self.height - roi_height
        roi = pavement_mask[y_bottom:self.height, x_center - roi_width // 2:x_center + roi_width // 2]

        # Calculate the percentage of pavement pixels in the ROI
        pavement_pixels = cv2.countNonZero(roi)
        total_pixels = roi.size
        pavement_ratio = pavement_pixels / total_pixels

        # Determine if the region is pavement or road based on the ratio
        if pavement_ratio > 0.5:
            return 'pavement'
        else:
            return 'road'
        
    def draw_pavement_box(self, result: str, direction: str, 
                     roi_height: int = 50, roi_width: int = 100) -> np.ndarray:
        """
        Draw a bounding box indicating the detected pavement region and direction arrow.
        
        Args:
            result (str): 'pavement' or 'road' based on center region detection
            direction (str): 'left', 'right', or 'none' for movement direction
            roi_height (int): Height of the ROI rectangle
            roi_width (int): Width of the ROI rectangle
        
        Returns:
            np.ndarray: Image with box and arrow overlay
        """
        overlay_image = self.image.copy()
        x_center = self.width // 2
        y_bottom = self.height - roi_height
        
        # Set colors
        box_color = (0, 255, 0) if result == 'pavement' else (0, 0, 255)
        arrow_color = (0, 0, 255)
        
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
        
        # Draw direction arrow only if needed
        if direction != 'none' and result == 'road':
            arrow_thickness = 2
            arrow_size = 20
            
            if direction == 'left':
                start_point = (x_center + roi_width // 2 + 10, y_bottom + roi_height // 2)
                end_point = (start_point[0] - arrow_size, start_point[1])
                cv2.arrowedLine(overlay_image, start_point, end_point, 
                            arrow_color, arrow_thickness, tipLength=0.5)
                cv2.putText(overlay_image, "Move Left", 
                        (start_point[0] - 90, start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
            
            elif direction == 'right':
                start_point = (x_center - roi_width // 2 - 10, y_bottom + roi_height // 2)
                end_point = (start_point[0] + arrow_size, start_point[1])
                cv2.arrowedLine(overlay_image, start_point, end_point, 
                            arrow_color, arrow_thickness, tipLength=0.5)
                cv2.putText(overlay_image, "Move Right", 
                        (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
        
        return overlay_image


    def detect_pavement_direction(self, road_mask: np.ndarray, pavement_mask: np.ndarray, 
                            roi_height: int = 50, roi_width: int = 100) -> str:
        """
        Determine whether the road region's center-bottom should go left or right to find pavement.
        
        Args:
            road_mask (np.ndarray): Binary mask for the road region
            pavement_mask (np.ndarray): Binary mask for the pavement region
            roi_height (int): Height of the ROI rectangle
            roi_width (int): Width of the ROI rectangle
        
        Returns:
            str: 'left', 'right', or 'none' based on direction to reach pavement
        """
        x_center = self.width // 2
        y_bottom = self.height - roi_height
        
        # Define the center region
        center_x_start = x_center - roi_width // 2
        center_x_end = x_center + roi_width // 2
        center_region = pavement_mask[y_bottom:self.height, center_x_start:center_x_end]
        
        # Check if center is already in pavement
        center_pavement_ratio = cv2.countNonZero(center_region) / center_region.size
        if center_pavement_ratio > 0.5:  # If more than 50% is pavement
            return 'none'
        
        # Define wider regions to check for pavement presence
        search_width = roi_width * 2  # Wider search area
        
        # Left region
        left_x_start = max(0, center_x_start - search_width)
        left_x_end = center_x_start
        left_region = pavement_mask[y_bottom:self.height, left_x_start:left_x_end]
        
        # Right region
        right_x_start = center_x_end
        right_x_end = min(self.width, center_x_end + search_width)
        right_region = pavement_mask[y_bottom:self.height, right_x_start:right_x_end]
        
        # Calculate pavement density in each region
        left_density = cv2.countNonZero(left_region) / left_region.size
        right_density = cv2.countNonZero(right_region) / right_region.size
        
        # Set threshold for significant difference
        threshold = 0.1
        
        if abs(left_density - right_density) < threshold:
            return 'none'  # No clear direction if difference is too small
        elif left_density > right_density:
            return 'left'
        else:
            return 'right'


    def _display_results(self, road_mask: np.ndarray, pavement_mask: np.ndarray,
                            road_segment: np.ndarray, pavement_segment: np.ndarray) -> None:
            """
            Display processing results.
            
            Args:
                road_mask (np.ndarray): Road binary mask
                pavement_mask (np.ndarray): Pavement binary mask
                road_segment (np.ndarray): Segmented road image
                pavement_segment (np.ndarray): Segmented pavement image
            """
            cv2.imshow('Road Mask', road_mask)
            cv2.imshow('Pavement Mask', pavement_mask)
            cv2.imshow('Road Segment', road_segment)
            cv2.imshow('Pavement Segment', pavement_segment)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def process(self, show_results: bool = True) -> tuple:
        """
        Process the image and perform road/pavement segmentation.
        
        Args:
            show_results (bool): Whether to display results
        
        Returns:
            tuple: (road_segment, pavement_segment, result_image, direction)
        """
        # Create masks and segments
        roi_mask = self.create_roi_mask()
        hue = self.extract_hue_channel()
        masked_hue = cv2.bitwise_and(hue, hue, mask=roi_mask)
        
        road_mask = self.create_road_mask(masked_hue)
        road_mask = self.refine_mask(road_mask)
        
        pavement_mask = self.create_pavement_mask(masked_hue, road_mask)
        pavement_mask = self.refine_mask(pavement_mask)
        
        road_segment = self.segment_image(road_mask)
        pavement_segment = self.segment_image(pavement_mask)
        
        # Check center region
        center_result = self.check_center_region(road_mask, pavement_mask)
        
        # Only detect direction if we're on road
        direction = 'none'
        if center_result == 'road':
            direction = self.detect_pavement_direction(road_mask, pavement_mask)
        
        # Create result image
        result_image = self.draw_pavement_box(center_result, direction)
        
        if show_results:
            self._display_results(road_mask, pavement_mask, 
                                road_segment, pavement_segment)
            cv2.imshow('Result Image', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return road_segment, pavement_segment, result_image, direction



if __name__ == "__main__":
    segmenter = RoadSegmentation('testimage2.jpg')
    road_segment, pavement_segment,result_image, direction = segmenter.process()