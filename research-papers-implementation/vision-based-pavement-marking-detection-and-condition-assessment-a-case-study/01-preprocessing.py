import cv2
import numpy as np

def apply_smoothing(image):
    return cv2.medianBlur(image, 5)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    return cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)

def increase_sharpness(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def adjust_saturation(image, saturation_scale=1.0):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255).astype('uint8')
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def inverse_perspective_transform(image, src_points, dst_points):
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

def preprocess_video(video_path, output_video_path, src_points, dst_points):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use a compatible codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            smoothed_image = apply_smoothing(frame)
            adjusted_image = adjust_brightness_contrast(smoothed_image, brightness=-10, contrast=-10)
            sharp_image = increase_sharpness(adjusted_image)
            saturated_image = adjust_saturation(sharp_image, saturation_scale=1.2)
            transformed_image = inverse_perspective_transform(saturated_image, src_points, dst_points)
            transformed_image = cv2.resize(transformed_image, (width, height))  # Ensure consistent size
            out.write(transformed_image)
        except Exception as e:
            print(f"Error processing frame: {e}")
            break

    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = r"C:\Users\taimo\Desktop\PedestrianNavigationCV-FAU\research-papers-implementation\data\01-sample-video.mp4"
    output_video_path = r"C:\Users\taimo\Desktop\PedestrianNavigationCV-FAU\research-papers-implementation\vision-based-pavement-marking-detection-and-condition-assessment-a-case-study\processed_video.mp4"
    src_points = np.float32([[100, 100], [200, 100], [100, 200], [200, 200]]) 
    dst_points = np.float32([[100, 100], [200, 100], [100, 200], [200, 200]]) 

    preprocess_video(video_path, output_video_path, src_points, dst_points)
