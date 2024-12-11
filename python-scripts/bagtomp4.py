import pyrealsense2 as rs
import cv2
import numpy as np
import time

# File paths
input_bag_file = r"C:\Users\taimo\Desktop\PedestrianNavigationCV-FAU\python-scripts\20241202_140240_(2)[1].bag"
output_video_file = "output_video.mp4"

# Configure RealSense pipeline to read .bag file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(input_bag_file)

# Start streaming
pipeline.start(config)

# Get the frame rate
profile = pipeline.get_active_profile()
device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False)

# Set desired FPS
color_stream = profile.get_stream(rs.stream.color)
fps = int(color_stream.fps())
# fps = 15
frame_interval = 1.0 / fps  # Time between frames in seconds
width = profile.get_stream(rs.stream.color).as_video_stream_profile().width()
height = profile.get_stream(rs.stream.color).as_video_stream_profile().height()

# Set up OpenCV VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

# Frame timing control
last_time = time.time()

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()

        # Get the color frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Write the frame to the video file
        out.write(color_image)

        # display the frame (for debugging)
        cv2.imshow('Color Frame', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # # Frame interval control
        current_time = time.time()
        elapsed_time = current_time - last_time
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)
        last_time = time.time()

except RuntimeError:
    print("End of bag file reached.")

finally:
    # Release resources
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
